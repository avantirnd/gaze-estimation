import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import mediapipe as mp

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze
import uniface
from uniface.constants import RetinaFaceWeights


class GazeEstimation:
    def __init__(
        self,
        model_name="mobilenetv2",
        weight="weights/mobilenetv2.pt",
        dataset="gaze360",
        device=None,
        detector="retinaface",           # "retinaface" or "blazeface"
        blaze_model_selection=1,         # 0: close-range, 1: full-range
        blaze_min_conf=0.5
    ):
        if dataset not in data_config:
            raise ValueError(f"Unknown dataset: {dataset}. Options: {list(data_config.keys())}")
        cfg = data_config[dataset]
        self.bins = cfg["bins"]
        self.binwidth = cfg["binwidth"]
        self.angle = cfg["angle"]

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx_tensor = torch.arange(self.bins, device=self.device, dtype=torch.float32)

        # --- face detector choice ---
        self.detector = detector.lower()
        if self.detector == "retinaface":
            self.face_detector = uniface.RetinaFace(RetinaFaceWeights.MNET_V2)
            self._detect_faces = self._detect_retinaface
        elif self.detector == "blazeface":
            self.mp_fd = mp.solutions.face_detection.FaceDetection(
                model_selection=blaze_model_selection,
                min_detection_confidence=blaze_min_conf
            )
            self._detect_faces = self._detect_blazeface
        else:
            raise ValueError("detector must be 'retinaface' or 'blazeface'")

        # --- gaze model ---
        self.model = get_model(model_name, self.bins, inference_mode=True).to(self.device).eval()
        sd = torch.load(weight, map_location=self.device)
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "net", "module")):
            for k in ("state_dict", "model", "net", "module"):
                if k in sd:
                    sd = sd[k]
                    break
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _preprocess(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = self.transform(rgb).unsqueeze(0).to(self.device)
        return t
    
    # ---- detectors ----
    def _detect_retinaface(self, frame_bgr):
        bboxes, _ = self.face_detector.detect(frame_bgr)
        return [tuple(map(int, box[:4])) for box in bboxes] if bboxes is not None else []

    def _detect_blazeface(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_fd.process(rgb)
        out = []
        if res.detections:
            for d in res.detections:
                rb = d.location_data.relative_bounding_box
                x1 = int(rb.xmin * w); y1 = int(rb.ymin * h)
                x2 = int((rb.xmin + rb.width) * w); y2 = int((rb.ymin + rb.height) * h)
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2))
        return out

    @torch.no_grad()
    def estimate_gaze(self, frame_bgr, max_faces=8):
        """
        Batched forward pass with optional max_faces cap.
        Returns: [{'bbox':..., 'pitch_rad':..., 'yaw_rad':...}, ...]
        """
        h, w = frame_bgr.shape[:2]
        bboxes = self._detect_faces(frame_bgr)
        if not bboxes:
            return []

        if max_faces is not None:
            bboxes = bboxes[:max_faces]

        crops, keep_bboxes = [], []
        for (x1, y1, x2, y2) in bboxes:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(self._preprocess(crop).squeeze(0))
            keep_bboxes.append((x1, y1, x2, y2))

        if not crops:
            return []

        batch = torch.stack(crops, dim=0).to(self.device)  # (N,C,H,W)
        pitch_logits, yaw_logits = self.model(batch)

        pitch_prob = F.softmax(pitch_logits, dim=1)
        yaw_prob   = F.softmax(yaw_logits,   dim=1)

        pitch_deg = (pitch_prob * self.idx_tensor).sum(dim=1) * self.binwidth - self.angle
        yaw_deg   = (yaw_prob   * self.idx_tensor).sum(dim=1) * self.binwidth - self.angle

        pitch_rad = torch.deg2rad(pitch_deg).tolist()
        yaw_rad   = torch.deg2rad(yaw_deg).tolist()

        return [
            {"bbox": bb, "pitch_rad": p, "yaw_rad": y}
            for bb, p, y in zip(keep_bboxes, pitch_rad, yaw_rad)
        ]

    def draw(self, frame_bgr, detections):
        """Draws in-place using utils.helpers.draw_bbox_gaze"""
        for det in detections:
            draw_bbox_gaze(frame_bgr, det["bbox"], det["pitch_rad"], det["yaw_rad"])
        return frame_bgr
