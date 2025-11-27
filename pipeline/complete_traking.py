import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2


# -------------------------------------------------------------------
# 1) MODULE RE-ID PREENTRAINE (ResNet50)
# -------------------------------------------------------------------
class PretrainedReIDModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        print(" Modèle Re-ID initialisé")

    def _load_model(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()  # extraire uniquement les features
        model = model.to(self.device)
        model.eval()
        return model

    def extract(self, frame, bbox):
        """Retourne le vecteur d'apparence normalisé."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0: return None

        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        tensor = self.transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).cpu().numpy().flatten()

        n = np.linalg.norm(feat)
        return feat / n if n > 0 else None


# -------------------------------------------------------------------
# 2) BOTSORT + RE-ID
# -------------------------------------------------------------------
class BotSortTracker:
    def __init__(self, track_thresh=0.4, track_buffer=30,
                 match_thresh=0.7, appearance_thresh=0.6):

        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.appearance_thresh = appearance_thresh

        self.tracked_tracks = []
        self.removed_tracks = []

        self.frame_id = 0
        self.max_time_lost = track_buffer
        self.next_id = 1

        # re-id model
        self.reid = PretrainedReIDModel()
        self.appearance_memory = {}

    # -------------------------------------------------------------------
    def update(self, detections, frame):
        """
        detections = [[x1,y1,x2,y2,conf,class], ...]
        frame = image BGR
        """
        self.frame_id += 1

        dets = []
        for det in detections:
            if len(det) >= 6:
                feat = self.reid.extract(frame, det[:4])
                dets.append({
                    "bbox": det[:4],
                    "score": det[4],
                    "class_id": det[5],
                    "features": feat
                })

        self._match(dets)
        self._clean_tracks()

        return self._export()

    # -------------------------------------------------------------------
    def _match(self, detections):
        active_tracks = [t for t in self.tracked_tracks]

        if len(active_tracks) == 0:
            for det in detections:
                if det["score"] > self.track_thresh:
                    self._create_track(det)
            return

        cost_iou = np.ones((len(active_tracks), len(detections)))
        cost_app = np.ones((len(active_tracks), len(detections)))

        # remplir matrices coûts
        for i, trk in enumerate(active_tracks):
            for j, det in enumerate(detections):
                iou = self._iou(trk["bbox"], det["bbox"])
                cost_iou[i, j] = 1 - iou

                # similarity apparence
                if det["features"] is not None and trk["track_id"] in self.appearance_memory:
                    app_sim = self._cosine(self.appearance_memory[trk["track_id"]],
                                           det["features"])
                    cost_app[i, j] = 1 - app_sim
                else:
                    cost_app[i, j] = 1  # très mauvais

        # matrice coût combinée
        cost = 0.5 * cost_iou + 0.5 * cost_app

        row, col = linear_sum_assignment(cost)

        matched_tracks = []
        matched_dets = []

        for i, j in zip(row, col):
            if cost[i, j] < (1 - self.match_thresh):
                trk = active_tracks[i]
                det = detections[j]

                # update track
                trk["bbox"] = det["bbox"]
                trk["score"] = det["score"]
                trk["last_update"] = self.frame_id

                # update appearance
                if det["features"] is not None:
                    self.appearance_memory[trk["track_id"]] = det["features"]

                matched_tracks.append(i)
                matched_dets.append(j)

        # create new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets and det["score"] > self.track_thresh:
                self._create_track(det)

    # -------------------------------------------------------------------
    def _create_track(self, det):
        track = {
            "track_id": self.next_id,
            "bbox": det["bbox"],
            "score": det["score"],
            "last_update": self.frame_id
        }
        self.tracked_tracks.append(track)
        if det["features"] is not None:
            self.appearance_memory[self.next_id] = det["features"]
        self.next_id += 1

    # -------------------------------------------------------------------
    def _clean_tracks(self):
        self.tracked_tracks = [
            t for t in self.tracked_tracks
            if (self.frame_id - t["last_update"] <= self.max_time_lost)
        ]

    # -------------------------------------------------------------------
    def _export(self):
        results = []
        for t in self.tracked_tracks:
            x1, y1, x2, y2 = t["bbox"]
            results.append({
                "id": t["track_id"],
                "bbox": t["bbox"],
                "score": t["score"],
                "width": x2 - x1,
                "height": y2 - y1
            })
        return results

    # -------------------------------------------------------------------
    @staticmethod
    def _iou(a, b):
        x1, y1, x2, y2 = a
        x1b, y1b, x2b, y2b = b

        xi1 = max(x1, x1b)
        yi1 = max(y1, y1b)
        xi2 = min(x2, x2b)
        yi2 = min(y2, y2b)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    @staticmethod
    def _cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
