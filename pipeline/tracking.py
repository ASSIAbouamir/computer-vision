import numpy as np
from scipy.optimize import linear_sum_assignment

class BotSortTracker:
    def __init__(self, track_thresh=0.4, track_buffer=30, match_thresh=0.7):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        
        self.frame_id = 0
        self.max_time_lost = track_buffer
        self.next_id = 1
        
        print("✅ Tracker Bot-SORT initialisé")

    def update(self, detections):
        """Met à jour avec les détections corrigées"""
        self.frame_id += 1
        
        detections_list = []
        for det in detections:
            detections_list.append({
                'bbox': det['bbox'],
                'score': det['confidence'],
                'center': det['center'],
                'foot': det['foot'],
                'class_id': det['class_id']
            })
        
        self._match_detections_to_tracks(detections_list)
        self._manage_tracks()
        
        return self._get_active_tracks()

    def _match_detections_to_tracks(self, detections):
        """Association avec positions corrigées"""
        for track in self.tracked_tracks:
            track['active'] = True
        
        active_tracks = [t for t in self.tracked_tracks if t['active']]
        
        if not detections:
            for track in active_tracks:
                track['active'] = False
            return

        if not active_tracks:
            for det in detections:
                if det['score'] > self.track_thresh:
                    self._create_new_track(det)
            return

        # Matrice de coût avec positions corrigées
        cost_matrix = np.ones((len(active_tracks), len(detections)))
        
        for i, track in enumerate(active_tracks):
            for j, det in enumerate(detections):
                iou = self._calculate_iou(track['bbox'], det['bbox'])
                cost_matrix[i, j] = 1 - iou

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < (1 - self.match_thresh):
                matched_pairs.append((i, j))
        
        # Mise à jour avec positions corrigées
        for i, j in matched_pairs:
            track = active_tracks[i]
            det = detections[j]
            
            track.update({
                'bbox': det['bbox'],
                'score': det['score'],
                'center': det['center'],
                'foot': det['foot'],
                'last_update': self.frame_id,
                'active': True
            })
        
        matched_track_indices = [i for i, _ in matched_pairs]
        for i, track in enumerate(active_tracks):
            if i not in matched_track_indices:
                track['active'] = False

        matched_det_indices = [j for _, j in matched_pairs]
        for j, det in enumerate(detections):
            if j not in matched_det_indices and det['score'] > self.track_thresh:
                self._create_new_track(det)

    def _create_new_track(self, detection):
        """Crée un track avec positions corrigées"""
        new_track = {
            'track_id': self.next_id,
            'bbox': detection['bbox'],
            'score': detection['score'],
            'center': detection['center'],
            'foot': detection['foot'],
            'start_frame': self.frame_id,
            'last_update': self.frame_id,
            'active': True
        }
        self.tracked_tracks.append(new_track)
        # print(f"🆕 Track {self.next_id} créé")
        self.next_id += 1

    def _calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def _manage_tracks(self):
        current_time = self.frame_id
        tracks_to_remove = []
        
        for track in self.tracked_tracks:
            if not track['active']:
                if current_time - track['last_update'] > self.max_time_lost:
                    tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.tracked_tracks.remove(track)
            self.removed_tracks.append(track)

    def _get_active_tracks(self):
        """Retourne les tracks avec positions multiples"""
        active_tracks = []
        for track in self.tracked_tracks:
            if track['active']:
                x1, y1, x2, y2 = track['bbox']
                w = x2 - x1
                h = y2 - y1
                
                active_tracks.append({
                    'id': track['track_id'],
                    'frame': self.frame_id,
                    'bbox': track['bbox'],
                    'center': track['center'],  # Centre de la bbox
                    'foot': track['foot'],      # Position des pieds
                    'width': w,
                    'height': h,
                    'score': track['score']
                })
        
        return active_tracks
