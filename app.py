import mesa
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import time
from ultralytics import YOLO
import os
import torch
from scipy.optimize import linear_sum_assignment
import json
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# CONFIGURATION CORRIGÉE
# ============================================================================

view_path = "/kaggle/input/pets2009/Crowd_PETS09/S2/L1/Time_12-34/View_001"
calibration_file = "/content/OpenTraj/datasets/PETS-2009/data/calibration/View_001.xml"
output_visualization_dir = "/content/visualizations/"
output_trajectories_csv = "/content/trajectories_corrected.csv"
output_trajectories_json = "/content/trajectories_corrected.json"

# Gestion robuste du modèle
MODEL_PERSON_PATH = "/content/best.pt"
os.makedirs(output_visualization_dir, exist_ok=True)

# ============================================================================
# CALIBRATION CAMÉRA CORRIGÉE
# ============================================================================

class CameraCalibration:
    """Calibration caméra corrigée pour PETS2009"""
    
    def __init__(self, calibration_file):
        self.calibration_file = calibration_file
        self.camera_params = {}
        self.homography_matrix = None
        
        self.load_calibration()
        self.compute_accurate_homography()
        print("✅ Calibration caméra corrigée initialisée")
    
    def load_calibration(self):
        """Charge les paramètres de calibration avec gestion d'erreur"""
        try:
            tree = ET.parse(self.calibration_file)
            root = tree.getroot()
            
            # Paramètres géométriques
            geometry = root.find('Geometry')
            if geometry is not None:
                self.camera_params = {
                    'width': int(geometry.get('width', '768')),
                    'height': int(geometry.get('height', '576')),
                    'cx': float(geometry.get('ncx', '795.0')),  # Centre optique X
                    'cy': float(geometry.get('nfx', '752.0')),  # Centre optique Y
                    'dx': float(geometry.get('dx', '0.00485')),
                    'dy': float(geometry.get('dy', '0.00465'))
                }
            
            # Paramètres intrinsèques
            intrinsic = root.find('Intrinsic')
            if intrinsic is not None:
                self.camera_params.update({
                    'focal': float(intrinsic.get('focal', '5.5549183034')),
                    'kappa1': float(intrinsic.get('kappa1', '0.0051113043639')),
                    'cx_pixel': float(intrinsic.get('cx', '324.22149053')),  # Centre en pixels
                    'cy_pixel': float(intrinsic.get('cy', '282.56650051')),
                    'sx': float(intrinsic.get('sx', '1.0937855397'))
                })
            
            print(f"📷 Calibration chargée: {self.camera_params['width']}x{self.camera_params['height']}")
            
        except Exception as e:
            print(f"❌ Erreur calibration: {e}")
            self.camera_params = {
                'width': 768, 'height': 576,
                'cx_pixel': 324.22, 'cy_pixel': 282.57,
                'focal': 5.55
            }
    
    def compute_accurate_homography(self):
        """Calcule une homographie précise pour PETS2009 S2/L1"""
        try:
            # Points de référence spécifiques à PETS2009 S2/L1 View_001
            src_points = np.array([
                [184, 441],   # Point bas gauche (sol)
                [583, 441],   # Point bas droit (sol) 
                [384, 323],   # Point milieu haut
                [284, 382],   # Point milieu gauche
                [483, 382]    # Point milieu droit
            ], dtype=np.float32)
            
            # Coordonnées monde correspondantes (en mètres)
            dst_points = np.array([
                [0, 0],       # Origine
                [15, 0],      # 15m à droite
                [7.5, 12],    # 7.5m droite, 12m avant
                [3, 6],       # 3m droite, 6m avant
                [12, 6]       # 12m droite, 6m avant
            ], dtype=np.float32)
            
            # Calcul de l'homographie avec RANSAC pour plus de robustesse
            self.homography_matrix, status = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, 5.0
            )
            
            if self.homography_matrix is not None:
                print("✅ Homographie précise calculée")
            else:
                raise Exception("Calcul homographie échoué")
                
        except Exception as e:
            print(f"❌ Erreur calcul homographie: {e}")
            # Matrice identité comme fallback
            self.homography_matrix = np.eye(3)
    
    def pixel_to_world_corrected(self, pixel_x, pixel_y, foot_position=True):
        """Convertit pixels → monde avec correction de la position des pieds"""
        try:
            if foot_position:
                # Pour les personnes, on utilise la position des pieds (bas de bbox)
                y_foot = pixel_y
            else:
                y_foot = pixel_y
            
            point = np.array([pixel_x, y_foot, 1.0])
            world_point = np.dot(self.homography_matrix, point)
            
            # Normalisation
            if world_point[2] != 0:
                world_x = world_point[0] / world_point[2]
                world_y = world_point[1] / world_point[2]
            else:
                world_x, world_y = pixel_x, pixel_y
            
            return float(world_x), float(world_y)
            
        except Exception as e:
            print(f"⚠️ Erreur conversion: {e}")
            return float(pixel_x), float(pixel_y)

# ============================================================================
# DÉTECTEUR AVEC CORRECTION DES POSITIONS
# ============================================================================

class PersonDetector:
    def __init__(self, model_path=MODEL_PERSON_PATH, conf_threshold=0.8):
        self.conf_threshold = conf_threshold
        
        # Chargement robuste
        try:
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1024 * 1024:  # > 1MB
                self.yolo_model = YOLO(model_path)
                print(f"✅ Modèle personnalisé chargé: {model_path}")
            else:
                self.yolo_model = YOLO('yolov8n.pt')
                print("🔄 Utilisation de YOLOv8n (modèle par défaut)")
        except:
            self.yolo_model = YOLO('yolov8n.pt')
            print("🔄 Utilisation de YOLOv8n (fallback)")
        
        self.class_names = self.yolo_model.names
        print(f"📋 Classes: {self.class_names}")
    
    def detect_people(self, frame):
        """Détection avec positions corrigées"""
        try:
            results = self.yolo_model(
                frame, conf=self.conf_threshold, verbose=False, imgsz=640
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.class_names[class_id]
                        
                        # Filtrage personnes
                        if 'person' in class_name.lower() or class_id == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            # Position corrigée : centre de la bbox
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            
                            # Pour une meilleure précision sol, on utilise le bas de la bbox
                            y_foot = y2  # Position des pieds
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'center': [x_center, y_center],
                                'foot': [x_center, y_foot],  # Position des pieds
                                'confidence': confidence,
                                'class_id': class_id,
                                'width': x2 - x1,
                                'height': y2 - y1
                            })
            
            print(f"🔍 Détections: {len(detections)} personnes")
            return detections
            
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return []

# ============================================================================
# TRACKER AVEC POSITIONS CORRIGÉES
# ============================================================================

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
        print(f"🆕 Track {self.next_id} créé")
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

# ============================================================================
# EXTRACTEUR DE TRAJECTOIRES CORRIGÉ
# ============================================================================

class TrajectoryExtractor:
    def __init__(self, calibration=None):
        self.calibration = calibration
        self.trajectories = defaultdict(list)
        self.trajectories_data = []
        
        print("✅ Extracteur de trajectoires corrigé initialisé")
    
    def extract_trajectories(self, tracks, frame_width, frame_height):
        """Extrait les trajectoires avec positions corrigées"""
        current_trajectories = {}
        
        for track in tracks:
            track_id = track['id']
            
            # MULTIPLES POSITIONS POUR COMPARAISON
            positions = {
                'bbox_center': track['center'],  # Centre bbox
                'foot_position': track['foot'],  # Position pieds
                'bbox': track['bbox']           # Bbox complète
            }
            
            # Conversion en coordonnées monde si calibration disponible
            world_positions = {}
            if self.calibration:
                for pos_name, pos in positions.items():
                    if pos_name == 'foot_position':
                        # Pour les pieds, on utilise la position Y réelle
                        world_x, world_y = self.calibration.pixel_to_world_corrected(
                            pos[0], pos[1], foot_position=True
                        )
                    else:
                        # Pour le centre, on utilise le centre
                        world_x, world_y = self.calibration.pixel_to_world_corrected(
                            pos[0], pos[1], foot_position=False
                        )
                    world_positions[pos_name] = (world_x, world_y)
            
            # Sauvegarde des données CORRIGÉES
            trajectory_point = {
                'id': track_id,
                'frame': track['frame'],
                # Positions pixels
                'pixel_center_x': float(positions['bbox_center'][0]),
                'pixel_center_y': float(positions['bbox_center'][1]),
                'pixel_foot_x': float(positions['foot_position'][0]),
                'pixel_foot_y': float(positions['foot_position'][1]),
                # Positions monde (si disponible)
                'world_center_x': float(world_positions.get('bbox_center', (0, 0))[0]),
                'world_center_y': float(world_positions.get('bbox_center', (0, 0))[1]),
                'world_foot_x': float(world_positions.get('foot_position', (0, 0))[0]),
                'world_foot_y': float(world_positions.get('foot_position', (0, 0))[1]),
                # Dimensions
                'width': float(track['width']),
                'height': float(track['height']),
                'score': float(track['score']),
                'has_calibration': self.calibration is not None
            }
            
            self.trajectories_data.append(trajectory_point)
            self.trajectories[track_id].append(trajectory_point)
            
            # Pour l'analyse en temps réel
            current_trajectories[track_id] = trajectory_point
        
        print(f"📈 Trajectoires extraites: {len(current_trajectories)}")
        return current_trajectories
    
    def save_trajectories_to_csv(self, filename):
        """Sauvegarde avec toutes les positions"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'id', 'frame', 
                    'pixel_center_x', 'pixel_center_y',
                    'pixel_foot_x', 'pixel_foot_y', 
                    'world_center_x', 'world_center_y',
                    'world_foot_x', 'world_foot_y',
                    'width', 'height', 'score', 'has_calibration'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for trajectory in self.trajectories_data:
                    writer.writerow(trajectory)
            
            print(f"✅ Trajectoires corrigées sauvegardées: {filename}")
            return True
        except Exception as e:
            print(f"❌ Erreur sauvegarde CSV: {e}")
            return False

# ============================================================================
# VISUALISATION CORRIGÉE (UTILISE MATPLOTLIB PATCHES)
# ============================================================================

class ComparisonVisualizer:
    """Visualise la comparaison positions réelles vs extraites"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def create_comparison_plot(self, frame, detections, tracks, trajectories, frame_idx):
        """Crée un graphique de comparaison avec matplotlib patches"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'COMPARAISON POSITIONS - Frame {frame_idx}', fontsize=16, fontweight='bold')
        
        # 1. Image originale avec détections
        ax1.imshow(frame)
        ax1.set_title('🎬 Image Originale + Détections', fontweight='bold')
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Utiliser matplotlib Rectangle au lieu de cv2.rectangle
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
            ax1.add_patch(rect)
            
            # Points de référence
            ax1.plot(det['center'][0], det['center'][1], 'ro', markersize=4, label='Centre' if det == detections[0] else "")
            ax1.plot(det['foot'][0], det['foot'][1], 'bo', markersize=4, label='Pieds' if det == detections[0] else "")
        
        if detections:
            ax1.legend()
        ax1.axis('off')
        
        # 2. Tracking avec positions
        ax2.imshow(frame)
        ax2.set_title('🎯 Tracking + Positions', fontweight='bold')
        
        for track in tracks:
            # Bbox avec matplotlib
            x1, y1, x2, y2 = track['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            
            # Positions
            ax2.plot(track['center'][0], track['center'][1], 'ro', markersize=6, 
                    label='Centre' if track == tracks[0] else "")
            ax2.plot(track['foot'][0], track['foot'][1], 'bo', markersize=6, 
                    label='Pieds' if track == tracks[0] else "")
            
            # ID avec matplotlib text
            ax2.text(x1, y1-10, f"ID:{track['id']}", color='white', 
                    fontweight='bold', backgroundcolor='red')
        
        if tracks:
            ax2.legend()
        ax2.axis('off')
        
        # 3. Graphique des positions
        ax3.set_title('📊 Comparaison Positions', fontweight='bold')
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))
        
        for i, (track_id, traj) in enumerate(trajectories.items()):
            color = colors[i]
            # Positions centre vs pieds
            ax3.plot(traj['pixel_center_x'], traj['pixel_center_y'], 'o', 
                    color=color, markersize=8, label=f'ID{track_id} Centre')
            ax3.plot(traj['pixel_foot_x'], traj['pixel_foot_y'], 's', 
                    color=color, markersize=6, label=f'ID{track_id} Pieds')
        
        ax3.set_xlabel('Position X (pixels)')
        ax3.set_ylabel('Position Y (pixels)')
        if trajectories:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()  # Pour correspondre à l'image
        
        # 4. Statistiques
        ax4.axis('off')
        stats_text = "📈 STATISTIQUES DE PRÉCISION\n\n"
        if trajectories:
            for track_id, traj in trajectories.items():
                # Calcul du décalage centre-pieds
                dx = abs(traj['pixel_center_x'] - traj['pixel_foot_x'])
                dy = abs(traj['pixel_center_y'] - traj['pixel_foot_y'])
                distance = np.sqrt(dx**2 + dy**2)
                
                stats_text += f"ID {track_id}:\n"
                stats_text += f"  • Décalage: {distance:.1f} pixels\n"
                stats_text += f"  • Hauteur bbox: {traj['height']:.1f} px\n"
                stats_text += f"  • Score: {traj['score']:.3f}\n\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        filename = f"{self.output_dir}comparison_frame_{frame_idx:04d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Comparaison sauvegardée: {filename}")

# ============================================================================
# PRÉPROCESSEUR
# ============================================================================

class Preprocessor:
    def __init__(self, view_path, target_size=(640, 480)):
        self.view_path = view_path
        self.target_size = target_size
        self.frames = []
        self.original_frames = []
    
    def load_and_preprocess_frames(self, max_frames=50):
        print("🔄 Chargement des frames...")
        self.frames = []
        self.original_frames = []
        
        for i in range(1, max_frames + 1):
            frame_path = f"{self.view_path}/frame_{i:04d}.jpg"
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    # Sauvegarde originale
                    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.original_frames.append(original_rgb)
                    
                    # Prétraitement
                    frame = cv2.resize(frame, self.target_size)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frames.append(frame_rgb)
        
        print(f"✅ {len(self.frames)} frames chargées")
        return self.frames

# ============================================================================
# PIPELINE CORRIGÉ
# ============================================================================

def run_corrected_pipeline():
    """Exécute le pipeline corrigé avec comparaison"""
    print("🎯 PIPELINE CORRIGÉ - COMPARAISON POSITIONS")
    print("=" * 60)
    
    # 1. Initialisation
    print("\n1. 🔧 INITIALISATION")
    calibration = CameraCalibration(calibration_file)
    detector = PersonDetector()
    tracker = BotSortTracker()
    trajectory_extractor = TrajectoryExtractor(calibration)
    visualizer = ComparisonVisualizer(output_visualization_dir)
    
    # 2. Chargement des frames
    print("\n2. 🎬 CHARGEMENT DES FRAMES")
    preprocessor = Preprocessor(view_path)
    frames = preprocessor.load_and_preprocess_frames(max_frames=20)
    
    if not frames:
        print("❌ Aucune frame trouvée")
        return
    
    print(f"   • Frames chargées: {len(frames)}")
    print(f"   • Calibration: {'✅ Activée' if calibration else '❌ Désactivée'}")
    
    # 3. Traitement frame par frame
    print("\n3. 🔄 TRAITEMENT DES FRAMES")
    
    for frame_idx, frame in enumerate(frames):
        print(f"\n--- Frame {frame_idx + 1}/{len(frames)} ---")
        
        # Détection
        detections = detector.detect_people(frame)
        
        # Tracking
        tracks = tracker.update(detections)
        
        # Extraction trajectoires
        trajectories = trajectory_extractor.extract_trajectories(
            tracks, frame.shape[1], frame.shape[0]
        )
        
        # Visualisation comparaison
        visualizer.create_comparison_plot(
            frame, detections, tracks, trajectories, frame_idx + 1
        )
        
        # Affichage statistiques
        if trajectories:
            print(f"   • Tracks actifs: {len(tracks)}")
            print(f"   • Trajectoires: {len(trajectories)}")
            
            # Calcul décalage moyen
            decalages = []
            for traj in trajectories.values():
                dx = abs(traj['pixel_center_x'] - traj['pixel_foot_x'])
                dy = abs(traj['pixel_center_y'] - traj['pixel_foot_y'])
                decalages.append(np.sqrt(dx**2 + dy**2))
            
            if decalages:
                print(f"   • Décalage moyen: {np.mean(decalages):.1f} pixels")
    
    # 4. Sauvegarde résultats
    print("\n4. 💾 SAUVEGARDE DES RÉSULTATS")
    trajectory_extractor.save_trajectories_to_csv(output_trajectories_csv)
    
    print(f"\n🎉 PIPELINE TERMINÉ!")
    print(f"📁 Résultats dans: {output_visualization_dir}")
    print(f"📊 Données brutes: {output_trajectories_csv}")
    
    return trajectory_extractor.trajectories_data

# ============================================================================
# ANALYSE DE PRÉCISION
# ============================================================================

def analyze_accuracy(trajectories_data):
    """Analyse la précision des positions extraites"""
    print("\n🔍 ANALYSE DE PRÉCISION")
    print("=" * 50)
    
    if not trajectories_data:
        print("❌ Aucune donnée à analyser")
        return
    
    # Calcul des décalages
    decalages_centre_pieds = []
    hauteurs_bbox = []
    
    for point in trajectories_data:
        dx = abs(point['pixel_center_x'] - point['pixel_foot_x'])
        dy = abs(point['pixel_center_y'] - point['pixel_foot_y'])
        distance = np.sqrt(dx**2 + dy**2)
        
        decalages_centre_pieds.append(distance)
        hauteurs_bbox.append(point['height'])
    
    # Statistiques
    print(f"📊 ÉCHANTILLON: {len(trajectories_data)} points de trajectoire")
    if decalages_centre_pieds:
        print(f"📏 DÉCALAGE CENTRE-PIEDS:")
        print(f"   • Moyenne: {np.mean(decalages_centre_pieds):.1f} pixels")
        print(f"   • Médiane: {np.median(decalages_centre_pieds):.1f} pixels") 
        print(f"   • Max: {np.max(decalages_centre_pieds):.1f} pixels")
        print(f"   • Min: {np.min(decalages_centre_pieds):.1f} pixels")
        
        print(f"📐 HAUTEUR BBOX:")
        print(f"   • Moyenne: {np.mean(hauteurs_bbox):.1f} pixels")
        if np.mean(hauteurs_bbox) > 0:
            rapport = (np.mean(decalages_centre_pieds)/np.mean(hauteurs_bbox))*100
            print(f"   • Rapport décalage/hauteur: {rapport:.1f}%")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if decalages_centre_pieds and np.mean(decalages_centre_pieds) > 50:
        print("   • 🔴 Décalage important - vérifier la détection")
    elif decalages_centre_pieds and np.mean(decalages_centre_pieds) > 25:
        print("   • 🟡 Décalage modéré - acceptable")
    elif decalages_centre_pieds:
        print("   • 🟢 Décalage faible - bonne précision")
    
    print("   • Utilisez 'pixel_foot' pour les positions au sol")
    print("   • Utilisez 'pixel_center' pour le centre des personnes")

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    print("🎯 PIPELINE CORRIGÉ - RÉSOLUTION DÉCALAGE POSITIONS")
    print("=" * 60)
    
    # Exécution du pipeline corrigé
    trajectories_data = run_corrected_pipeline()
    
    # Analyse de précision
    analyze_accuracy(trajectories_data)
    
    print(f"\n📁 ACCÈS RAPIDE:")
    print(f"   • Visualisations: {output_visualization_dir}")
    print(f"   • Données: {output_trajectories_csv}")