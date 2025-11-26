from collections import defaultdict
import csv

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
        
        # print(f"📈 Trajectoires extraites: {len(current_trajectories)}")
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
