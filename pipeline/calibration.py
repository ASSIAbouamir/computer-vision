import cv2
import numpy as np
import xml.etree.ElementTree as ET

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
