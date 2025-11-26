from ultralytics import YOLO
import os

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.3):
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
            
            # print(f"🔍 Détections: {len(detections)} personnes")
            return detections
            
        except Exception as e:
            print(f"❌ Erreur détection: {e}")
            return []
