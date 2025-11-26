import cv2
import os

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
        
        # Support for video files as well as directories
        if os.path.isfile(self.view_path):
            cap = cv2.VideoCapture(self.view_path)
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.original_frames.append(original_rgb)
                
                frame_resized = cv2.resize(frame, self.target_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                count += 1
            cap.release()
        else:
            # Directory mode (original app.py behavior)
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
