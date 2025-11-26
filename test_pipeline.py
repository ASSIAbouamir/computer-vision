import cv2
import numpy as np
import os
import shutil
from pipeline.tracking import BotSortTracker
from pipeline.trajectory import TrajectoryExtractor
from pipeline.graph import InteractionGraph
from pipeline.metrics import MetricsCalculator
from pipeline.anomaly import AnomalyDetector
from simulation.model import CrowdModel
import pandas as pd

# Mock Detector
class MockDetector:
    def detect_people(self, frame):
        # Return random detections
        detections = []
        for i in range(3): # 3 people
            x = np.random.randint(0, 500)
            y = np.random.randint(0, 400)
            w = 50
            h = 100
            detections.append({
                'bbox': [x, y, x+w, y+h],
                'center': [x+w/2, y+h/2],
                'foot': [x+w/2, y+h],
                'confidence': 0.9,
                'class_id': 0,
                'width': w,
                'height': h
            })
        return detections

def test_pipeline():
    print("🧪 Testing Pipeline Components...")
    output_dir = "test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Initialize
    detector = MockDetector()
    tracker = BotSortTracker()
    trajectory_extractor = TrajectoryExtractor(calibration=None) # No calibration for test
    graph_builder = InteractionGraph(distance_threshold=200) # Large threshold to ensure edges
    metrics_calc = MetricsCalculator()
    anomaly_detector = AnomalyDetector()

    metrics_history = []

    # Process 10 dummy frames
    print("   Processing 10 dummy frames...")
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_idx = i + 1
        
        # Vision
        detections = detector.detect_people(frame)
        tracks = tracker.update(detections)
        trajectories = trajectory_extractor.extract_trajectories(tracks, 640, 480)
        
        # Graph
        G = graph_builder.build_graph(trajectories)
        metrics = metrics_calc.calculate_metrics(G, trajectories, frame_idx)
        metrics_history.append(metrics)
        
        # Anomaly
        is_anomaly, score = anomaly_detector.update(metrics)
        
        print(f"   Frame {frame_idx}: {len(tracks)} tracks, {G.number_of_edges()} edges, Density: {metrics['density']:.2f}")

    # Simulation
    print("   Testing Simulation...")
    sim_model = CrowdModel(N=5, width=100, height=100)
    for i in range(5):
        sim_model.step()
    
    print("✅ Test Complete. Pipeline logic seems sound.")

if __name__ == "__main__":
    test_pipeline()
