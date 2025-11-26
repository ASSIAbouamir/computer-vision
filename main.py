import cv2
import os
import numpy as np
import pandas as pd
from pipeline.calibration import CameraCalibration
from pipeline.detection import PersonDetector
from pipeline.tracking import BotSortTracker
from pipeline.trajectory import TrajectoryExtractor
from pipeline.video import Preprocessor
from pipeline.visualization import Visualizer
from pipeline.graph import InteractionGraph
from pipeline.metrics import MetricsCalculator
from pipeline.anomaly import AnomalyDetector
from pipeline.anomaly import AnomalyDetector
from simulation.model import CrowdModel

# Configuration
VIEW_PATH = r"C:\Users\user1\Desktop\computer_vision\Crowd_PETS09\S2\L1\Time_12-34\View_001" 
CALIBRATION_FILE = "calibration.xml" # Optional
OUTPUT_DIR = "output_results"

def main():
    print("🚀 Starting Collective Behavior Analysis Pipeline")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize Components
    print("🔧 Initializing components...")
    calibration = CameraCalibration(CALIBRATION_FILE)
    detector = PersonDetector()
    tracker = BotSortTracker()
    trajectory_extractor = TrajectoryExtractor(calibration)
    preprocessor = Preprocessor(VIEW_PATH)
    visualizer = Visualizer(OUTPUT_DIR)
    
    graph_builder = InteractionGraph(distance_threshold=50) # 50 pixels or meters depending on calibration usage
    metrics_calc = MetricsCalculator()
    anomaly_detector = AnomalyDetector()

    # 2. Load Frames
    frames = preprocessor.load_and_preprocess_frames(max_frames=50)
    if not frames:
        print("❌ No frames found. Check VIEW_PATH.")
        return

    metrics_history = []
    all_anomalies = []

    # 3. Process Video
    print("🎬 Processing video...")
    for i, frame in enumerate(frames):
        frame_idx = i + 1
        print(f"   Frame {frame_idx}/{len(frames)}")

        # A. Vision Pipeline
        detections = detector.detect_people(frame)
        tracks = tracker.update(detections)
        trajectories = trajectory_extractor.extract_trajectories(tracks, frame.shape[1], frame.shape[0])
        
        # B. Graph & Metrics
        G = graph_builder.build_graph(trajectories)
        metrics = metrics_calc.calculate_metrics(G, trajectories, frame_idx)
        metrics_history.append(metrics)
        
        # C. Anomaly Detection
        is_anomaly, score = anomaly_detector.update(metrics)
        if is_anomaly:
            print(f"   ⚠️ Anomaly detected at frame {frame_idx} (Score: {score:.2f})")
            all_anomalies.append({
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'score': score
            })

        # D. Visualization
        visualizer.plot_frame_with_tracks(frame, tracks, frame_idx)
        visualizer.plot_graph_on_frame(frame, G, {id: (t['pixel_foot_x'], t['pixel_foot_y']) for id, t in trajectories.items()}, frame_idx)

    # 4. Post-Processing Visualization
    print("📊 Generating reports...")
    visualizer.plot_metrics(metrics_history)
    visualizer.plot_anomalies(metrics_history, all_anomalies)
    
    # Save Metrics to CSV
    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
    
    # 5. Simulation Comparison
    print("🤖 Running Multi-Agent Simulation...")
    try:
        # Initialize simulation with parameters derived from video (e.g., initial density)
        if metrics_history:
            avg_density = np.mean([m['density'] for m in metrics_history])
            # Map video density to simulation agents
            # This is a simplification. In a real system, we'd map physical area.
            num_agents = int(avg_density * 100) + 10 # Dummy mapping
            
            # Pass initial metrics to seed simulation
            init_metrics = metrics_history[0] if metrics_history else None
            sim_model = CrowdModel(N=num_agents, width=100, height=100, scenario='normal', init_metrics=init_metrics)
            sim_metrics = []
            
            for _ in range(len(frames)):
                sim_model.step()
                sim_metrics.append({
                    'density': sim_model.datacollector.model_vars['Density'][-1],
                    'avg_speed': sim_model.datacollector.model_vars['AvgSpeed'][-1]
                })
                
            print("   Simulation complete.")
            
            # Save sim results
            pd.DataFrame(sim_metrics).to_csv(f"{OUTPUT_DIR}/sim_metrics.csv", index=False)
            
            # Plot Sim vs Real
            visualizer.plot_sim_vs_real(metrics_history, sim_metrics)
            
    except Exception as e:
        print(f"⚠️ Simulation failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Advanced Visualization
    print("🎨 Generating advanced visualizations...")
    visualizer.plot_heatmap(trajectory_extractor.trajectories)

    print("✅ Pipeline complete!")
    print(f"📁 Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
