import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import networkx as nx

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Use Agg backend for non-interactive plotting
        plt.switch_backend('Agg')
    
    def plot_frame_with_tracks(self, frame, tracks, frame_idx):
        """Plot frame with bounding boxes and IDs"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame)
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-10, f"ID:{track['id']}", color='white', 
                    fontweight='bold', backgroundcolor='red')
            
        ax.axis('off')
        plt.tight_layout()
        filename = f"{self.output_dir}/frame_{frame_idx:04d}.png"
        plt.savefig(filename)
        plt.close()

    def plot_graph_on_frame(self, frame, graph, positions, frame_idx):
        """Overlay interaction graph on the frame"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame)
        
        # Draw edges
        for u, v in graph.edges():
            if u in positions and v in positions:
                pos_u = positions[u]
                pos_v = positions[v]
                ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'c-', alpha=0.5, linewidth=1)
        
        # Draw nodes
        for node, pos in positions.items():
            ax.plot(pos[0], pos[1], 'ro', markersize=5)
            
        ax.axis('off')
        plt.tight_layout()
        filename = f"{self.output_dir}/graph_{frame_idx:04d}.png"
        plt.savefig(filename)
        plt.close()

    def plot_metrics(self, metrics_history):
        """Plot time-series of metrics"""
        if not metrics_history:
            return

        frames = [m['frame'] for m in metrics_history]
        densities = [m['density'] for m in metrics_history]
        clustering = [m['clustering'] for m in metrics_history]
        avg_speeds = [m['avg_speed'] for m in metrics_history]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        ax1.plot(frames, densities, 'b-', label='Density')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(frames, clustering, 'g-', label='Clustering Coeff')
        ax2.set_ylabel('Clustering')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(frames, avg_speeds, 'r-', label='Avg Speed')
        ax3.set_ylabel('Speed (px/frame)')
        ax3.set_xlabel('Frame')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/metrics_plot.png")
        plt.close()

    def plot_anomalies(self, metrics_history, anomalies):
        """Plot metrics with anomaly regions highlighted"""
        if not metrics_history:
            return

        frames = [m['frame'] for m in metrics_history]
        densities = [m['density'] for m in metrics_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(frames, densities, 'b-', label='Density')
        
        for anomaly in anomalies:
            plt.axvspan(anomaly['start_frame'], anomaly['end_frame'], color='red', alpha=0.3, label='Anomaly')
            
        plt.xlabel('Frame')
        plt.ylabel('Density')
        plt.title('Anomaly Detection')
        # Deduplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.savefig(f"{self.output_dir}/anomalies_plot.png")
        plt.close()

    def plot_heatmap(self, trajectories):
        """Plot 2D histogram of positions"""
        if not trajectories:
            return
            
        all_x = []
        all_y = []
        for t_list in trajectories.values():
            for t in t_list:
                all_x.append(t['pixel_foot_x'])
                all_y.append(t['pixel_foot_y'])
                
        if not all_x:
            return
            
        plt.figure(figsize=(10, 8))
        plt.hist2d(all_x, all_y, bins=50, cmap='hot')
        plt.colorbar(label='Count')
        plt.title('Crowd Density Heatmap')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.gca().invert_yaxis() # Match image coordinates
        
        plt.savefig(f"{self.output_dir}/heatmap.png")
        plt.close()

    def plot_sim_vs_real(self, real_metrics, sim_metrics):
        """Compare Real vs Simulation metrics"""
        if not real_metrics or not sim_metrics:
            return
            
        # Align lengths
        min_len = min(len(real_metrics), len(sim_metrics))
        real_metrics = real_metrics[:min_len]
        sim_metrics = sim_metrics[:min_len]
        
        frames = range(min_len)
        
        real_density = [m['density'] for m in real_metrics]
        sim_density = [m['density'] for m in sim_metrics]
        
        real_speed = [m['avg_speed'] for m in real_metrics]
        sim_speed = [m['avg_speed'] for m in sim_metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Density Comparison
        ax1.plot(frames, real_density, 'b-', label='Real Density')
        ax1.plot(frames, sim_density, 'r--', label='Sim Density')
        ax1.set_title('Density Comparison: Real vs Sim')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True)
        
        # Speed Comparison
        ax2.plot(frames, real_speed, 'b-', label='Real Speed')
        ax2.plot(frames, sim_speed, 'r--', label='Sim Speed')
        ax2.set_title('Speed Comparison: Real vs Sim')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Avg Speed')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sim_vs_real.png")
        plt.close()
