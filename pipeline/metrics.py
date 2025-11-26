import networkx as nx
import numpy as np
from scipy.stats import entropy

class MetricsCalculator:
    def __init__(self):
        self.prev_trajectories = {}
    
    def calculate_metrics(self, G, trajectories_in_frame, frame_idx):
        """
        Calculate complex network metrics and other crowd statistics.
        """
        num_nodes = G.number_of_nodes()
        
        # Calculate kinematics regardless of node count to update history
        avg_speed, _, directional_entropy = self.calculate_kinematics(trajectories_in_frame, self.prev_trajectories)
        self.prev_trajectories = trajectories_in_frame.copy()

        if num_nodes == 0:
            return {
                'frame': frame_idx,
                'density': 0,
                'clustering': 0,
                'avg_degree': 0,
                'components': 0,
                'avg_speed': 0,
                'directional_entropy': 0
            }
            
        # Network Metrics
        density = nx.density(G)
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0
            
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0
        
        components = nx.number_connected_components(G)
        
        # Centrality (Betweenness) - computationally expensive for large graphs, but fine for crowd size
        try:
            betweenness = nx.betweenness_centrality(G)
            avg_betweenness = sum(betweenness.values()) / num_nodes if num_nodes > 0 else 0
        except:
            avg_betweenness = 0
            
        # Inter-individual distance variation
        # We can use the edges weights (distance) if graph is fully connected or just calculate pairwise from trajectories
        # For efficiency, let's use the edges we already calculated if they represent close interactions
        # Or better, calculate pairwise distances for all nodes to get a global "spread" metric
        positions = [np.array([t['pixel_foot_x'], t['pixel_foot_y']]) for t in trajectories_in_frame.values()]
        if len(positions) > 1:
            # Calculate all pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(positions)
            dist_variation = np.std(distances)
            avg_distance = np.mean(distances)
        else:
            dist_variation = 0
            avg_distance = 0
        
        return {
            'frame': frame_idx,
            'density': density,
            'clustering': clustering,
            'avg_degree': avg_degree,
            'components': components,
            'avg_speed': avg_speed,
            'directional_entropy': directional_entropy,
            'avg_betweenness': avg_betweenness,
            'dist_variation': dist_variation,
            'avg_distance': avg_distance
        }

    def calculate_kinematics(self, current_trajs, prev_trajs):
        """
        Calculate speed and direction based on current and previous frame.
        """
        speeds = []
        angles = []
        
        for track_id, curr in current_trajs.items():
            if track_id in prev_trajs:
                prev = prev_trajs[track_id]
                dx = curr['pixel_foot_x'] - prev['pixel_foot_x']
                dy = curr['pixel_foot_y'] - prev['pixel_foot_y']
                
                speed = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx)
                
                speeds.append(speed)
                angles.append(angle)
                
        avg_speed = np.mean(speeds) if speeds else 0
        speed_var = np.var(speeds) if speeds else 0
        
        # Directional Entropy
        if angles:
            # Discretize angles into bins
            hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi), density=True)
            # Add small epsilon to avoid log(0)
            hist = hist + 1e-10
            dir_entropy = entropy(hist)
        else:
            dir_entropy = 0
            
        return avg_speed, speed_var, dir_entropy
