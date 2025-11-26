import networkx as nx
import numpy as np

class InteractionGraph:
    def __init__(self, distance_threshold=50):
        """
        Args:
            distance_threshold: Distance in pixels (or meters if calibrated) to consider an interaction.
        """
        self.distance_threshold = distance_threshold

    def build_graph(self, trajectories_in_frame):
        """
        Builds a graph for a single frame.
        Args:
            trajectories_in_frame: List of dicts or Dict of dicts containing 'id' and positions.
                                   Expected format: {'id': 1, 'pixel_foot_x': 100, 'pixel_foot_y': 200, ...}
        Returns:
            G: networkx.Graph
        """
        G = nx.Graph()
        
        # Add nodes
        ids = []
        positions = []
        
        # Handle both list and dict input
        items = trajectories_in_frame.values() if isinstance(trajectories_in_frame, dict) else trajectories_in_frame
        
        for item in items:
            node_id = item['id']
            # Use foot position for more accurate ground distance
            pos = np.array([item['pixel_foot_x'], item['pixel_foot_y']])
            
            G.add_node(node_id, pos=pos)
            ids.append(node_id)
            positions.append(pos)
            
        # Add edges based on distance
        num_nodes = len(ids)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.distance_threshold:
                    # Weight could be inverse of distance
                    weight = 1.0 / (dist + 1e-6)
                    G.add_edge(ids[i], ids[j], weight=weight, distance=dist)
                    
        return G
