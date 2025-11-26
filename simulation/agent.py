import mesa
import numpy as np

class PedestrianAgent(mesa.Agent):
    """An agent with fixed initial position."""

    def __init__(self, unique_id, model, pos, speed=1.0, heading=None):
        # MESA v3.x: super().__init__() takes no args
        super().__init__(model)
        self.unique_id = unique_id
        self.pos = pos
        self.speed = speed
        # Random initial heading if not provided
        self.heading = heading if heading is not None else np.random.uniform(0, 2 * np.pi)
        # Random goal for simulation
        self.goal = (np.random.random() * model.space.width, np.random.random() * model.space.height)

    def step(self):
        # Simple movement logic: move towards goal or random walk
        # For "panic" scenario, we might increase speed or align with neighbors
        
        if self.model.scenario == 'panic':
            self.speed = 2.0
            # Align with neighbors (flocking)
            neighbors = self.model.space.get_neighbors(self.pos, radius=20, include_center=False)
            if neighbors:
                avg_heading = np.mean([n.heading for n in neighbors])
                self.heading = avg_heading
        
        # Move
        dx = self.speed * np.cos(self.heading)
        dy = self.speed * np.sin(self.heading)
        
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        
        # Boundary check
        new_x = max(0, min(new_x, self.model.space.width))
        new_y = max(0, min(new_y, self.model.space.height))
        
        self.model.space.move_agent(self, (new_x, new_y))
