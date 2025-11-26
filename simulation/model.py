import mesa
from .agent import PedestrianAgent
import numpy as np

class CrowdModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height, scenario='normal', init_metrics=None):
        super().__init__()
        self.num_agents = N
        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.steps_count = 0

        self.scenario = scenario
        self.init_metrics = init_metrics
        
        # Create agents - MESA v3.x automatically manages them
        for i in range(self.num_agents):
            x = self.random.random() * width
            y = self.random.random() * height
            a = PedestrianAgent(i, self, (x, y))
            self.space.place_agent(a, (x, y))

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Density": compute_density,
                "AvgSpeed": compute_avg_speed
            }
        )

    def step(self):
        self.datacollector.collect(self)
        # MESA v3.x: iterate through agents
        agents_list = list(self.agents)
        self.random.shuffle(agents_list)
        for agent in agents_list:
            agent.step()
        self.steps_count += 1

def compute_density(model):
    # Simple density: agents per area (or just N for fixed area)
    return model.num_agents / (model.space.width * model.space.height)

def compute_avg_speed(model):
    agents_list = list(model.agents)
    speeds = [agent.speed for agent in agents_list]
    return np.mean(speeds) if speeds else 0
