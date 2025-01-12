import itertools
import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config): 
        # HW3 2.1
        return np.sqrt(np.sum((prev_config - next_config) ** 2))
    
    def sample_random_config(self, goal_prob, goal):
        # HW3 2.1
        if np.random.uniform(0,1) < goal_prob:
            return goal
        
        x = np.random.uniform(self.env.xlimit[0], self.env.xlimit[1])
        y = np.random.uniform(self.env.ylimit[0], self.env.ylimit[1])
        config = np.array([x, y])
        
        if not self.config_validity_checker(config): # if not valid, sample again
            return self.sample_random_config(goal_prob, goal)
        
        return config
        
    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)


