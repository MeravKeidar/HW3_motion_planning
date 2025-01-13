import numpy as np
from RRTTree import RRTTree
import time


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob, start, goal):
        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

        # eta for extracting new nodes
        x_range = bb.env.xlimit[1] - bb.env.xlimit[0]
        y_range = bb.env.ylimit[1] - bb.env.ylimit[0]
        self.eta = 0.03 * np.sqrt(x_range**2 + y_range**2) 
        
    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.tree.add_vertex(self.start)
        n = 10000 #number of iterations
        for _ in range(n):
            xrand = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_idx, xnear = self.tree.get_nearest_config(xrand)
            xnew = self.extend(xnear, xrand)
            
            if self.bb.edge_validity_checker(xnear, xnew):
                new_idx = self.tree.add_vertex(xnew)
                self.tree.add_edge(near_idx,new_idx)
            else:
                continue
            
            if np.allclose(xnew, self.goal, atol=1e-3):
                path = [self.goal]
                curr_idx = new_idx
                while curr_idx != 0:
                    curr_idx = self.tree.edges[curr_idx]
                    path.append(self.tree.vertices[curr_idx].config)
                return np.array(path[::-1])
        return []
    
    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # HW3 2.2.2
        if len(plan) == 0:
            return float('inf') 
        return sum(self.bb.compute_distance(plan[i], plan[i+1]) for i in range(len(plan)-1))

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        if self.ext_mode == "E1":
            return rand_config
        
        if self.ext_mode == "E2":
            distance = self.bb.compute_distance(rand_config,near_config)
            
            if distance <= self.eta:
                return rand_config
            
            direction = (rand_config - near_config) / distance
            return near_config + self.eta * direction