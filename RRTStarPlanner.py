import numpy as np
from RRTTree import RRTTree
import time


class RRTStarPlanner(object):

    def __init__(self, bb, ext_mode, max_step_size, start, goal,
                 max_itr=None, stop_on_goal=None, k=None, goal_prob=0.01):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k

        self.step_size = max_step_size

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        self.tree.add_vertex(self.start)
        n = 10000 
        for _ in range(n):
            xrand = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_idx, xnear = self.tree.get_nearest_config(xrand)
            xnew = self.extend(xnear, xrand)
        
            if not self.bb.edge_validity_checker(xnear, xnew):
                continue
            
            new_idx = self.tree.add_vertex(xnew)
            self.tree.add_edge(near_idx,new_idx)
            k_nearest_ids, _ = self.tree.get_k_nearest_neighbors(xnew, self.k)
            
            for node_idx in k_nearest_ids:
                if node_idx != near_idx:
                    self.rewire(new_idx, near_idx)
            
            for node_idx in k_nearest_ids:
                if node_idx != near_idx:
                    self.rewire(near_idx, new_idx)
            
            if np.allclose(xnew, self.goal, atol=1e-3):
                path = [self.goal]
                curr_idx = new_idx
                while curr_idx != 0:
                    curr_idx = self.tree.edges[curr_idx]
                    path.append(self.tree.vertices[curr_idx].config)
                return np.array(path[::-1])
            
        return []

    def rewire(self, potential_parent_idx, child_idx):
        """
        RRT* rewiring function
        Input: Indices of potential parent and child nodes
        Output: True if rewiring occurred
        """
        x_potential_parent = self.tree.vertices[potential_parent_idx].config
        x_child = self.tree.vertices[child_idx].config
        
        edge_cost = self.bb.compute_distance(x_potential_parent, x_child)
        new_cost = self.tree.vertices[potential_parent_idx].cost + edge_cost
    
        if new_cost < self.tree.vertices[child_idx].cost:
            if not self.bb.edge_validity_checker(x_potential_parent, x_child):
                return False
            # Rewire if new path is better
            self.tree.edges[child_idx] = potential_parent_idx
            self.tree.vertices[child_idx].set_cost(new_cost)
            return True
        return False

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
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

