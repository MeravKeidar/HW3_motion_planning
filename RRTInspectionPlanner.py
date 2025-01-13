import numpy as np
from RRTTree import RRTTree
import time
from twoD.environment import MapEnvironment



class RRTInspectionPlanner(object):

    def __init__(self, bb, start, ext_mode, goal_prob, coverage,env : MapEnvironment):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        self.env = env
        # set step size - remove for students
        self.step_size = min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)
        x_range = bb.env.xlimit[1] - bb.env.xlimit[0]
        y_range = bb.env.ylimit[1] - bb.env.ylimit[0]
        self.eta = 0.03 * np.sqrt(x_range**2 + y_range**2) 

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        
        start_inspected_points = self.bb.get_inspected_points(self.start)
        self.tree.add_vertex(self.start, start_inspected_points)
        all_points = set(map(tuple, self.env.inspection_points))
        inspected_points = set(map(tuple, start_inspected_points))
        n = 10000 #number of iterations
        for _ in range(n):
            if np.random.uniform(0,1) < self.goal_prob:
                uninspected_points = all_points - inspected_points
                if len(uninspected_points) > 0:
                   # target_point = np.array(list(uninspected_points)[np.random.randint(len(uninspected_points))])
                   pass 
                   #print(xrand)
            else:
                xrand = self.bb.sample_random_config(0,None)
            near_idx, xnear = self.tree.get_nearest_config(xrand)
            xnew = self.extend(xnear, xrand)
            
            if self.bb.edge_validity_checker(xnear, xnew):
                new_inspected_points = self.bb.compute_union_of_points(self.bb.get_inspected_points(xnew),self.tree.vertices[near_idx].inspected_points)
                new_idx = self.tree.add_vertex(xnew,new_inspected_points)
                inspected_points.update(map(tuple,new_inspected_points))
                self.tree.add_edge(near_idx,new_idx)
            else:
                continue
            if self.tree.max_coverage > self.coverage:
                path = [self.tree.vertices[self.tree.max_coverage_id].config]
                curr_idx = self.tree.max_coverage_id
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

