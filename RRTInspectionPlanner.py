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
                xrand = self.sample_biased_config(uninspected_points)
            else:
                xrand = self.bb.sample_random_config(0,None)
                
            near_idx, xnear = self.tree.get_nearest_config(xrand)
            xnew = self.extend(xnear, xrand)
            
            if self.bb.edge_validity_checker(xnear, xnew):
                new_inspected_points = self.bb.compute_union_of_points(self.bb.get_inspected_points(xnew),self.tree.vertices[near_idx].inspected_points)
                new_idx = self.tree.add_vertex(xnew,new_inspected_points)
                inspected_points.update(map(tuple,new_inspected_points))
                self.tree.add_edge(near_idx,new_idx)
                if self.tree.max_coverage > self.coverage:
                    return self.extract_path()
            
        return []
    
    def sample_biased_config(self, uninspected_points):
        """
        Sample a configuration biased towards seeing uninspected points.
        @param uninspected_points: Set of points that haven't been inspected yet
        """
        if not uninspected_points:
            return self.bb.sample_random_config(0, None)

        best_config = None
        max_visible = 0

        for _ in range(10):
            test_config = self.bb.sample_random_config(0, None)
            visible_points = self.bb.get_inspected_points(test_config)

            if len(visible_points) == 0:
                continue

            visible_set = set(map(tuple, visible_points))
            num_visible = len(uninspected_points.intersection(visible_set))

            distance_from_current = self.bb.compute_distance(self.tree.vertices[self.tree.max_coverage_id].config, test_config)
            score = num_visible * (1.0 / (1 + 0.1 * distance_from_current))
        
            if score > max_visible:
                max_visible = num_visible
                best_config = test_config

            # Early exit if we found a configuration that sees multiple points
            if max_visible >= 3:
                return best_config

        return best_config if best_config is not None else self.bb.sample_random_config(0, None)

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

    def extract_path(self):
        """Extract the path from the tree"""
        path = [self.tree.vertices[self.tree.max_coverage_id].config]
        curr_idx = self.tree.max_coverage_id
        while curr_idx != 0:
            curr_idx = self.tree.edges[curr_idx]
            path.append(self.tree.vertices[curr_idx].config)
        return np.array(path[::-1])
    
    def prune_nodes(self):
        """Remove nodes that don't add to the visibility"""
        nodes_to_remove = set()
        for vid, vertex in self.tree.vertices.items():
            if vid == 0:  # Keep root
                continue
            parent_id = self.tree.edges[vid]
            parent_visibility = set(map(tuple, self.tree.vertices[parent_id].inspected_points))
            current_visibility = set(map(tuple, vertex.inspected_points))
            if len(current_visibility - parent_visibility) == 0:
                nodes_to_remove.add(vid)

        for vid in nodes_to_remove:
            self.tree.vertices.pop(vid)
            self.tree.edges.pop(vid)
            
    
    def optimized_plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states 
        in the configuration space. Uses lazy collision checking and efficient point tracking.
        '''
        start_inspected_points = self.bb.get_inspected_points(self.start)
        self.tree.add_vertex(self.start, start_inspected_points)

        all_points = set(map(tuple, self.env.inspection_points))
        inspected_points = set(map(tuple, start_inspected_points))
        target_size = int(len(all_points) * self.coverage)

        n = 10000 
        batch_size = 50  # Process configurations in batches
        candidates = []  # Store candidate configurations

        for i in range(n):
            # Generate candidate configurations in batches
            if len(candidates) == 0:
                candidates = []
                for _ in range(batch_size):
                    if np.random.uniform(0,1) < self.goal_prob:
                        uninspected_points = all_points - inspected_points
                        xrand = self.sample_biased_config(uninspected_points)
                    else:
                        xrand = self.bb.sample_random_config(0, None)

                    near_idx, xnear = self.tree.get_nearest_config(xrand)
                    xnew = self.extend(xnear, xrand)
                    candidates.append((xnear, xnew, near_idx))

            xnear, xnew, near_idx = candidates.pop()

            # Lazy collision checking
            if not self.bb.edge_validity_checker_lazy(xnear, xnew):
                continue
            # Full collision check only if lazy check passes
            if not self.bb.edge_validity_checker(xnear, xnew):
                continue
            
            # Compute new inspected points
            visible_points = self.bb.get_inspected_points(xnew)
            if len(visible_points) == 0:
                continue

            # Use pre-computed points from parent
            parent_points = self.tree.vertices[near_idx].inspected_points
            new_inspected_points = self.bb.compute_union_of_points(visible_points, parent_points)

            # Quick size check before set operations
            if len(new_inspected_points) <= len(parent_points):
                continue

            # Add vertex and update points
            new_idx = self.tree.add_vertex(xnew, new_inspected_points)
            inspected_points.update(map(tuple, new_inspected_points))
            self.tree.add_edge(near_idx, new_idx)
            
            if len(inspected_points) >= target_size:
                return self.extract_path()

            # Periodically prune redundant nodes (every 1000 iterations)
            if i % 1000 == 999:
                self.prune_nodes()

        return []
    
    
    def sample_biased_config_ik(self, uninspected_points):
        """
        Sample configuration biased towards seeing uninspected points using IK
        @param uninspected_points: Set of points that haven't been inspected yet
        @return: A configuration or None if no solution found
        """
        if not uninspected_points:
            return self.bb.sample_random_config(0, None)

        points_array = np.array(list(uninspected_points))

        # Find point with most uninspected neighbors
        best_score = -1
        best_point_idx = 0
        for i, point in enumerate(points_array):
            neighbors = 0
            for other_point in points_array:
                if not np.array_equal(point, other_point):
                    dist = np.linalg.norm(point - other_point)
                    if dist < self.bb.vis_dist:
                        neighbors += 1
            if neighbors > best_score:
                best_score = neighbors
                best_point_idx = i

        target_point = points_array[best_point_idx]
        
        candidate_sols = self.bb.compute_ik_for_point(target_point)
        if len(candidate_sols) == 0:
            return self.bb.sample_random_config(0, None)
        sols = []
        for sol in candidate_sols:
            if not self.bb.config_validity_checker(sol):
                continue
            if np.max(sol) > np.pi or np.min(sol) < -np.pi:
                continue
            sols.append(sol)

        if not sols:
            return self.bb.sample_random_config(0, None)

        # From valid solutions, pick the one that can see the most points
        final_sol = None
        max_visible = 0
        for sol in sols:
            visible_points = self.bb.get_inspected_points(sol)
            visible_set = set(map(tuple, visible_points))
            num_visible = len(uninspected_points.intersection(visible_set))

            if num_visible > max_visible:
                max_visible = num_visible
                final_sol = sol

        return final_sol if final_sol is not None else self.bb.sample_random_config(0, None)