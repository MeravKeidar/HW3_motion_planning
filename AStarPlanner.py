import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        
        self.epsilon = 20
        plan = self.a_star(self.start, self.goal)
        return np.array(plan)

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        '''
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        '''
        return self.bb.compute_distance(state, self.goal)
    
    def a_star(self, start_loc, goal_loc):
        open_set = []
        closed_set = set()
        came_from = {}
        
        g_score = {tuple(start_loc): 0}
        f_score = {tuple(start_loc): self.epsilon * self.compute_heuristic(start_loc)}
        heapq.heappush(open_set, (f_score[tuple(start_loc)], tuple(start_loc)))
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            current_np = np.array(current)
            # Add to expanded nodes 
            if current_np.tolist() not in [node.tolist() for node in self.expanded_nodes]:
                self.expanded_nodes.append(current_np)
            # Check if goal reached
            if np.array_equal(current_np, goal_loc):
                path = []
                while current in came_from:
                    path.append(np.array(current))
                    current = came_from[current]
                path.append(start_loc)
                return path[::-1]
            closed_set.add(current)

            # Check all possible moves
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                neighbor_np = np.array(neighbor)
                if tuple(neighbor) in closed_set:
                    continue
                if not self.bb.config_validity_checker(neighbor_np):
                    continue
                temp_g_score = g_score[current] + np.sqrt(dx**2 + dy**2)
                
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:  # This path is better than previous ones
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = g_score[neighbor] + self.epsilon * self.compute_heuristic(neighbor_np)
                    if neighbor not in [x[1] for x in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
        return []

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''
        # used for visualizing the expanded nodes
        return self.expanded_nodes
