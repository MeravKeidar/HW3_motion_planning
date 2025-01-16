import itertools
import numpy as np
from shapely.geometry import Point, LineString

class BuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # define robot properties
        self.links = np.array([80.0, 70.0, 40.0, 40.0])
        self.dim = len(self.links)

        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        '''
        Compute the euclidean distance betweeen two given configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        '''
        # HW2 4.2.1
        return np.sqrt(np.sum((prev_config - next_config) ** 2))
    
    def sample_random_config(self, goal_prob, goal):
        # HW3 2.1
        if np.random.uniform(0,1) < goal_prob:
            return goal
        
        q1 = np.random.uniform(-np.pi, np.pi)
        q2 = np.random.uniform(-np.pi, np.pi) 
        q3 = np.random.uniform(-np.pi, np.pi)
        q4 = np.random.uniform(-np.pi, np.pi)
    
        config = np.array([q1, q2, q3, q4])
        
        if not self.config_validity_checker(config): # if not valid, sample again
            return self.sample_random_config(goal_prob, goal)
        
        return config
    
    def compute_path_cost(self, path):
        totat_cost = 0
        for i in range(len(path) - 1):
            totat_cost += self.compute_distance(path[i], path[i + 1])
        return totat_cost

    def compute_forward_kinematics(self, given_config):
        '''
        Compute the 2D position (x,y) of each one of the links (including end-effector) and return.
        @param given_config Given configuration.
        '''
        # positions are 2D points + angle (3 dimensions) for each of the links of the robot
        # HW2 4.2.2
        res = np.zeros((4, 2))
        curr_x = 0
        curr_y = 0
        angle = given_config[0]

        for i in range(len(self.links)):
            curr_x += self.links[i] * np.cos(angle)
            curr_y += self.links[i] * np.sin(angle)
            res[i] = [curr_x, curr_y]
            if i < len(given_config) - 1:
                angle = self.compute_link_angle(angle, given_config[i+1])
        return res

    def compute_ee_angle(self, given_config):
        '''
        Compute the 1D orientation of the end-effector w.r.t. world origin (or first joint)
        @param given_config Given configuration.
        '''
        ee_angle = given_config[0]
        for i in range(1, len(given_config)):
            ee_angle = self.compute_link_angle(ee_angle, given_config[i])

        return ee_angle

    def compute_link_angle(self, link_angle, given_angle):
        '''
        Compute the 1D orientation of a link given the previous link and the current joint angle.
        @param link_angle previous link angle.
        @param given_angle Given joint angle.
        '''
        if link_angle + given_angle > np.pi:
            return link_angle + given_angle - 2 * np.pi
        elif link_angle + given_angle < -np.pi:
            return link_angle + given_angle + 2 * np.pi
        else:
            return link_angle + given_angle

    def validate_robot(self, robot_positions):
        '''
        Verify that the given set of links positions does not contain self collisions.
        @param robot_positions Given links positions.
        '''
        # HW2 4.2.1
        lines = []
        robot_positions = np.concatenate([np.zeros((1, 2)), robot_positions])
        for i in range(1,len(robot_positions)):
            lines.append(LineString([robot_positions[i], robot_positions[i-1]]))
            
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if (lines[i].crosses(lines[j]) or 
                    lines[i].overlaps(lines[j]) or
                    lines[i].contains(lines[j]) or 
                    lines[j].contains(lines[i])):
                    return False
        return True

    def config_validity_checker(self, config):
        '''
        Verify that the config (given or stored) does not contain self collisions or links that are out of the world boundaries.
        Return false if the config is not applicable, and true otherwise.
        @param config The given configuration of the robot.
        '''
        # compute robot links positions
        robot_positions = self.compute_forward_kinematics(given_config=config)

        # add position of robot placement ([0,0] - position of the first joint)
        robot_positions = np.concatenate([np.zeros((1,2)), robot_positions])

        # verify that the robot do not collide with itself
        if not self.validate_robot(robot_positions=robot_positions):
            return False

        # verify that all robot joints (and links) are between world boundaries
        non_applicable_poses = [(x[0] < self.env.xlimit[0] or x[1] < self.env.ylimit[0] or x[0] > self.env.xlimit[1] or x[1] > self.env.ylimit[1]) for x in robot_positions]
        if any(non_applicable_poses):
            return False
 
        # verify that all robot links do not collide with obstacle edges
        # for each obstacle, check collision with each of the robot links
        robot_links = [LineString([Point(x[0],x[1]),Point(y[0],y[1])]) for x,y in zip(robot_positions.tolist()[:-1], robot_positions.tolist()[1:])]
        for obstacle_edges in self.env.obstacles_edges:
            for robot_link in robot_links:
                obstacle_collisions = [robot_link.crosses(x) for x in obstacle_edges]
                if any(obstacle_collisions):
                    return False

        return True

    def edge_validity_checker_lazy(self, config1, config2):
        """Simplified initial collision check before detailed validation"""
        # Quick check of endpoints first
        if not (self.config_validity_checker(config1) and self.config_validity_checker(config2)):
            return False

        # Coarse interpolation first
        required_diff = 0.2  # Larger steps for initial check
        interpolation_steps = int(np.linalg.norm(config2 - config1) // required_diff)

        if interpolation_steps > 0:
            interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)
            for config in interpolated_configs:
                if not self.config_validity_checker(config):
                    return False

        return True  # If coarse check passes, do detailed check

    def edge_validity_checker(self, config1, config2):
        '''
        A function to check if the edge between two configurations is free from collisions. The function will interpolate between the two states to verify
        that the links during motion do not collide with anything.
        @param config1 The source configuration of the robot.
        @param config2 The destination configuration of the robot.
        '''
        # interpolate between first config and second config to verify that there is no collision during the motion
        required_diff = 0.05
        interpolation_steps = int(np.linalg.norm(config2 - config1) // required_diff)
        if interpolation_steps > 0:
            interpolated_configs = np.linspace(start=config1, stop=config2, num=interpolation_steps)

            # compute robot links positions for interpolated configs
            configs_positions = np.apply_along_axis(self.compute_forward_kinematics, 1, interpolated_configs)

            # compute edges between joints to verify that the motion between two configs does not collide with anything
            edges_between_positions = []
            for j in range(self.dim):
                for i in range(interpolation_steps - 1):
                    edges_between_positions.append(LineString(
                        [Point(configs_positions[i, j, 0], configs_positions[i, j, 1]),
                         Point(configs_positions[i + 1, j, 0], configs_positions[i + 1, j, 1])]))

            # check collision for each edge between joints and each obstacle
            for edge_pos in edges_between_positions:
                for obstacle_edges in self.env.obstacles_edges:
                    obstacle_collisions = [edge_pos.crosses(x) for x in obstacle_edges]
                    if any(obstacle_collisions):
                        return False

            # add position of robot placement ([0,0] - position of the first joint)
            configs_positions = np.concatenate([np.zeros((len(configs_positions), 1, 2)), configs_positions], axis=1)

            # verify that the robot do not collide with itself during motion
            for config_positions in configs_positions:
                if not self.validate_robot(config_positions):
                    return False

            # verify that all robot joints (and links) are between world boundaries
            if len(np.where(configs_positions[:, :, 0] < self.env.xlimit[0])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 1] < self.env.ylimit[0])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 0] > self.env.xlimit[1])[0]) > 0 or \
                    len(np.where(configs_positions[:, :, 1] > self.env.ylimit[1])[0]) > 0:
                return False

        return True

    def get_inspected_points(self, config):
        '''
        A function to compute the set of points that are visible to the robot with the given configuration.
        The function will return the set of points that is visible in terms of distance and field of view (FOV) and are not hidden by any obstacle.
        @param config The given configuration of the robot.
        '''
        # get robot end-effector position and orientation for point of view
        ee_pos = self.compute_forward_kinematics(given_config=config)[-1]
        ee_angle = self.compute_ee_angle(given_config=config)

        # define angle range for the ee given its position and field of view (FOV)
        ee_angle_range = np.array([ee_angle - self.ee_fov / 2, ee_angle + self.ee_fov / 2])

        # iterate over all inspection points to find which of them are currently inspected
        inspected_points = np.array([])
        for inspection_point in self.env.inspection_points:

            # compute angle of inspection point w.r.t. position of ee
            relative_inspection_point = inspection_point - ee_pos
            inspection_point_angle = self.compute_angle_of_vector(vec=relative_inspection_point)

            # check that the point is potentially visible with the distance from the end-effector
            if np.linalg.norm(relative_inspection_point) <= self.vis_dist:

                # if the resulted angle is between the angle range of the ee, verify that there are no interfering obstacles
                if self.check_if_angle_in_range(angle=inspection_point_angle, ee_range=ee_angle_range):

                    # define the segment between the inspection point and the ee
                    ee_to_inspection_point = LineString(
                        [Point(ee_pos[0], ee_pos[1]), Point(inspection_point[0], inspection_point[1])])

                    # check if there are any collisions of the vector with some obstacle edge
                    inspection_point_hidden = False
                    for obstacle_edges in self.env.obstacles_edges:
                        for obstacle_edge in obstacle_edges:
                            if ee_to_inspection_point.intersects(obstacle_edge):
                                inspection_point_hidden = True

                    # if inspection point is not hidden by any obstacle, add it to the visible inspection points
                    if not inspection_point_hidden:
                        if len(inspected_points) == 0:
                            inspected_points = np.array([inspection_point])
                        else:
                            inspected_points = np.concatenate([inspected_points, [inspection_point]], axis=0)

        return inspected_points

    def compute_angle_of_vector(self, vec):
        '''
        A utility function to compute the angle of the vector from the end-effector to a point.
        @param vec Vector from the end-effector to a point.
        '''
        vec = vec / np.linalg.norm(vec)
        if vec[1] > 0:
            return np.arccos(vec[0])
        else:  # vec[1] <= 0
            return -np.arccos(vec[0])

    def check_if_angle_in_range(self, angle, ee_range):
        '''
        A utility function to check if an inspection point is inside the FOV of the end-effector.
        @param angle The angle beteen the point and the end-effector.
        @param ee_range The FOV of the end-effector.
        '''
        # ee range is in the expected order
        if abs((ee_range[1] - self.ee_fov) - ee_range[0]) < 1e-5:
            if angle < ee_range.min() or angle > ee_range.max():
                return False
        # ee range reached the point in which pi becomes -pi
        else:
            if angle > ee_range.min() or angle < ee_range.max():
                return False

        return True

    def compute_union_of_points(self, points1, points2):
        '''
        Compute a union of two sets of inpection points.
        @param points1 list of inspected points.
        @param points2 list of inspected points.
        '''
        # TODO: HW3 2.3.2
        points1_set = set(map(tuple, points1))
        points2_set = set(map(tuple, points2))
        union_set = points1_set.union(points2_set)
        return np.array(list(union_set))

    def compute_coverage(self, inspected_points):
        '''
        Compute the coverage of the map as the portion of points that were already inspected.
        @param inspected_points list of inspected points.
        '''
        return len(inspected_points) / len(self.env.inspection_points)
    
    def compute_ik_for_point(self, point):
        """
        Compute inverse kinematics solutions for 4-DOF planar robot to reach viewing position for a point
        @param point: The target point [x, y] to look at
        @return: numpy array of valid configurations that might see the point
        """
        try:
            if not isinstance(point, np.ndarray) or point.shape != (2,):
                return []
            if not np.all(np.isfinite(point)):
                return []

            theta = np.matrix(np.zeros((4, 8)))
            optimal_distance = self.vis_dist * 0.7  # hyperparameter for viewing distance

            if np.all(np.abs(point) < 1e-10):  # point at origin
                return []

            ee_angle = np.arctan2(point[1], point[0])
            if not np.isfinite(ee_angle):
                return []

            ee_x = point[0] - optimal_distance * np.cos(ee_angle)
            ee_y = point[1] - optimal_distance * np.sin(ee_angle)

            # wrist position
            wx = ee_x - self.links[3] * np.cos(ee_angle)
            wy = ee_y - self.links[3] * np.sin(ee_angle)

            if not (np.isfinite(wx) and np.isfinite(wy)):
                return []

            # base angle and distance to wrist
            psi = np.arctan2(wy, wx)
            if not np.isfinite(psi):
                return []

            d = np.sqrt(wx**2 + wy**2)
            if not np.isfinite(d):
                return []

            max_reach = self.links[0] + self.links[1] + self.links[2]
            min_reach = abs(self.links[0] - self.links[1] - self.links[2])
            if d > max_reach or d < min_reach:
                return []

            # first joint angles
            if abs(d) < 1e-10 or d > 2 * self.links[0]:
                return []

            cos_phi = d / (2 * self.links[0])
            if abs(cos_phi) > 1:
                return []

            phi = np.arccos(cos_phi)
            if not np.isfinite(phi):
                return []

            # Set first joint angles for all 8 possible solutions
            theta[0, 0:4] = psi + phi
            theta[0, 4:8] = psi - phi

            solutions = []
            # remaining joint angles
            for i in range(8):
                try:
                    # First link endpoint
                    x1 = self.links[0] * np.cos(theta[0, i])
                    y1 = self.links[0] * np.sin(theta[0, i])

                    if not (np.isfinite(x1) and np.isfinite(y1)):
                        continue

                    # Distance and angle to wrist
                    dx = wx - x1
                    dy = wy - y1
                    d2 = np.sqrt(dx*dx + dy*dy)
                    if not np.isfinite(d2) or d2 < 1e-10:
                        continue

                    alpha2 = np.arctan2(dy, dx)
                    if not np.isfinite(alpha2):
                        continue
                    
                    # Second joint angle using cosine law
                    cos_beta2 = (d2*d2 + self.links[1]*self.links[1] - self.links[2]*self.links[2]) / (2 * d2 * self.links[1])
                    if abs(cos_beta2) <= 1:
                        beta2 = np.arccos(cos_beta2)
                        if not np.isfinite(beta2):
                            continue

                        # remaining joint angles
                        theta[1, i] = alpha2 + (-1)**(i//2) * beta2 - theta[0, i]
                        theta[2, i] = ee_angle - theta[0, i] - theta[1, i]
                        theta[3, i] = ee_angle - sum(theta[:3, i])

                        if not np.all(np.isfinite(theta[:, i])):
                            continue
                        
                        # Normalize
                        config = np.array(theta[:, i]).flatten()
                        config = np.mod(config + np.pi, 2 * np.pi) - np.pi
 
                        if self.config_validity_checker(config):
                            visible_points = self.get_inspected_points(config)
                            if len(visible_points) > 0:
                                for vp in visible_points:
                                    if np.linalg.norm(point - vp) < 1e-6:
                                        solutions.append(config)
                                        break
                                    
                except Exception:
                    continue

            return np.array(solutions) if solutions else []

        except Exception:
            return []