import numpy as np


class BuildingBlocks3D(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # TODO: HW2 5.2.1
        to_bias = np.random.uniform(low = 0, high = 1)
        if to_bias <= goal_prob:
            return goal_conf
        config = []
        for joint_limit in self.ur_params.mechamical_limits.values():
            config.append(np.random.uniform(low = joint_limit[0], high = joint_limit[1]))
        return config

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
        coords = self.transform.conf2sphere_coords(conf)
        radii = self.ur_params.sphere_radius
        obstacle_rad = self.env.radius
        for joint,values in coords.items():
            for center in values:
                if center[0] >= 0.4: #wall in manipulator env, HW3 relevant
                    return False
                if center[0] !=0 and center[1] != 0 and center[2] < radii[joint]:
                    #print(f'{joint} hit floor at {center}, dist should be more than {radii[joint]}')
                    return False
                for obstacle in self.env.obstacles:
                    obstacle_dist = np.sqrt(np.sum((center - obstacle)**2))
                    rad_sum = radii[joint] + obstacle_rad
                    if  obstacle_dist < rad_sum:
                        #print(f'{joint} is {coords[joint]} ')
                        #print(f'center is {center} , obstacle is {obstacle} , distance is {obstacle_dist} , should be more than {rad_sum}')
                        return False
        for collision in self.possible_link_collisions:
            joint1 = collision[0]
            joint2 = collision[1]
            rad_sum = radii[joint1] +  radii[joint2]
            joint1_centers = coords[joint1]
            joint2_centers = coords[joint2]
            for i in range(len(joint1_centers)):
                for j in range(i+1,len(joint2_centers)):
                    center1 = joint1_centers[i]
                    center2 = joint2_centers[j]
                    actual_distance = np.sqrt(np.sum((center1 - center2)**2))
                    if  actual_distance < rad_sum:
                        #print(f'{joint1} is {joint1_centers} and {joint2} is {joint2_centers}')
                        #print(f'center 1 is {center1} , center2 is {center2} , distance is {actual_distance} , should be more than {rad_sum}')
                        return False
                            
        return True


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # TODO: HW2 5.2.4
        if not self.config_validity_checker(current_conf) :
            #print('current conf in collision')
            return False
        if not self.config_validity_checker(prev_conf):
            #print('prev conf in collision')
            return False
        res = self.resolution
        num_steps = max(3,int(self.compute_distance(prev_conf, current_conf)/res))
        #print(f'resolution is {res} , and num steps is {num_steps}')
        for i in range(num_steps):
            t = i/ (num_steps -1)

            temp_conf = [prev_conf[j] + t * (current_conf[j] - prev_conf[j]) for j in range(len(prev_conf))]
            if not self.config_validity_checker(temp_conf):
                #print(f'failed at step {i}')
                return False
        #print('transition approved!')
        return True

    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
