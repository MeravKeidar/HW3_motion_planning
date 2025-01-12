import numpy as np
import os
from datetime import datetime
from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner
from twoD.visualizer import Visualizer
import time
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}

def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.get_expanded_nodes(), show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01)
    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01)
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.01, k=1, max_step_size=None)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=1)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1 )

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------

    rrt_star_planner = RRTStarPlanner(max_step_size=0.5,
                                      start=env2_start,
                                      goal=env2_goal,
                                      max_itr=4000,
                                      stop_on_goal=True,
                                      bb=bb,
                                      goal_prob=0.05,
                                      ext_mode="E2")

    path = rrt_star_planner.plan()

    if path is not None:

        # create a folder for the experiment
        # Format the time string as desired (YYYY-MM-DD_HH-MM-SS)
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

        # create the folder
        exps_folder_name = os.path.join(os.getcwd(), "exps")
        if not os.path.exists(exps_folder_name):
            os.mkdir(exps_folder_name)
        exp_folder_name = os.path.join(exps_folder_name, "exp_pbias_"+ str(bb.p_bias) + "_max_step_size_" + str(rrt_star_planner.max_step_size) + "_" + time_str)
        if not os.path.exists(exp_folder_name):
            os.mkdir(exp_folder_name)

        # save the path
        np.save(os.path.join(exp_folder_name, 'path'), path)

        # save the cost of the path and time it took to compute
        with open(os.path.join(exp_folder_name, 'stats'), "w") as file:
            file.write("Path cost: {} \n".format(rrt_star_planner.compute_cost()))

        visualizer.show_path(path)

def run_dot_2d_rrt(ext_mode, goal_prob):
    start_time = time.time()
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob)
    # execute plan
    plan = planner.plan()
    execution_time = time.time() - start_time
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)
    return plan, planner.compute_cost(plan), execution_time

def run_2d_rrt_motion_planning(ext_mode, goal_prob):
    start_time = time.time()
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob)
    # execute plan
    plan = planner.plan()
    execution_time = time.time() - start_time
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])
    return plan, planner.compute_cost(plan), execution_time

def run_rrt_experiments():
    # Part 1: Extend function comparison using dot environment
    print("\nPart 1: Extend Function Comparison (Dot Environment)")
    print("==================================================")
    
    extend_configs = [
        ("E1", 0.05),
        ("E1", 0.20),
        ("E2", 0.05),
        ("E2", 0.20)
    ]
    
    for ext_mode, goal_prob in extend_configs:
        print(f"\nTesting {ext_mode} with {goal_prob*100}% goal bias:")
        plan, cost, time = run_dot_2d_rrt(ext_mode, goal_prob)
        print(f"Cost: {cost:.2f}")
        print(f"Time: {time:.2f} seconds")
        print(f"Path shape: {plan.shape}")
    
    # Part 2: Performance analysis with 2D manipulator
    print("\nPart 2: Performance Analysis (2D Manipulator)")
    print("============================================")
    
    results = {}
    num_trials = 10
    
    for ext_mode, goal_prob in extend_configs:
        key = f"{ext_mode}_goal{int(goal_prob*100)}"
        costs = []
        times = []
        
        print(f"\nRunning {num_trials} trials for {ext_mode} with {goal_prob*100}% goal bias")
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            plan, cost, execution_time = run_2d_rrt_motion_planning(ext_mode, goal_prob)
            costs.append(cost)
            times.append(execution_time)
        
        results[key] = {
            'mean_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
        
        print(f"\nResults for {key}:")
        print(f"Average Cost: {results[key]['mean_cost']:.2f} ± {results[key]['std_cost']:.2f}")
        print(f"Average Time: {results[key]['mean_time']:.2f}s ± {results[key]['std_time']:.2f}s")
    
    return results

if __name__ == "__main__":
    # run_dot_2d_astar()
    # run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    #run_2d_rrt_motion_planning()\
    # analyze_rrt_performance()
    # run_2d_rrt_inspection_planning()
    # run_3d()
    results = run_rrt_experiments()
