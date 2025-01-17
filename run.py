import gc
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
import json
from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')


# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}

def run_dot_2d_astar():
    start = time.time()
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])
    
    # execute plan
    plan = planner.plan()
    if len(plan) == 0:
        return float('inf') 
    print(f'epsilon is {planner.epsilon}')
    print(f'time diff is : {time.time() - start}')
    print(f'cost is  {sum(bb.compute_distance(plan[i], plan[i+1]) for i in range(len(plan)-1))}')
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.get_expanded_nodes(), show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    start = time.time()
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01)
    # execute plan
    plan = planner.plan()
    if len(plan) == 0:
        return float('inf') 
    print(f'eta is {planner.eta}')
    print(f'time diff is : {time.time() - start}')
    print(f'cost is  {sum(bb.compute_distance(plan[i], plan[i+1]) for i in range(len(plan)-1))}')
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
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5,env = planning_env)
    # execute plan
    plan = planner.plan()
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.01, k=1, max_step_size=None)
    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=False)

# def run_3d():
    options = [
        # (0.05,0.05),
        # (0.075,0.05),
        # (0.1,0.05),
        # (0.125,0.05),
        # (0.2,0.05),
        # (0.25,0.05),
        # (0.3,0.05),
        # (0.4,0.05),
        # (0.05,0.2),
        # (0.075,0.2),
        # (0.1,0.2),
        # (0.125,0.2),
        # (0.2,0.2),
        # (0.25,0.2),
        # (0.3,0.2),
        (0.4,0.2)
    ]
    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------
    now_3d = time.gmtime()
    with open(f'3d_experiment_results.txt', 'w') as f:        
        results = {}
        num_trials = 5
        
        for step, goal in options:
            key = f"{step}_goal{int(goal*100)}"
            costs = []
            times = []
            successes = 0 
            
            f.write(f"\nRunning {num_trials} trials for {step} step size with {goal*100}% goal bias\n")
            
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}")
                visualize = False
                if trial == 0:
                    visualize=True
                plan, cost, execution_time = run_3d_experiment(step,goal,visualize)
                if plan is not None and len(plan) > 0:
                    costs.append(cost)
                    times.append(execution_time)
                    successes +=1
            success_rate = (successes / num_trials) * 100
            if successes > 0: 
                results[key] = {
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'success_rate': success_rate
                }
                
                f.write(f"\nResults for {key}:\n")
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Average Cost: {results[key]['mean_cost']:.2f} ± {results[key]['std_cost']:.2f}\n")
                f.write(f"Average Time: {results[key]['mean_time']:.2f}s ± {results[key]['std_time']:.2f}s\n")
            else:
                f.write(f"\nResults for {key}:\n")
                f.write("No successful trials\n")
    
def run_3d_experiment(step, goal, visualize = True):
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                            ur_params=ur_params,
                            env=env,
                            resolution=0.1 )
    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
    rrt_star_planner = RRTStarPlanner(max_step_size=step,
                                        start=env2_start,
                                        goal=env2_goal,
                                        max_itr=2000,
                                        stop_on_goal=True,
                                        bb=bb,
                                        goal_prob=goal,
                                        ext_mode="E2")
    # execute plan
    start_time = time.time()
    plan = rrt_star_planner.plan()
    execution_time = time.time() - start_time
    if visualize:
        if plan is not None and len(plan) > 0:
            visualizer.show_path(plan)
            # plt.show()
            # filename = f"experiment_step_{str(step).replace('.', '_')}_goal_{str(goal).replace('.', '_')}.png"
            # plt.savefig(filename)
            # plt.close()
        else:
            print("plan failed \n")
    return plan, rrt_star_planner.compute_cost(plan), execution_time

def run_dot_2d_rrt_experiment(ext_mode, goal_prob):
   start_time = time.time()
   planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
   bb = DotBuildingBlocks2D(planning_env)
   planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob)
   # execute plan
   plan = planner.plan()
   execution_time = time.time() - start_time
   DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)
   return plan, planner.compute_cost(plan), execution_time

def run_2d_rrt_motion_planning_experiment(ext_mode, goal_prob):
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
        plan, cost, time = run_dot_2d_rrt_experiment(ext_mode, goal_prob)
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
            plan, cost, execution_time = run_2d_rrt_motion_planning_experiment(ext_mode, goal_prob)
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

def run_dot_2d_rrt_experiment(ext_mode, goal_prob, visualize = False):
    start_time = time.time()
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob)
    # execute plan
    plan = planner.plan()
    execution_time = time.time() - start_time
    if plan is not None and len(plan) > 0 and visualize:
        save_path = f'rrt_{ext_mode}_goal{int(goal_prob*100)}.png'
        DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), save_path=save_path)
    return plan, planner.compute_cost(plan), execution_time

def run_2d_rrt_motion_planning_experiment(ext_mode, goal_prob, visualize = False):
    start_time = time.time()
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob)
    # execute plan
    plan = planner.plan()
    execution_time = time.time() - start_time
    if plan is not None and len(plan) > 0 and visualize:
        save_path = f'rrt_manipulator_{ext_mode}_goal{int(goal_prob*100)}.gif'
        Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], save_path=save_path)
    return plan, planner.compute_cost(plan), execution_time

def run_rrt_experiments():
    with open('rrt_experiment_results.txt', 'w') as f:
        # Part 1: Extend function comparison using dot environment
        f.write("\nPart 1: Extend Function Comparison (Dot Environment)\n")
        f.write("==================================================\n")
        
        extend_configs = [
            ("E1", 0.05),
            ("E1", 0.20),
            ("E2", 0.05),
            ("E2", 0.20)
        ]
        
        for ext_mode, goal_prob in extend_configs:
            f.write(f"\nTesting {ext_mode} with {goal_prob*100}% goal bias:\n")
            plan, cost, time = run_dot_2d_rrt_experiment(ext_mode, goal_prob)
            f.write(f"Cost: {cost:.2f}\n")
            f.write(f"Time: {time:.2f} seconds\n")
        
        # Part 2: Performance analysis with 2D manipulator
        f.write("\nPart 2: Performance Analysis (2D Manipulator)\n")
        f.write("============================================\n")
        
        results = {}
        num_trials = 10
        
        for ext_mode, goal_prob in extend_configs:
            key = f"{ext_mode}_goal{int(goal_prob*100)}"
            costs = []
            times = []
            successes = 0 
            
            f.write(f"\nRunning {num_trials} trials for {ext_mode} with {goal_prob*100}% goal bias\n")
            
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}")
                visualize = False
                # if trial == 0:
                #     visualize=True
                plan, cost, execution_time = run_2d_rrt_motion_planning_experiment(ext_mode, goal_prob,visualize)
                if plan is not None and len(plan) > 0:
                    costs.append(cost)
                    times.append(execution_time)
                    successes +=1
            success_rate = (successes / num_trials) * 100
            if successes > 0: 
                results[key] = {
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'success_rate': success_rate
                }
                
                f.write(f"\nResults for {key}:\n")
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Average Cost: {results[key]['mean_cost']:.2f} ± {results[key]['std_cost']:.2f}\n")
                f.write(f"Average Time: {results[key]['mean_time']:.2f}s ± {results[key]['std_time']:.2f}s\n")
            else:
                f.write(f"\nResults for {key}:\n")
                f.write("No successful trials\n")

def run_dot_2d_rrt_star_experiment(ext_mode, goal_prob,k, visualize = False):
    start_time = time.time()
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob, k=k, max_step_size=None)
    # execute plan
    plan = planner.plan()
    execution_time = time.time() - start_time
    if plan is not None and len(plan) > 0 and visualize:
        save_dir = "rrt_star"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'rrt_star_dot_{ext_mode}_goal{int(goal_prob*100)}_k{k}.png')

        DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), save_path=save_path)
    return plan, planner.compute_cost(plan), execution_time

def run_2d_rrt_star_experiment(ext_mode, goal_prob,k, visualize = False):
    start_time = time.time()
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode=ext_mode, goal_prob=goal_prob, k=k, max_step_size=None)
    # execute plan
    plan = planner.plan()
    execution_time = time.time() - start_time
    if plan is not None and len(plan) > 0 and visualize:
        save_dir = "rrt_star"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'rrt_star_manipulator_{ext_mode}_goal{int(goal_prob*100)}_k{k}.gif')
        Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], save_path=save_path)
    return plan, planner.compute_cost(plan), execution_time

def run_rrt_star_experiments():
    results_dir = "rrt_star"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open('rrt_star_experiment_results.txt', 'w') as f:
        # Part 1: Dot environment
        f.write("\nPart 1: Extend Function Comparison (Dot Environment)\n")
        f.write("==================================================\n")
        
        extend_configs = [
            ("E1", 0.05),
            ("E1", 0.20),
            ("E2", 0.05),
            ("E2", 0.20)
        ]
        configs = [
            ("E2", 0.20,10),
            ("E2", 0.20,20),
            ("E2", 0.20,50)
        ]
        for ext_mode, goal_prob, k  in configs:
            f.write(f"\nTesting {ext_mode} with {goal_prob*100}% goal bias, and k = {k}\n")
            plan, cost, time = run_dot_2d_rrt_star_experiment(ext_mode, goal_prob,k, visualize= False)
            f.write(f"Cost: {cost:.2f}\n")
            f.write(f"Time: {time:.2f} seconds\n")
            
        # Part 2: Performance analysis with 2D manipulator
        f.write("\nPart 2: Performance Analysis (2D Manipulator)\n")
        f.write("============================================\n")
        
        results = {}
        num_trials = 10
        
        for ext_mode, goal_prob, k in configs:
            key = f"{ext_mode}_goal{int(goal_prob*100)}_k{k}"
            costs = []
            times = []
            successes = 0 
            f.write(f"\nRunning {num_trials} trials for {ext_mode} with {goal_prob*100}% goal bias, and k = {k}\n")
            
            for trial in range(num_trials):
                print(f"Trial {trial + 1}/{num_trials}")
                visualize = False
                if trial == 0:
                    visualize = True
                plan, cost, execution_time = run_2d_rrt_star_experiment(ext_mode, goal_prob,k,visualize = False)
                if plan is not None and len(plan) > 0:
                    costs.append(cost)
                    times.append(execution_time)
                    successes +=1
            success_rate = (successes / num_trials) * 100
            if successes > 0: 
                results[key] = {
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'success_rate': success_rate
                }
                
                f.write(f"\nResults for {key}:\n")
                f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write(f"Average Cost: {results[key]['mean_cost']:.2f} ± {results[key]['std_cost']:.2f}\n")
                f.write(f"Average Time: {results[key]['mean_time']:.2f}s ± {results[key]['std_time']:.2f}s\n")
            else:
                f.write(f"\nResults for {key}:\n")
                f.write("No successful trials\n")     

def run_inspection_comparison():
    """Compare different RRT inspection planning methods"""
    results_dir = "inspection_comparison_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test parameters
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0])}
    num_trials = 10
    coverage_levels = [0.5, 0.75]
    goal_probs = [0.2, 0.3]
    
    # All combinations of methods
    methods = {
        'original_regular': {'planner': 'plan', 'sampling': 'regular'},
        'original_ik': {'planner': 'plan', 'sampling': 'ik'},
        'optimized_regular': {'planner': 'optimized_plan', 'sampling': 'regular'},
        'optimized_ik': {'planner': 'optimized_plan', 'sampling': 'ik'}
    }
    
    results = {method: {} for method in methods.keys()}
    labels = []
    
    for coverage in coverage_levels:
        for goal_prob in goal_probs:
            print(f"\nTesting with coverage={coverage}, goal_prob={goal_prob}")
            label = f"Cov={coverage}\nProb={goal_prob}"
            labels.append(label)
            
            for method_name, method_config in methods.items():
                results[method_name][label] = {
                    'times': [],
                    'costs': [],
                    'successes': 0
                }
                
                for trial in range(num_trials):
                    print(f"Method: {method_name}, Trial {trial + 1}/{num_trials}")
                    
                    # Setup environment
                    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
                    bb = BuildingBlocks2D(planning_env)
                    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], 
                                                ext_mode="E2", goal_prob=goal_prob, 
                                                coverage=coverage, env=planning_env)
                    
                    # Set sampling method
                    if method_config['sampling'] == 'ik':
                        planner.sample_biased_config = planner.sample_biased_config_ik
                    
                    # Run appropriate planner method
                    start_time = time.time()
                    plan = getattr(planner, method_config['planner'])()
                    execution_time = time.time() - start_time
                    
                    if len(plan) > 0:
                        results[method_name][label]['times'].append(execution_time)
                        results[method_name][label]['costs'].append(planner.compute_cost(plan))
                        results[method_name][label]['successes'] += 1
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot execution times
    plt.subplot(131)
    x = np.arange(len(labels))
    width = 0.2
    for i, (method_name, method_data) in enumerate(results.items()):
        times = [np.mean(method_data[label]['times']) if method_data[label]['times'] else 0 for label in labels]
        plt.bar(x + i*width - width*1.5, times, width, label=method_name)
    plt.ylabel('Average Time (s)')
    plt.title('Execution Time Comparison')
    plt.xticks(x, labels, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot costs
    plt.subplot(132)
    for i, (method_name, method_data) in enumerate(results.items()):
        costs = [np.mean(method_data[label]['costs']) if method_data[label]['costs'] else 0 for label in labels]
        plt.bar(x + i*width - width*1.5, costs, width, label=method_name)
    plt.ylabel('Average Path Cost')
    plt.title('Path Cost Comparison')
    plt.xticks(x, labels, rotation=45)
    
    # Plot success rates
    plt.subplot(133)
    for i, (method_name, method_data) in enumerate(results.items()):
        success_rates = [(method_data[label]['successes'] / num_trials) * 100 for label in labels]
        plt.bar(x + i*width - width*1.5, success_rates, width, label=method_name)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.xticks(x, labels, rotation=45)
    
    plt.tight_layout()
    
    # Save results
    exp_name = f"comparison_results_{timestamp}"
    plt.savefig(os.path.join(results_dir, f"{exp_name}_plot.png"))
    
    # Save detailed results
    with open(os.path.join(results_dir, f"{exp_name}_summary.txt"), 'w') as f:
        for coverage in coverage_levels:
            for goal_prob in goal_probs:
                label = f"Cov={coverage}\nProb={goal_prob}"
                f.write(f"\nResults for coverage={coverage}, goal_prob={goal_prob}:\n")
                f.write("="*50 + "\n")
                
                for method_name, method_data in results.items():
                    f.write(f"\n{method_name}:\n")
                    data = method_data[label]
                    
                    if data['successes'] > 0:
                        avg_time = np.mean(data['times'])
                        std_time = np.std(data['times'])
                        avg_cost = np.mean(data['costs'])
                        std_cost = np.std(data['costs'])
                        success_rate = (data['successes'] / num_trials) * 100
                        
                        f.write(f"Success Rate: {success_rate:.1f}%\n")
                        f.write(f"Average Time: {avg_time:.2f}s ± {std_time:.2f}s\n")
                        f.write(f"Average Cost: {avg_cost:.2f} ± {std_cost:.2f}\n")
                    else:
                        f.write("No successful trials\n")
                f.write("\n" + "-"*50 + "\n")
    
    return results

def run_3d():

    options = [
        (0.05,0.05), (0.075,0.05), (0.1,0.05), (0.125,0.05),
        (0.2,0.05), (0.25,0.05), (0.3,0.05), (0.4,0.05),
        (0.05,0.2), (0.075,0.2), (0.1,0.2), (0.125,0.2),
        (0.2,0.2), (0.25,0.2), (0.3,0.2), (0.4,0.2)
    ]
    
    best_path = None
    best_cost = float('inf')
    best_config = None
    num_trials = 20
    
    # Process one configuration at a time
    for step, goal in options:
        print(f"\nProcessing step={step}, goal={goal}")
        costs = []
        times = []
        successes = 0
        
        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")
            
            plan, cost, execution_time = run_3d_experiment(step, goal, visualize=False)
            
            # Clear matplotlib memory
            plt.close('all')
            
            if plan is not None and len(plan) > 0:
                costs.append(cost)
                times.append(execution_time)
                successes += 1
                
                # Update best path if better
                if cost < best_cost:
                    best_cost = cost
                    best_config = (step, goal, trial)
                    np.save('best_path.npy', plan)
                    with open('best_path_info.txt', 'w') as f:
                        f.write(f"Step size: {step}\n")
                        f.write(f"Goal bias: {goal}\n")
                        f.write(f"Trial: {trial}\n")
                        f.write(f"Cost: {cost}\n")
            
            # Force garbage collection after each trial
            if trial % 5 == 0: 
                gc.collect()
        
        if successes > 0:
            with open('results.txt', 'a') as f:
                f.write(f"\nResults for step={step}, goal={goal}:\n")
                f.write(f"Success rate: {(successes/num_trials)*100:.1f}%\n")
                f.write(f"Average cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}\n")
                f.write(f"Average time: {np.mean(times):.2f}s ± {np.std(times):.2f}s\n")
            
            np.savez(f'config_results_{step}_{goal}.npz',
                    costs=np.array(costs),
                    times=np.array(times),
                    success_rate=(successes/num_trials)*100)
        
        # Clear all lists
        costs.clear()
        times.clear()
        gc.collect()
    
    plot_results(options)

def plot_results(options):
    """Create plots from saved configuration results"""
    # Create separate figures for each p_bias
    for p_bias in [0.05, 0.2]:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Results for p_bias = {p_bias}')
        
        step_sizes = []
        mean_costs = []
        mean_times = []
        success_rates = []
        
        # Load results for this p_bias
        for step, goal in options:
            if goal == p_bias:
                try:
                    data = np.load(f'config_results_{step}_{goal}.npz')
                    step_sizes.append(step)
                    mean_costs.append(np.mean(data['costs']))
                    mean_times.append(np.mean(data['times']))
                    success_rates.append(data['success_rate'])
                except:
                    continue
        
        # Create plots
        ax1.plot(mean_times, mean_costs, 'o-')
        ax1.set_xlabel('Computation Time (s)')
        ax1.set_ylabel('Path Cost')
        ax1.set_title('Cost vs Computation Time')
        ax1.grid(True)
        
        ax2.plot(mean_times, success_rates, 'o-')
        ax2.set_xlabel('Computation Time (s)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate vs Computation Time')
        ax2.grid(True)
        
        ax3.plot(step_sizes, mean_costs, 'o-')
        ax3.set_xlabel('Max Step Size')
        ax3.set_ylabel('Path Cost')
        ax3.set_title('Cost vs Max Step Size')
        ax3.grid(True)
        
        ax4.plot(step_sizes, success_rates, 'o-')
        ax4.set_xlabel('Max Step Size')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Success Rate vs Max Step Size')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results_pbias_{int(p_bias*100)}.png')
        plt.close('all')

if __name__ == "__main__":
    #run_dot_2d_astar()
    # run_dot_2d_rrt()
    # run_dot_2d_rrt_star()
    #run_2d_rrt_motion_planning()
    # analyze_rrt_performance()
    #run_2d_rrt_inspection_planning()
    # run_3d_experiment(0.75,0.2)
    run_3d()
    #results = run_rrt_experiments()
    #run_rrt_star_experiments()
    #run_inspection_comparison()
