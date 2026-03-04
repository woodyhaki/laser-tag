from collections import OrderedDict
import gymnasium
import cv2
import numpy as np
import time
import infrastructure.pytorch_util as ptu
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize
import pdb

def sample_n_trajectory_v1(env, policy, ntraj, max_path_length, render=False,show_window=False):
    paths = []
    for i in range(ntraj):
        path = sample_trajectory_v1(env, policy, max_path_length, render,show_window)
        paths.append(path)
    return paths

def sample_trajectory_v1(env,policy_model,max_path_length, render=False,show_window=False):
    if isinstance(env,gymnasium.Env):
        ob,_ =  env.reset()
    elif isinstance(env,DummyVecEnv):
        print("environment is DummyVecEnv!")
        ob =  env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    total_reward = 0
    while True:
        ac, _ = policy_model.predict(ob,deterministic=True)
        if isinstance(env, DummyVecEnv):
            next_ob, rew, done, info = env.step(ac)
        elif isinstance(env,gymnasium.Env):
            next_ob, rew, terminated, truncated, info = env.step(ac)

        total_reward += rew
        # rollout can end due to done, or due to max_path_length
        steps += 1
        if isinstance(env, DummyVecEnv):
            rollout_done = steps >= max_path_length or all(done)
        elif isinstance(env, gymnasium.Env):
            rollout_done = steps >= max_path_length or terminated or truncated
        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)
        ob = next_ob # jump to next timestep

        # render image of the simulated env
        if render:
            if isinstance(env, DummyVecEnv):
                img = env.render(show_window = show_window)
            elif isinstance(env,gymnasium.Env):
                img = env.render(show_window = show_window)
            img = np.transpose(img,(1,2,0))
            image_obs.append(np.transpose(cv2.resize(img,(400, 400)),(2,0,1)))
        # end the rollout if the rollout ended
        if rollout_done:
            break
    ep_reward = np.sum(rewards)    
    return {"observation" : obs,
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": next_obs,
            "terminal": np.array(terminals, dtype=np.float32)}

def sample_n_sb3_trajectory(env, policy, ntraj, max_path_length, render=False,show_window=False):
    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_sb3_trajectory(env, policy, max_path_length, render,show_window)
        paths.append(path)
    return paths

def sample_sb3_trajectory(env,policy_model,max_path_length, render=False,show_window=False):
    if isinstance(env,gymnasium.Env):
        ob,_ =  env.reset()
    elif isinstance(env,VecNormalize):
        print("environment is VecNormalize!")
        env = env.venv
        ob =  env.reset()
    elif isinstance(env,DummyVecEnv):
        print("environment is DummyVecEnv!")
        ob =  env.reset()

    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    total_reward = 0
    while True:
        ac, _ = policy_model.predict(ob)
        if isinstance(env, DummyVecEnv):
            next_ob, rew, done, info = env.step(ac)
        elif isinstance(env,gymnasium.Env):
            next_ob, rew, terminated, truncated, info = env.step(ac)
        total_reward += rew
        # rollout can end due to done, or due to max_path_length
        steps += 1
        if isinstance(env, DummyVecEnv):
            rollout_done = steps >= max_path_length or all(done)
        elif isinstance(env,gymnasium.Env):
            rollout_done = steps >= max_path_length or terminated or truncated
        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)
        ## print("next_ob:",next_ob['robot_state'],ob['robot_state'])
        
        ob = next_ob # jump to next timestep
        # render image of the simulated env
        if render:
            if isinstance(env, DummyVecEnv):
                img = env.render(show_window = show_window)
            elif isinstance(env,gymnasium.Env):
                img = env.render()
            img = np.transpose(img,(1,2,0))
            #print(img.shape)
            image_obs.append(cv2.resize(img,(400, 400)))
            
        # end the rollout if the rollout ended
        if rollout_done:
            break
    ep_reward = np.sum(rewards)
    return {"observation" : obs,
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": next_obs,
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectory(env, policy, adjacency_matrix,max_path_length, render,random_reset=True):
    """Sample a rollout in the environment from a policy."""
    
    # initialize env for the beginning of a new rollout
    if random_reset:
        ob,_ = env.random_reset()
    else:
        ob,_ =  env.reset()
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            img = env.render(show_window = False)
            img = np.transpose(img,(1,2,0))
            image_obs.append(cv2.resize(img,(500, 500)))
        
        ob = torch.FloatTensor(ob).unsqueeze(0).to(ptu.device)
        ac = policy(ob, adjacency_matrix)
        
        if isinstance(ac, torch.Tensor):
            if len(ac.shape) == 3:
                ac = ac.squeeze(0)
            ac = ac.detach().cpu().numpy()
        next_ob, rew, terminated, truncated, _ = env.step(ac)

        steps += 1
        rollout_done = steps >= max_path_length  #TODO # HINT: this is either 0 or 1
        
        # record result of taking that action
        obs.append(ob.detach().cpu().numpy())
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {"observation" : obs,
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": next_obs,
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy,adjacency_matrix, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy,adjacency_matrix, max_path_length, render,random_reset = True)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, adjacency_matrix,ntraj, max_path_length, render):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy,adjacency_matrix, max_path_length, render)
        paths.append(path)
    return paths

########################################
########################################
def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs

def compute_paths_metrics(paths):
    expert_returns = [path["reward"].sum() for path in paths]
    ep_lens = [len(path["reward"]) for path in paths]
    logs = OrderedDict()
    logs["Path_AverageReturn"] = np.mean(expert_returns)
    logs["Path_StdReturn"] = np.std(expert_returns)
    logs["Path_MaxReturn"] = np.max(expert_returns)
    logs["Path_MinReturn"] = np.min(expert_returns)
    logs["Path_AverageEpLen"] = np.mean(ep_lens)
    return logs
    
    
############################################
############################################
def get_pathlength(path):
    return len(path["reward"])

def render_expert_data(expert_data):
    fig, ax = plt.subplots()
    ax.clear()
    print(expert_data.shape)
    A_history = []
    B_history = []
    time_cnt = 0
    for obs in expert_data:
        ax.clear()
        print("time_cnt:", time_cnt)
        # obs = obs[0]
        A_pos = obs[0][0:2]  # [x, y]
        B_pos = obs[1][0:2]  # [x, y]
        A_theta = obs[0][2]  # heading angle
        B_theta = obs[1][2]  # heading angle

        # Draw current robot positions
        ax.plot(A_pos[0], A_pos[1], 'ro')  # A (Target)
        ax.plot(B_pos[0], B_pos[1], 'bo')  # B (Robot)

        # Connect current positions of A and B
        ax.plot([A_pos[0], B_pos[0]], [A_pos[1], B_pos[1]], 'k--')

        # Draw heading direction for A
        heading_length = 0.1
        a_heading_x = A_pos[0] + heading_length * np.cos(A_theta)
        a_heading_y = A_pos[1] + heading_length * np.sin(A_theta)
        ax.arrow(A_pos[0], A_pos[1], a_heading_x - A_pos[0], a_heading_y - A_pos[1],
                 head_width=0.05, head_length=0.05, fc='r', ec='r')

        # Draw heading direction for B
        b_heading_x = B_pos[0] + heading_length * np.cos(B_theta)
        b_heading_y = B_pos[1] + heading_length * np.sin(B_theta)
        ax.arrow(B_pos[0], B_pos[1], b_heading_x - B_pos[0], b_heading_y - B_pos[1],
                 head_width=0.05, head_length=0.05, fc='b', ec='b')

        # Update trajectory history
        A_history.append(A_pos)
        B_history.append(B_pos)

        # Draw trajectory history
        ax.plot(*zip(*A_history), 'r-', label="A (Target) History" if not A_history else "")
        ax.plot(*zip(*B_history), 'b-', label="B (Robot) History" if not B_history else "")

        # Show current frame and pause briefly to simulate animation
        time_cnt += 1
        plt.draw()
        plt.pause(0.1)
    
    # Draw complete trajectory history and add legend
    ax.plot(*zip(*A_history), 'r-', label="A (Target) History")
    ax.plot(*zip(*B_history), 'b-', label="B (Robot) History")
    ax.legend()
    plt.show()
    
def render_to_array(fig):
    # Get the current figure canvas
    canvas = fig.canvas

    # Force rendering (ensure figure is updated)
    canvas.draw()

    # Retrieve rendered image from canvas
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)

    # Get figure width and height
    width, height = canvas.get_width_height()

    # Convert to (H, W, C) format
    img_array = img_array.reshape((height, width, 3))  # default RGB

    # Convert to (C, H, W) format
    img_array = img_array.transpose(2, 0, 1)

    return img_array

def check_collision(obstacle1, obstacle2,min_dist=1.0):
    """
    Check if two obstacles (represented as circles) are colliding.
    Each obstacle is represented as (x, y, diameter).
    Returns True if there is a collision, False otherwise.
    """
    x1, y1, d1 = obstacle1
    x2, y2, d2 = obstacle2
    distance = np.linalg.norm(np.array([x1 - x2, y1 - y2]))  # Euclidean distance between centers
    return distance < (d1 / 2 + d2 / 2 + min_dist)  # If the distance is less than the sum of their radii

def check_target_collision(obstacle, target_pos, min_dist=1.0):
    """
    Check if the target position is too close to the obstacle.
    obstacle: (x, y, diameter) of the obstacle.
    target_pos: (x, y) of the target position.
    min_dist: Minimum allowed distance between the target and the obstacle.
    Returns True if the distance between the target and obstacle is less than min_dist, False otherwise.
    """
    x, y, diameter = obstacle
    distance_to_target = np.linalg.norm(np.array([x - target_pos[0], y - target_pos[1]]))
    return distance_to_target < (diameter / 2 + min_dist)  # Check if the distance is less than allowed minimum distance


def generate_obstacles(N_obs, min_diameter, max_diameter, x_range, y_range, target_pos, seed):
    """
    Generate obstacles with fixed random seed, ensuring no overlap and target position safety.
    """
    rng = np.random.default_rng(seed)
    obstacles = []
    
    for _ in range(N_obs):
        while True:
            x = rng.uniform(x_range[0], x_range[1])
            y = rng.uniform(y_range[0], y_range[1])
            diameter = rng.uniform(min_diameter, max_diameter)
            new_obstacle = np.array([x, y, diameter])

            collision = False
            for existing_obstacle in obstacles:
                if check_collision(new_obstacle, existing_obstacle):
                    collision = True
                    break

            for t in target_pos:
                if check_target_collision(new_obstacle, t):
                    collision = True
                    break

            if not collision:
                obstacles.append(new_obstacle)
                break

    return np.array(obstacles)
def generate_initial_pos(num_agent, 
                         x_range, 
                         y_range, 
                         target_pos,
                         dist_to_target_initial, 
                         obstacles,
                         seed=42):
    initial_positions = np.zeros((num_agent, 3))
    
    obstacle_radius = obstacles[:, 2]
    
    # Initialize counter
    attempts = 0
    max_attempts = 1000  # Maximum attempts to avoid infinite loops
    
    for i in range(num_agent):
        while attempts < max_attempts:
            # Randomly generate the robot's initial position
            x_init = np.random.uniform(x_range[0], x_range[1])
            y_init = np.random.uniform(y_range[0], y_range[1])
            
            # Compute distance between robot and target position
            distance_to_target = np.linalg.norm(np.array([x_init, y_init]) - target_pos[i])
            
            # Check if the distance to target is greater than the required minimum distance
            if distance_to_target < dist_to_target_initial:
                attempts += 1
                continue

            collision_free = True
            for id, obs in enumerate(obstacles):
                obs_x, obs_y, _ = obs
                obstacle_distance = np.linalg.norm(np.array([x_init, y_init]) - np.array([obs_x, obs_y])) - obstacle_radius[id]
                if obstacle_distance < 0.5:
                    collision_free = False
                    break
            
            # If no collision with obstacles and distance requirement is satisfied, generate heading
            if collision_free:
                # Randomly generate robot heading in range [0, 2 * pi]
                heading = np.random.uniform(-np.pi, np.pi)
                initial_positions[i] = [x_init, y_init, heading]
                break
            else:
                attempts += 1
        
        # If maximum attempts exceeded, exit
        if attempts >= max_attempts:
            raise ValueError(f"Could not find a valid position in {max_attempts} attempts. Consider adjusting range or constraints.")
    
    return initial_positions

def rotate_point(x, y, angle):
    """
    Rotate point (x, y) around the origin by a specified angle (in radians).
    """
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return new_x, new_y

def generate_random_initial_robot_states(obstacle_state, boundary, num_agents, min_distance=0.5):
    while True:
        # Randomly generate initial_area (initial position)
        initial_area = np.array([np.random.uniform(boundary[0], boundary[1]), 
                                 np.random.uniform(boundary[2], boundary[3])])

        # Compute central angle for regular N-gon (angle between vertices)
        angle_step = 2 * np.pi / num_agents
        
        # Array to store initial positions
        initial_states = []

        # Flag to check collisions
        collision_found = False
        
        # Compute each robot’s initial position
        for i in range(num_agents):
            # Compute angle for each vertex
            angle = i * angle_step
            
            # Compute position offset relative to initial_area
            x_offset = min_distance / 2 * np.cos(angle)
            y_offset = min_distance / 2 * np.sin(angle)
            
            # Add offset to target position
            agent_position = initial_area + np.array([x_offset, y_offset])
            
            # Check for collision with obstacles
            for obstacle in obstacle_state:
                ox, oy, diameter = obstacle  # obstacle center and diameter
                distance_to_obstacle = np.linalg.norm(np.array([ox, oy]) - agent_position)
                
                # If closer than obstacle radius + safety distance → collision
                if distance_to_obstacle <= diameter / 2 + min_distance:
                    collision_found = True
                    break
            
            if collision_found:
                break  # stop if collision detected

            # If no collision, save robot state with random heading
            heading = np.random.uniform(-np.pi, np.pi)
            agent_state = np.append(agent_position, heading)
            initial_states.append(agent_state)

        # Return array if all positions valid and collision-free
        if not collision_found and len(initial_states) == num_agents:
            return np.array(initial_states)


def generate_initial_positions(num_agents, initial_area, obstacle_state, collision_thres=0.4):
    # Compute central angle for regular N-gon (angle between vertices)
    angle_step = 2 * np.pi / num_agents
    
    # Array to store initial positions
    initial_positions = np.zeros((num_agents, 2))
    
    # Compute each robot’s initial position
    for i in range(num_agents):
        angle = i * angle_step
        x_offset = collision_thres * np.cos(angle)
        y_offset = collision_thres * np.sin(angle)
        agent_position = initial_area + np.array([x_offset, y_offset])
        
        # Check collision with obstacles
        for obstacle in obstacle_state:
            ox, oy, diameter = obstacle  # obstacle center and diameter
            if np.linalg.norm(np.array([ox, oy]) - agent_position) - diameter / 2 < collision_thres:
                raise ValueError(f"Agent at position {agent_position} collides with obstacle at ({ox}, {oy}).")
        
        initial_positions[i] = agent_position
    
    return initial_positions


def generate_target_positions(num_agents, target_area, s):
    # Compute central angle for regular N-gon (angle between vertices)
    angle_step = 2 * np.pi / num_agents
    R = s / 2 / np.sin(np.pi / num_agents)
    
    # Array to store target positions
    target_positions = np.zeros((num_agents, 2))
    
    # Compute each robot’s target position
    for i in range(num_agents):
        angle = i * angle_step
        x_offset = R * np.cos(angle)
        y_offset = R * np.sin(angle)
        target_positions[i] = target_area + np.array([x_offset, y_offset])
    
    return target_positions

def visualize_positions(target_positions, target_area, collision_thres, num_agents):
    # Plot target positions
    plt.figure(figsize=(6, 6))
    plt.scatter(target_positions[:, 0], target_positions[:, 1], color='red', label='Target Positions')
    
    # Plot target area (center point)
    plt.scatter(target_area[0], target_area[1], color='blue', label='Target Area', s=100, marker='x')
    
    # Draw polygon connecting target positions
    target_positions_closed = np.vstack([target_positions, target_positions[0]])
    plt.plot(target_positions_closed[:, 0], target_positions_closed[:, 1], 'k-', label=f'Polygon (N={num_agents})')
    
    # Set axis limits
    plt.xlim(target_area[0] - collision_thres - 0.1, target_area[0] + collision_thres + 0.1)
    plt.ylim(target_area[1] - collision_thres - 0.1, target_area[1] + collision_thres + 0.1)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add title and labels
    plt.title(f"Target Positions for {num_agents} Agents")
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Show legend
    plt.legend()
    plt.show()
    
class ObstacleVisualizer:
    def __init__(self, obstacle_state, ax=None):
        self.obstacle_state = obstacle_state
        self.ax = ax if ax is not None else plt.gca()  # Use provided axis, or create a new one
    
    def rotate_point(self, x, y, angle):
        """Rotate point (x, y) around origin by a specified angle (in radians)."""
        new_x = x * np.cos(angle) - y * np.sin(angle)
        new_y = x * np.sin(angle) + y * np.cos(angle)
        return new_x, new_y
    
    def draw_obstacles(self):
        for obstacle in self.obstacle_state:
            x, y, _ = obstacle  # obstacle center, diameter, heading (not used)
            length = 0.6
            half_length = length / 2
            # Four corners of square relative to center (x, y)
            corners = np.array([
                [-half_length, -half_length],  # bottom left
                [ half_length, -half_length],  # bottom right
                [ half_length,  half_length],  # top right
                [-half_length,  half_length]   # top left
            ])
            
            # Rotate corners (currently with angle=0 → no rotation)
            rotated_corners = np.array([
                self.rotate_point(corner[0], corner[1], 0) for corner in corners
            ])
            
            # Translate corners to obstacle center (x, y)
            rotated_corners[:, 0] += x
            rotated_corners[:, 1] += y
            
            # Draw rotated square
            obstacle_polygon = plt.Polygon(rotated_corners, color='g', alpha=0.5)
            self.ax.add_patch(obstacle_polygon)

def angle_difference(theta1, theta2):
    """
    Compute the shortest angular difference between two angles.

    This function returns the smallest signed difference between two angles,
    ensuring the result is always in the range [-π, π]. It correctly accounts 
    for the periodic nature of angles, avoiding issues where the naive 
    subtraction (theta1 - theta2) could lead to incorrect large differences.

    Args:
        theta1 (float): First angle in radians.
        theta2 (float): Second angle in radians.

    Returns:
        float: The shortest signed angular difference in radians, within [-π, π].
    """
    return np.arctan2(np.sin(theta1 - theta2), np.cos(theta1 - theta2))

def control_effort(action, omega_weight=2.0):
    """
    Compute a weighted control effort to penalize excessive action magnitudes.

    Args:
        action (np.array): A 3D action vector [vx, vy, omega].
        omega_weight (float): Weighting factor for angular velocity (default=2.0).

    Returns:
        float: The weighted control effort.
    """
    weighted_action = np.array([action[0], action[1], omega_weight * action[2]])
    return np.linalg.norm(weighted_action)

def generate_obstacles(n_obstacles=7, xy_range=(-2, 2), r=0.3, min_dist=0.1):
    """
    Generate non-overlapping circular obstacles
    Args:
        n_obstacles: number of obstacles
        xy_range: range of x, y coordinates (min, max)
        r: obstacle radius (can also be randomized)
        min_dist: additional safety margin between obstacles
    Returns:
        np.ndarray: [N, 3] array, each row is (x, y, r)
    """
    obstacles = []
    while len(obstacles) < n_obstacles:
        x = np.random.uniform(xy_range[0], xy_range[1])
        y = np.random.uniform(xy_range[0], xy_range[1])
        candidate = np.array([x, y, r], dtype=np.float32)

        # Check if the new obstacle collides with existing ones
        collision = False
        for obs in obstacles:
            dist = np.linalg.norm(candidate[:2] - obs[:2])
            if dist < (candidate[2] + obs[2] + min_dist):
                collision = True
                break

        if not collision:
            obstacles.append(candidate)

    return np.array(obstacles, dtype=np.float32)
