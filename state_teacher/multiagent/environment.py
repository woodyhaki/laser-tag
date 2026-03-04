import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pdb
import gymnasium
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gymnasium.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                # if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                #     act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                # else:
                act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-4, high=4, shape=(obs_dim,), dtype=np.float32))
            
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # for i, agent in enumerate(self.agents):
        #     print(f"{i} {agent}")

        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            
            info = self._get_info(agent)
            info_n['n'] = info
        #pdb.set_trace()
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        # if isinstance(action_space, MultiDiscrete):
        #     act = []
        #     size = action_space.high - action_space.low + 1
        #     index = 0
        #     for s in size:
        #         act.append(action[index:(index+s)])
        #         index += s
        #     action = act
        # else:
        action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 1.0

            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None


    def _init_render(self):
        self.fig, self.ax = plt.subplots(figsize=(18, 18))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlabel('X Axis', fontsize=20)
        self.ax.set_ylabel('Y Axis', fontsize=20)
        self.ax.tick_params(labelsize=20)
        self.ax.grid()

    def render(self, show_window=False, pause_time=0.1):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.agent_trajs = {}  # key: agent name, value: list of (x, y)

        self.ax.clear()
        self.ax.set_aspect('equal')

        # pdb.set_trace()
        self.ax.set_xlim(self.observation_space[0].low[0], self.observation_space[0].high[0])
        self.ax.set_ylim(self.observation_space[0].low[0], self.observation_space[0].high[0])
        self.ax.set_title(f"Step: {getattr(self, 'steps', 0)}", fontsize=14)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid()

        agents = self.world.agents
        if len(agents) < 2:
            return
        
        ref_agent = agents[0]  # The 0th agent is the reference enemy
        ref_pos = np.array(ref_agent.state.p_pos)
        ref_heading = ref_agent.state.p_ang
        
        for a in agents:
            name = a.name
            if name == "ally":
                label = 'Ally'
                color = 'blue'
            else:
                label = 'Enemy'
                color = 'red'

            px, py = a.state.p_pos
            p_heading = a.state.p_ang
            self.ax.add_patch(Circle((px, py), radius=0.15, color=color, label=label))
            arrow_len = 0.4
            self.ax.arrow(px, py, arrow_len * np.cos(p_heading), arrow_len * np.sin(p_heading),
                        head_width=0.05, color='black')

            # Initialize trajectory buffer
            if name not in self.agent_trajs:
                self.agent_trajs[name] = []
            self.agent_trajs[name].append((px, py))

            # Render trajectories
            for pos in self.agent_trajs[name][-30:]:
                self.ax.add_patch(Circle(pos, radius=0.02, color=color, alpha=0.5))
            
            # === Add distance and bearing annotation (relative to enemy) ===
            if a != ref_agent:
                agent_pos = np.array([px, py])
                delta = ref_pos - agent_pos
                distance = np.linalg.norm(delta)

                # Compute relative bearing angle (angle from agent to enemy)
                angle_to_enemy = np.arctan2(delta[1], delta[0])
                relative_bearing = angle_to_enemy - p_heading
                relative_bearing = np.rad2deg((relative_bearing + np.pi) % (2 * np.pi) - np.pi)  # wrap to [-180, 180]

                # Place annotation text above the midpoint between the two
                mid_x = (px + ref_pos[0]) / 2
                mid_y = (py + ref_pos[1]) / 2 + 0.3
                self.ax.text(mid_x, mid_y,
                            f'Dist: {distance:.2f} θ: {relative_bearing:.1f}°',
                            color='green', fontsize=20, ha='center')

            ## Render FOV
            if a.name == "ally":
                import matplotlib.patches as patches
                fov_angle = np.deg2rad(90)
                fov_range = 3.0
                fov_theta1 = np.rad2deg(p_heading - fov_angle / 2)
                fov_theta2 = np.rad2deg(p_heading + fov_angle / 2)

                fov_patch = patches.Wedge(center=(px, py),
                                        r=fov_range,
                                        theta1=fov_theta1,
                                        theta2=fov_theta2,
                                        facecolor='orange',
                                        alpha=0.3)
                self.ax.add_patch(fov_patch)
                
                ## Render Fire Cone
                fov_angle = np.deg2rad(20)
                fov_range = 3.0
                fov_theta1 = np.rad2deg(p_heading - fov_angle / 2)
                fov_theta2 = np.rad2deg(p_heading + fov_angle / 2)

                fov_patch = patches.Wedge(center=(px, py),
                                        r=fov_range,
                                        theta1=fov_theta1,
                                        theta2=fov_theta2,
                                        facecolor='red',
                                        alpha=0.3)
                self.ax.add_patch(fov_patch)

        ##-------------Draw obstacles as squares----------------------
        square_size = 0.6
        half_size = square_size / 2
        for ox, oy, radius in self.world.obstacle_states:
            square = plt.Rectangle((ox - half_size, oy - half_size),
                                square_size, square_size,
                                color='#CD853F', alpha=0.5)
            self.ax.add_patch(square)
        ##-----------------------------------------------------------

        self.ax.legend()
        if show_window:
            plt.draw()
            plt.pause(pause_time)

        fig = self.fig
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return img
    
    
    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
