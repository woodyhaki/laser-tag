import numpy as np
import pdb
# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 0.1
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 1
        # # physical damping
        # self.damping = 0.25
        # # contact response parameters
        # self.contact_force = 1e+2
        # self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        # update agent kinematic state (custom)
        for agent in self.agents:
            self.update_agent_state_kinematic(agent)

    def update_agent_state_kinematic(self, agent):
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise  

        # Kinematic model update:
        vx_local, vy_local, omega = agent.action.u  # 3D: vx, vy, omega in local frame
        theta = agent.state.p_ang  # heading in global frame

        #print("update_agent_state_kinematic!!!")
        # Rotate local velocity to global frame
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        v_global = rot @ np.array([vx_local, vy_local])
        #pdb.set_trace()
        # Apply update
        agent.state.p_pos += v_global * self.dt
        agent.state.p_ang += omega * self.dt

        # Keep angle within [-pi, pi]
        agent.state.p_ang = (agent.state.p_ang + np.pi) % (2 * np.pi) - np.pi

            # # gather agent action forces
    # def apply_action_force(self, p_force):
    #     # set applied forces
    #     for i,agent in enumerate(self.agents):
    #         if agent.movable:
    #             noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
    #             p_force[i] = agent.action.u + noise
    #     return p_force


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