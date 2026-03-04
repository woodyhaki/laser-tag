import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infrastructure.logger import Logger
import path_utils
import rl_utils as rl_utils
import sys
sys.path.append("multiagent-particle-envs")
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from feature_encoders import MultiHeadAttentionEncoder,TwoLayerFC

def make_env(scenario_name):
    # Create environment from scenario file script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, 
                        scenario.reset_world, 
                        scenario.reward,
                        scenario.observation,
                        scenario.info,
                        scenario.done
                        )
    return env

def onehot_from_logits(logits, eps=0.01):
    ''' Generate one-hot representation of the optimal action '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # Generate random actions and convert to one-hot
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # Select action using epsilon-greedy strategy
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """ Sample from Gumbel(0,1) distribution """
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Sample from Gumbel-Softmax distribution """
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature=1.0):
    """ Sample from Gumbel-Softmax distribution and discretize """
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # Return one-hot discrete actions for interaction with the environment,
    # while keeping gradients from y for correct backpropagation
    return y

class DDPG:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device,
                 action_low, action_high,num_nearest_obstacle,num_nearest_robot):
        self.device = device
        self.feature_extractor = MultiHeadAttentionEncoder(num_neaerest_other_robots=num_nearest_robot,
                                                            num_neaerest_obstacles=num_nearest_obstacle).to(device)
        self.actor = TwoLayerFC(hidden_dim, action_dim, hidden_dim).to(device)
        
        # self.actor = SNMLP(state_dim, hidden_dim,action_dim).to(device)
        # self.target_actor = SNMLP(state_dim, hidden_dim,action_dim).to(device)
                
        self.target_actor = TwoLayerFC(hidden_dim, action_dim,
                                        hidden_dim).to(device)
        
        self.critic = TwoLayerFC(critic_input_dim, 1, 4*hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        4*hidden_dim).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + list(self.actor.parameters()),
            lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        # Action bounds
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

    def take_action(self, state, explore=False, noise_scale=0.1):
        self.actor.eval()
        self.feature_extractor.eval()
        #pdb.set_trace()
        with torch.no_grad():
            feature = self.feature_extractor(state)
            action = self.actor(feature)             # shape: [1, action_dim]
            action = torch.tanh(action)              # output in [-1, 1]
            action = (action + 1) / 2                # scale to [0, 1]
            action = self.action_low + (self.action_high - self.action_low) * action  # scale to [low, high]

            if explore:
                noise = noise_scale * (self.action_high - self.action_low) * torch.randn_like(action)
                action = action + noise
                action = torch.clamp(action, self.action_low, self.action_high)
        return action.cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

    def save_actor(self, save_dir: str, filename: str = "ddpg_actor.pth"):
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, filename)
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'actor': self.actor.state_dict(),
        }, save_dir)
        print(f"[DDPG] Actor and FeatureExtractor saved to: {save_dir}")

    def load_actor(self, load_path: str):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.actor.load_state_dict(checkpoint['actor'])
        print(f"[DDPG] Actor and FeatureExtractor loaded from: {load_path}")

def squash_action(action,action_low,action_high):
    action = torch.tanh(action)              # output in [-1, 1]
    action = (action + 1) / 2                # scale to [0, 1]
    action = action_low + (action_high - action_low) * action  # scale to [low, high]
    return action

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau,num_nearest_obstacle,num_nearest_robot):
        self.agents = []
        self.num_agents = len(env.agents)
        for i in range(self.num_agents):
            if i == 0:
                self.agents.append(
                    DDPG(state_dims[i], action_dims[i], critic_input_dim,
                        hidden_dim, actor_lr, critic_lr, device,action_low=-0.1,action_high=0.1,num_nearest_obstacle=num_nearest_obstacle,num_nearest_robot=num_nearest_robot))
            else:
                self.agents.append(
                    DDPG(state_dims[i], action_dims[i], critic_input_dim,
                        hidden_dim, actor_lr, critic_lr, device,action_low=-0.1,action_high=0.1,num_nearest_obstacle=num_nearest_obstacle,num_nearest_robot=num_nearest_robot))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    @property
    def feature_extractors(self):
        return [agt.feature_extractor for agt in self.agents]
    
    def save_all_agents(self, save_dir: str):
        for i,a in enumerate(self.agents):
            a.save_actor(save_dir=f"{save_dir}_{i}.pth")
        print(f"saved all agents' actors and feature extractors @ {save_dir}")

    def load_all_agents(self, load_dir: str, train_epi:int):
        for i, a in enumerate(self.agents):
            path = f"{load_dir}/actor_{train_epi}_{i}.pth"
            a.load_actor(path)
        print(f"Loaded all agents' actors from {load_dir}")

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.num_agents)
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]
        action_low_i_agent =  cur_agent.action_low
        action_high_i_agent =  cur_agent.action_high

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            squash_action(
                target_actor(feature_extractor(_next_obs)), action_low_i_agent, action_high_i_agent
            )
            for feature_extractor, target_actor, _next_obs in zip(
                self.feature_extractors, self.target_policies, next_obs
            )
        ]
        #pdb.set_trace()
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(
            -1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        
        feature_i_agent = cur_agent.feature_extractor(obs[i_agent])
        cur_actor_out = cur_agent.actor(feature_i_agent)
        
        
        cur_act_vf_in = squash_action(cur_actor_out,action_low_i_agent,action_high_i_agent)
        #gumbel_softmax(cur_actor_out)
        
       # pdb.set_trace()
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                #all_actor_acs.append(onehot_from_logits(pi(_obs)))
                feature = self.feature_extractors[i](_obs)
                all_actor_acs.append(squash_action(pi(feature),action_low_i_agent,action_high_i_agent))
                
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

def evaluate(env_id, maddpg, n_episode=1, episode_length=25,render=False):
    for _ in range(n_episode):
        env = make_env(env_id)
        returns = np.zeros(len(env.agents))
        obs = env.reset()
        render_im_list = []
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            #print(actions)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
            if render:
                render_im = env.render(show_window=False)
                render_im = np.transpose(render_im,(1,2,0))
               #pdb.set_trace()
                render_im_list.append(np.transpose(render_im,(1,2,0)))
        env.close()
    return returns.tolist(),render_im_list

def test_policy_from_file(env_id, maddpg, model_path, model_epi, n_episode=1, episode_length=25,render=False):
    maddpg.load_all_agents(model_path,model_epi)
    return test_policy(env_id, maddpg, n_episode, episode_length, render)

def test_policy(env_id, maddpg, n_episode=1, episode_length=25,render=False):
    for _ in range(n_episode):
        env = make_env(env_id)
        returns = np.zeros(len(env.agents))
        obs = env.reset()
        render_im_list = []
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            #print(actions)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
            if render:
                env.render(show_window=True,pause_time=0.01)
        env.close()
    return returns.tolist(),render_im_list

def test_env(env):
    obs, _ = env.reset()
    for _ in range(100):
        action_array = [
            np.array([0.1,0,0.1]),
            np.array([0.1,0,0.1])
        ]
        #print(action_dict)
        obs, reward, terminated_dict,trunc_dict = env.step(action_array)
        # if terminated_dict["__all__"] or trunc_dict["__all__"]:
        #     break
        #print(obs)

        #if env.render_mode == "human":
        env.render(show_window=True,pause_time=1)
    env.close()

if __name__ == '__main__':
    num_nearest_obstacle = 3
    num_nearest_robot = 1
    num_episodes = 500_000
    episode_length = 100
    buffer_size = 100000
    hidden_dim = 128
    actor_lr = 5e-3
    critic_lr = 5e-3
    gamma = 0.95
    tau = 5e-2
    batch_size = 1600
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    update_interval = 100
    minimal_size = 10000

   # env_id = "my_adversary"
    env_id = "multi_adversary"
    env = make_env(env_id)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(3)
        
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])


    #------------------------Test Environment-----------------------
    # test_env(env)
    # exit()
    #---------------------------------------------------------------


    critic_input_dim = sum(state_dims) + sum(action_dims)
    maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,
                    action_dims, critic_input_dim, gamma, tau,num_nearest_obstacle,num_nearest_robot)

    ##------------------------Test Environment-----------------------
    # model_path = '[PATH]'

    # model_epi = 65999

    # test_policy_from_file(env_id,maddpg,model_path,model_epi, n_episode=1, episode_length=500,render=True)
    # exit()
    ##---------------------------------------------------------------

    summary_folder = path_utils.create_path_to_folder('./logs/')
    logger = Logger(summary_folder + '/maddpg')
    reward_weights = env.world.reward_weights
    logger.log_text("Weights:",reward_weights)

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_folder = path_utils.create_path_to_folder(model_dir)

    return_list = []
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()
        episode_rewards = np.zeros(len(env.agents))  # Initialize rewards for each agent
        episode_info_dict = {}
        for e_i in range(episode_length):
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, info = env.step(actions)
            #pdb.set_trace()

            for k, v in info['n'].items():
                if k not in episode_info_dict:
                    episode_info_dict[k] = []
                episode_info_dict[k].append(v)

            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            # Accumulate rewards for each agent
            episode_rewards += reward
        
            total_step += 1
            if replay_buffer.size(
            ) >= minimal_size and total_step % update_interval == 0:
                sample = replay_buffer.sample(batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x]
                                for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(device)
                        for aa in rearranged
                    ]
                #pdb.set_trace()

                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        print(f"i_episode {i_episode}")
        for k, v in episode_info_dict.items():
            if 'consecutive' in k or 'hit' in k:
                episode_info_dict[k] = v[-1]
            else:
                episode_info_dict[k] = np.mean(v, axis=0)
            if 'weighted' in k or 'hit' in k:
                #logger.log_scalar(episode_info_dict[k], f"agent_0/{k}", i_episode)
                logger.log_scalar(episode_info_dict[k], f"agent_1/{k}", i_episode)
                #logger.log_scalar(episode_info_dict[k], f"agent_2/{k}", i_episode)

        logger.log_scalar(np.mean(episode_rewards[0]), "agent_0/return", i_episode)
        logger.log_scalar(np.mean(episode_rewards[1]), "agent_1/return", i_episode)
        #logger.log_scalar(np.mean(episode_rewards[2]), "agent_2/return", i_episode)
        

        if (i_episode > 2000) and (i_episode + 1) % 1000 == 0:
            ep_returns,img_render = evaluate(env_id, maddpg, n_episode=1,episode_length=100, render=True)
            img_render_all = np.array(img_render, dtype=np.uint8)
            render_images = np.expand_dims(img_render_all,0)
            #pdb.set_trace()
            logger.log_video(render_images,"rollout/video",i_episode)
            logger.flush()
            return_list.append(ep_returns)
            print(f"Episode: {i_episode+1}, {ep_returns}")
            path_save = f"{model_folder}/actor_{i_episode}"
            maddpg.save_all_agents( path_save )
            
    return_array = np.array(return_list)
    
    for i, agent_name in enumerate(["adversary_0", "agent_1"]):
        plt.figure()
        pdb.set_trace()
        moving_avg_return = rl_utils.moving_average(return_array[:, i], 7)
        evaluate_epoch = np.arange(return_array.shape[0]) * 100
        print(evaluate_epoch.shape,moving_avg_return.shape)
        plt.plot(
            evaluate_epoch,
            moving_avg_return)
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by MADDPG")
        plt.show()