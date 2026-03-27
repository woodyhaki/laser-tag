from train_maddpg import make_env, test_policy_from_file,MADDPG
import rl_utils as rl_utils
import torch


if __name__ == '__main__':
    num_nearest_robot = 1
    num_nearest_obstacle = 3
    
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

    model_path = '[PATH]'
    
    model_epi = 28999

    test_policy_from_file(env_id,maddpg,model_path,model_epi, n_episode=1, episode_length=500,render=True)
    exit()