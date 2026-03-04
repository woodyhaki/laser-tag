import torch
import torch.nn as nn
import infrastructure.pytorch_util as ptu

def evaluate(policy,states,observations,gt_actions):
    criterion = nn.MSELoss()
    observations = observations.to(ptu.device)
    gt_actions = gt_actions.to(ptu.device)
    states = states.to(ptu.device)
    states_expanded = states.unsqueeze(2)
    observations[:,:,:,-1,-3:] = states_expanded
    #sampled_actions = policy.forward(observations)
    #sampled_actions = policy.forward_pro(states,observations)
    sampled_actions = policy.forward(observations)
    
    #pdb.set_trace()
    # Compute loss between sampled actions and target actions
    loss = criterion(sampled_actions, gt_actions[:,-1,:])  ##

    return {
        'Test Loss': ptu.to_numpy(loss),
    }
    
def log_mean_loss(current_epoch,logger,key,mean_loss):
    logger.log_scalar(mean_loss, key, current_epoch)
    logger.flush()

def log_loss(current_epoch,logger,logs):
    for key, value in logs[-1].items():
        print('{} : {}'.format(key, value))
        logger.log_scalar(value, key, current_epoch)
    logger.flush()
    
def log_params(logger,params):
    logger.log_text(params)