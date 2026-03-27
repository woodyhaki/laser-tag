import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_layers import MultiHeadAttention
import pdb


class MultiHeadAttentionEncoder(nn.Module):
    def __init__(   self,
                    num_neaerest_other_robots,
                    num_neaerest_obstacles,
                    feature_ex_hidden:int = 128,
                    features_dim:int = 128, ## encoder output size
                    num_heads:int = 1):
        super(MultiHeadAttentionEncoder, self).__init__()
        self.target_obs_dim = 4
        self.neighbor_hidden_size = feature_ex_hidden
        self.num_neaerest_robot = num_neaerest_other_robots  ## number of neighboring robot
        self.num_neaerest_obstacles = num_neaerest_obstacles
        
        # Embedding Layer
        fc_encoder_layer = feature_ex_hidden
        self.neighbor_embed_layer = nn.Sequential(nn.Linear(4, fc_encoder_layer),
                                                  nn.ELU(),
                                                  nn.Linear(fc_encoder_layer, fc_encoder_layer),)
        self.self_embed_layer = nn.Sequential(nn.Linear(3, fc_encoder_layer),
                                                  nn.ELU(),
                                                  nn.Linear(fc_encoder_layer, fc_encoder_layer),)
        self.obstacle_embed_layer = nn.Sequential(
            nn.Linear(3, fc_encoder_layer),
            nn.ELU(),
            nn.Linear(fc_encoder_layer, fc_encoder_layer),
        )
        self.target_embed_layer = nn.Sequential(
            nn.Linear(self.target_obs_dim, feature_ex_hidden),
            nn.ELU(),
            nn.Linear(feature_ex_hidden, feature_ex_hidden),
        )
        
        # Attention Layer
        self.attention_layer = MultiHeadAttention(num_heads, feature_ex_hidden, feature_ex_hidden, feature_ex_hidden)

        # MLP Layer
        self.encoder_output_size = features_dim
        #self.final_input_size = (self.num_neaerest_robot + self.num_neaerest_obstacles + 1 + 1) * feature_ex_hidden
        self.final_input_size = 2 * feature_ex_hidden
        
        #self.final_input_size = feature_ex_hidden * 4
        
       # pdb.set_trace()
        self.feed_forward = nn.Sequential(nn.Linear(self.final_input_size, self.encoder_output_size))
        self.apply(self.init_weights)
        print("Encoder construct ok!!")

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    # def forward(self, obs):
    #     # return np.concatenate( [self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm], \
    #     #                  local_obstacles_ob,local_robot_ob]    )
    #     #return np.concatenate([self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm]])
    #     target_obs = obs[:,3:7]
    #     #robot_obs = obs_dict['other_robot_state']
    #     self_obs = obs[:,0:3]
    #     obstacle_obs = obs[:,7:7+3*(self.num_neaerest_obstacles)].reshape((-1,self.num_neaerest_obstacles,3))
        
    #     #pdb.set_trace()
        
    #     others_obs = obs[:,7+3*(self.num_neaerest_obstacles):].reshape((-1,self.num_neaerest_robot,3))
        
    #     #neighbor_embed = self.neighbor_embed_layer(others_obs)
    #     obstacle_embed = self.obstacle_embed_layer(obstacle_obs)
    #     target_embed = self.target_embed_layer(target_obs)  # [1, feature_ex_hidden]
    #     self_embed = self.self_embed_layer(self_obs)
        
    #     if len(target_embed.shape) == 1:
    #         target_embed = target_embed.unsqueeze(0)
    #     elif len(target_embed.shape) == 2:
    #         target_embed = target_embed.unsqueeze(1)
    #     #print(f"neighbor_embed {target_embed.shape} obstacle_embed {obstacle_embed.shape}")

    #     if len(target_embed.shape) == 2:
    #         attn_embed = torch.cat((target_embed, obstacle_embed), dim=0).unsqueeze(0) 
    #     elif len(target_embed.shape) == 3:
    #         attn_embed = torch.cat((target_embed, obstacle_embed), dim=1)
    #     #print("input attention shape:",attn_embed.shape)
    #     #pdb.set_trace()
    #     attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)

    #     weights = torch.mean(attn_embed, dim=-1, keepdim=True)   # [batch, N, 1] (e.g., use mean to get scores)
    #     weights = torch.softmax(weights, dim=1)                  # [batch, N, 1]
    #     pooled = torch.sum(attn_embed * weights, dim=1)          # [batch, dim]


    #     if len(target_embed.shape) == 3:
    #         #pdb.set_trace()
    #         final_input = torch.cat((self_embed, pooled.squeeze(1)), dim=-1)
    #     elif len(target_embed.shape) == 2:
    #         final_input = torch.cat((self_embed.unsqueeze(0), pooled), dim=-1)

    #     out = self.feed_forward(final_input)
    #     return out

    def forward_mlp(self, obs):
        # return np.concatenate( [self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm], \
        #                  local_obstacles_ob,local_robot_ob]    )
        #return np.concatenate([self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm]])
        target_obs = obs[:,3:7]
        #robot_obs = obs_dict['other_robot_state']
        self_obs = obs[:,0:3]
        obstacle_obs = obs[:,7:7+3*(self.num_neaerest_obstacles)].reshape((-1,self.num_neaerest_obstacles,3))
        others_obs = obs[:,7+3*(self.num_neaerest_obstacles):].reshape((-1,self.num_neaerest_robot,4))
        
        neighbor_embed = self.neighbor_embed_layer(others_obs)
        obstacle_embed = self.obstacle_embed_layer(obstacle_obs)
        target_embed = self.target_embed_layer(target_obs)  # [1, feature_ex_hidden]
        self_embed = self.self_embed_layer(self_obs)
        
        if len(target_embed.shape) == 1:
            target_embed = target_embed.unsqueeze(0)
        elif len(target_embed.shape) == 2:
            target_embed = target_embed.unsqueeze(1)
        #print(f"neighbor_embed {target_embed.shape} obstacle_embed {obstacle_embed.shape}")
        pdb.set_trace()
        
        if len(target_embed.shape) == 2:
            attn_embed = torch.cat((self_embed.unsqueeze(0), target_embed, obstacle_embed, neighbor_embed), dim=0).unsqueeze(0) 
        elif len(target_embed.shape) == 3:
            #pdb.set_trace()
            attn_embed = torch.cat((self_embed.unsqueeze(1), target_embed, obstacle_embed, neighbor_embed), dim=1)

        # if len(target_embed.shape) == 3:
        #     #pdb.set_trace()
        #     final_input = torch.cat((self_embed, pooled.squeeze(1)), dim=-1)
        # elif len(target_embed.shape) == 2:
        #     final_input = torch.cat((self_embed.unsqueeze(0), pooled), dim=-1)
        #pdb.set_trace()
        batch_sz = obs.shape[0]
        final_input = attn_embed.reshape((batch_sz,-1))
        out = self.feed_forward(final_input)
        return out

    def forward(self, obs):
        # return np.concatenate( [self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm], \
        #                  local_obstacles_ob,local_robot_ob]    )
        #return np.concatenate([self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm]])
        target_obs = obs[:,3:7]
        #robot_obs = obs_dict['other_robot_state']
        self_obs = obs[:,0:3]
        obstacle_obs = obs[:,7:7+3*(self.num_neaerest_obstacles)].reshape((-1,self.num_neaerest_obstacles,3))
        others_obs = obs[:,7+3*(self.num_neaerest_obstacles):].reshape((-1,self.num_neaerest_robot,4))
        
        neighbor_embed = self.neighbor_embed_layer(others_obs)
        obstacle_embed = self.obstacle_embed_layer(obstacle_obs)
        target_embed = self.target_embed_layer(target_obs)  # [1, feature_ex_hidden]
        self_embed = self.self_embed_layer(self_obs)
        
        if len(target_embed.shape) == 1:
            target_embed = target_embed.unsqueeze(0)
        elif len(target_embed.shape) == 2:
            target_embed = target_embed.unsqueeze(1)
        #print(f"neighbor_embed {target_embed.shape} obstacle_embed {obstacle_embed.shape}")
        #pdb.set_trace()
        
        if len(target_embed.shape) == 2:
            attn_embed = torch.cat((target_embed, obstacle_embed, neighbor_embed), dim=0).unsqueeze(0) 
        elif len(target_embed.shape) == 3:
            #pdb.set_trace()
            attn_embed = torch.cat((target_embed, obstacle_embed, neighbor_embed), dim=1)

        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)
        weights = torch.mean(attn_embed, dim=-1, keepdim=True)   # [batch, N, 1] (e.g., use mean to get scores)
        weights = torch.softmax(weights, dim=1)                  # [batch, N, 1]
        pooled = torch.sum(attn_embed * weights, dim=1)          # [batch, dim]

        #pdb.set_trace()
        if len(target_embed.shape) == 3:
            #pdb.set_trace()
            final_input = torch.cat((self_embed, pooled.squeeze(1)), dim=-1)
        elif len(target_embed.shape) == 2:
            final_input = torch.cat((self_embed.unsqueeze(0), pooled), dim=-1)
        #pdb.set_trace()
        out = self.feed_forward(final_input)
        return out
    
    
    def get_out_size(self):
        return self.encoder_output_size


class OnboardMultiHeadAttentionEncoder(nn.Module):
    def __init__(   self,
                    num_neaerest_other_robots,
                    num_neaerest_obstacles,
                    feature_ex_hidden:int = 64,
                    features_dim:int = 128, ## encoder output size
                    num_heads:int = 1):
        super(MultiHeadAttentionEncoder, self).__init__()
        self.target_obs_dim = 4
        self.neighbor_hidden_size = feature_ex_hidden
        self.num_neaerest_robot = num_neaerest_other_robots  ## number of neighboring robot
        self.num_neaerest_obstacles = num_neaerest_obstacles
        
        # Embedding Layer
        fc_encoder_layer = feature_ex_hidden
        self.neighbor_embed_layer = nn.Sequential(nn.Linear(3, fc_encoder_layer),
                                                  nn.ELU(),
                                                  nn.Linear(fc_encoder_layer, fc_encoder_layer),)
        self.self_embed_layer = nn.Sequential(nn.Linear(3, fc_encoder_layer),
                                                  nn.ELU(),
                                                  nn.Linear(fc_encoder_layer, fc_encoder_layer),)
        self.obstacle_embed_layer = nn.Sequential(
            nn.Linear(3, fc_encoder_layer),
            nn.ELU(),
            nn.Linear(fc_encoder_layer, fc_encoder_layer),
        )
        self.target_embed_layer = nn.Sequential(
            nn.Linear(self.target_obs_dim, feature_ex_hidden),
            nn.ELU(),
            nn.Linear(feature_ex_hidden, feature_ex_hidden),
        )
        
        # Attention Layer
        self.attention_layer = MultiHeadAttention(num_heads, feature_ex_hidden, feature_ex_hidden, feature_ex_hidden)

        # MLP Layer
        self.encoder_output_size = features_dim
        #self.final_input_size = (self.num_neaerest_robot + self.num_neaerest_obstacles + 1) * feature_ex_hidden
        self.final_input_size = feature_ex_hidden * 2
        
       # pdb.set_trace()
        self.feed_forward = nn.Sequential(nn.Linear(self.final_input_size, self.encoder_output_size))
        self.apply(self.init_weights)
        print("Encoder construct ok!!")

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    def forward(self, obs):
        # return np.concatenate( [self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm], \
        #                  local_obstacles_ob,local_robot_ob]    )
                
        #return np.concatenate([self_pos, [self_theta], [delta_theta], local_enemy_relative_bearing, [enemy_norm]])
        target_obs = obs[:,3:7]
        #robot_obs = obs_dict['other_robot_state']
        self_obs = obs[:,0:3]
        obstacle_obs = obs[:,7:7+3*(self.num_neaerest_obstacles)].reshape((-1,self.num_neaerest_obstacles,3))
        
        #pdb.set_trace()
        
        others_obs = obs[:,7+3*(self.num_neaerest_obstacles):].reshape((-1,self.num_neaerest_robot,3))
        
        #neighbor_embed = self.neighbor_embed_layer(others_obs)
        obstacle_embed = self.obstacle_embed_layer(obstacle_obs)
        target_embed = self.target_embed_layer(target_obs)  # [1, feature_ex_hidden]
        self_embed = self.self_embed_layer(self_obs)
        
        if len(target_embed.shape) == 1:
            target_embed = target_embed.unsqueeze(0)
        elif len(target_embed.shape) == 2:
            target_embed = target_embed.unsqueeze(1)
        #print(f"neighbor_embed {target_embed.shape} obstacle_embed {obstacle_embed.shape}")

        if len(target_embed.shape) == 2:
            attn_embed = torch.cat((target_embed, obstacle_embed), dim=0).unsqueeze(0) 
        elif len(target_embed.shape) == 3:
            attn_embed = torch.cat((target_embed, obstacle_embed), dim=1)
        #print("input attention shape:",attn_embed.shape)
        #pdb.set_trace()
        attn_embed, attn_score = self.attention_layer(attn_embed, attn_embed, attn_embed)

        ## In this version of implementation, NO MLP is used, you can add MLP if needed.
        weights = torch.mean(attn_embed, dim=-1, keepdim=True)   # [batch, N, 1] (e.g., use mean to get scores)
        weights = torch.softmax(weights, dim=1)                  # [batch, N, 1]
        pooled = torch.sum(attn_embed * weights, dim=1)          # [batch, dim]


        if len(target_embed.shape) == 3:
            #pdb.set_trace()
            final_input = torch.cat((self_embed, pooled.squeeze(1)), dim=-1)
        elif len(target_embed.shape) == 2:
            final_input = torch.cat((self_embed.unsqueeze(0), pooled), dim=-1)

        out = self.feed_forward(final_input)
        return out
    
    def get_out_size(self):
        return self.encoder_output_size

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Approximate the largest singular value (spectral norm) using power iteration
def spectral_norm(W, n_iter=1):
    W_mat = W.view(W.size(0), -1)  # flatten into matrix [out_features, in_features]
    u = torch.randn(W_mat.size(0), 1, device=W.device)  # [out_features, 1]
    for _ in range(n_iter):
        v = F.normalize(W_mat.t() @ u, dim=0, eps=1e-12)
        u = F.normalize(W_mat @ v, dim=0, eps=1e-12)
    sigma = (u.T @ (W_mat @ v)).item()  # u^T W v
    return sigma

# Custom Linear layer + manual Spectral Normalization
class SNLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        W = self.linear.weight
        sigma = spectral_norm(W)
        W_sn = W / (0.75 * sigma + 1e-8)
        return F.linear(x, W_sn, self.linear.bias)

# MLP network (apply spectral normalization manually in each layer)
class SNMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = SNLinear(in_dim, hidden_dim)
        self.fc2 = SNLinear(hidden_dim, hidden_dim)
        self.fc3 = SNLinear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x