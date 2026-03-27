import torch
import torch.nn as nn
import numpy as np
import infrastructure.pytorch_util as ptu
import pdb
import numpy as np
from policies.base_policy import *
from policies.MLP_policy import *
from PIL import Image
from vision_encoder import *
from simple_encoders import SimpleImageEncoder


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size, dilation=1):
        """
        input_size: dimension of input features
        hidden_size: number of channels in hidden convolutional layers
        output_size: dimension of output features
        num_layers: number of convolutional layers
        kernel_size: size of convolutional kernels
        dilation: dilation factor, controls the size of the receptive field
        """
        super(TemporalConvolutionalNetwork, self).__init__()
        
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Define convolutional layers for the TCN
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1) * dilation, dilation=dilation)
            )
            layers.append(nn.ReLU())
        
        # Stack convolutional layers
        self.conv_layers = nn.Sequential(*layers)
        
        # Define the final fully connected layer, output feature dimension = output_size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_size]
        # Convert to [batch_size, input_size, seq_len] for Conv1d
        x = x.transpose(1, 2)  # -> [batch_size, input_size, seq_len]
        
        # Extract features through convolutional layers
        x = self.conv_layers(x)
        
        # Take the output at the last time step (TCN is causal, so the last step contains the most relevant info)
        x = x[:, :, -1]
        x = self.fc(x)
        
        return x

class VisionImitationPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 params,
                 state_dim,
                 action_dim,
                 seq_len,
                 n_layers,
                 hiddent_size,
                 input_is_depth,
                 image_encoder_name,
                 image_encoder_path,
                 is_teacher = False,
                 **kwargs
                 ):
        super().__init__() 

        # init vars
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.hidden_size = hiddent_size
        self.seq_len = seq_len
        self.pro_state_dim = params['pro_state_dim']
        self.is_teacher = is_teacher
        
        #============ init image encoder ===================================

        if ('resnet' in image_encoder_name) or ('multimae' in image_encoder_name) or ('dino' in image_encoder_name) or \
            ('ViT' in image_encoder_name):
            self.image_encoder = ImageEncoder(params, input_is_depth, image_encoder_name, image_encoder_path)
        elif 'simple_encoder' in image_encoder_name:
            self.image_encoder = SimpleImageEncoder(depth=8, width=1)
            #self.image_encoder = SimpleCNNEncoder()
        else:
            raise Exception(f"Wrong type of encoder:{image_encoder_name}")

        if params['freeze_encoder'] and ('simple_encoder' not in image_encoder_name):
            for param in self.image_encoder.parameters():
                param.requires_grad = False


        ### Freeze last fewer layers
        # if params['freeze_encoder']:
        #     for param in self.image_encoder.parameters():
        #         param.requires_grad = False

        #     num_unfrozen_blocks = 3
        #     for block in self.image_encoder.blocks[-num_unfrozen_blocks:]:
        #         for param in block.parameters():
        #             param.requires_grad = True

        #     for param in self.image_encoder.norm.parameters():
        #         param.requires_grad = True

        #     if hasattr(self.image_encoder, 'head'):
        #         for param in self.image_encoder.head.parameters():
        #             param.requires_grad = True
        
        self.image_encoder.to(ptu.device)

        #===================================================================
        ##========== init state encoder ====================================
        self.pro_state_encoder = build_mlp(input_size = self.state_dim,
                                           output_size = self.pro_state_dim, 
                                           n_layers = self.n_layers, 
                                           size = self.hidden_size)
        
        # self.pro_state_encoder = nn.Sequential(
        #     nn.Linear(self.state_dim, self.hidden_size),
        #     nn.ELU(),
        #     nn.Linear(self.hidden_size, self.pro_state_dim),
        # )
        self.pro_state_encoder.to(ptu.device)
        
        ##========== init RNN ====================================
        self.rnn = nn.LSTM(input_size= 
                           self.pro_state_dim + params['image_encoder_out_dim'], \
                           hidden_size=128, num_layers=2, batch_first=True)
        
        # self.tcn_image = TemporalConvolutionalNetwork( input_size = self.image_embedding_dim * self.num_patches,
        #                                                hidden_size = 128,
        #                                                output_size = 128,
        #                                                num_layers = 2,
        #                                                kernel_size = 3)
        self.rnn.to(ptu.device)
        #self.tcn_image.to(ptu.device)
        ##========== init MLP ====================================
        self.fc = build_mlp( input_size = 128,
                             output_size = self.action_dim,
                             n_layers=self.n_layers, 
                             size=self.hidden_size  )
        self.fc.to(ptu.device)
        if 'simple_encoder' not in image_encoder_name:
            self.learning_rates = [
                {'params': self.image_encoder.image_encoder.parameters(), 'lr': 1e-05, \
                'name': 'image_encoder'},
                {'params': self.image_encoder.image_encoder_linear_projection.parameters(), 'lr': 1e-4, \
                'name': 'image_encoder_linear_projection' },
                {'params': self.rnn.parameters(), 'lr': 1e-4,\
                'name':  'rnn' },
                {'params': self.pro_state_encoder.parameters(), 'lr': 1e-4, \
                'name':  'pro_state_encoder'   },
                {'params': self.fc.parameters(), 'lr': 1e-4, \
                'name':  'fc'}
            ]
        else:
            self.learning_rates = [
                {'params': self.image_encoder.image_encoder.parameters(), 'lr': 1e-05, \
                'name': 'image_encoder'},
                {'params': self.rnn.parameters(), 'lr': 1e-4,\
                'name':  'rnn' },
                {'params': self.pro_state_encoder.parameters(), 'lr': 1e-4, \
                'name':  'pro_state_encoder'   },
                {'params': self.fc.parameters(), 'lr': 1e-4, \
                'name':  'fc'}
            ]


        self.optimizer = optim.AdamW(self.learning_rates)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)
        if params['freeze_action']:
            for param in self.rnn.parameters():
                param.requires_grad = False
            for param in self.pro_state_encoder.parameters():
                param.requires_grad = False
            for param in self.fc.parameters():
                param.requires_grad = False

    def save(self, filepath):
        """
        :param filepath: path to save expert policy
        """
        torch.save(self.state_dict(), filepath)

    def save_img_encoder(self, filepath):
        """
        :param filepath: path to save expert policy
        """
        torch.save(self.image_encoder.state_dict(), filepath)
        
        
    def load(self,filepath):
        state_dict = torch.load(filepath,weights_only=True)
        self.load_state_dict(state_dict,strict=False)

    def get_embedding(self, input ) -> Any:
        # encode patches of an image
        if len(input.shape) == 4:
            input = input.unsqueeze(1)
        image = input
        
        if image.shape[2] == 4: ## multimae, RGBD mode
            depth = image[:,:,3,:,:].unsqueeze(2)
            rgb_img = image[:,:,0:3,:,:]
            image_embedding = self.image_encoder.forward_multi_mae(depth,rgb_img)
        else:
            image_embedding = self.image_encoder(image)
        #pdb.set_trace()
        pro_states = input[:,:,-1,-1,-3:].squeeze(2)

        target_pos = torch.FloatTensor([-1.2,2.4]).cuda()
        temp = target_pos - pro_states[:,:,:2]
        heading = pro_states[:, :, 2].unsqueeze(-1)

        temp = torch.cat((temp,heading),dim=-1)
        state_embedding = self.pro_state_encoder(temp)
        #state_embedding = self.pro_state_encoder(pro_states)
        
        #pdb.set_trace()
        embedding = torch.cat((state_embedding,image_embedding),dim=-1)
        return embedding

    def forward(self, input ) -> Any:
        """
        Defines the forward pass of the expert policy
        image: 
        return:
            action: sampled action(s) from the policy
        """

        # encode patches of an image
        if len(input.shape) == 4:
            input = input.unsqueeze(1)
        image = input

        image_embedding = self.image_encoder(image)
        embedding = image_embedding
        #print("state_embedding:",state_embedding.shape)
        #pdb.set_trace()
        #print("patch_embeddings:",patch_embedding.shape)
       
        # x = self.tcn_image(patch_embedding.float())
        # actions = self.fc(x)
        x, (hn, cn) = self.rnn(embedding.float())
        
        actions = self.fc(x[:,-1,:])
        return actions
    
    def forward_pro(self, pro_states, image, dropout_prob = None) -> Any:
        """
        Defines the forward pass of the expert policy
        pro_states:global state, N x 3 mat, [x y theta]
        image: 
        return:
            action: sampled action(s) from the policy
        """
        # encode patches of an image
        if dropout_prob is not None:
            patch_embedding = self.image_encoder(image,dropout_prob)
        else:
            patch_embedding = self.image_encoder(image)
        
        #pdb.set_trace()
        target_pos = torch.FloatTensor([-1.2,2.4]).cuda()
        temp = target_pos - pro_states[:,:,:2]
        heading = pro_states[:, :, 2].unsqueeze(-1)

        temp = torch.cat((temp,heading),dim=-1)
        state_embedding = self.pro_state_encoder(temp)
        patch_embedding = self.patch_linear_projection(patch_embedding)
        
        #pdb.set_trace()
        embedding = torch.cat((state_embedding,patch_embedding),dim=-1)
        #print("state_embedding:",state_embedding.shape)
        #pdb.set_trace()
        #print("patch_embeddings:",patch_embedding.shape)
       
        # x = self.tcn_image(patch_embedding.float())
        # actions = self.fc(x)
        #pdb.set_trace()
        x, (hn, cn) = self.rnn(embedding.float())
        
        #pdb.set_trace()
        actions = self.fc(x[:,-1,:])
        return actions

    def update(self, states, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        criterion = nn.MSELoss()
        observations = observations.to(ptu.device)
        actions = actions.to(ptu.device)
        states = states.to(ptu.device)
        
        states_expanded = states.unsqueeze(2)

        if observations.shape[2] == 1:
            observations[:,:,:,-1,-3:] = states_expanded
        elif observations.shape[2] == (3 or 4):
            observations[:, :, -1, -3:, -1] = states_expanded.squeeze(2)
        #sampled_actions = self.forward(observations,attention_weight)
        #sampled_actions = self.forward_pro(states,observations)
        sampled_actions = self.forward(observations)
        
        #pdb.set_trace()
        # Compute loss between sampled actions and target actions
        loss = criterion(sampled_actions, actions[:,-1,:])

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()
        # current_lr = self.optimizer.param_groups[0]['lr']
        # print(f"Current learning rate: {current_lr:.6f}")
        return {
            'action Loss': ptu.to_numpy(loss),
        }