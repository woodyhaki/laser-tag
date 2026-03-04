import timm
import torch
import torch.nn as nn
import infrastructure.path_utils as path_utils

import infrastructure.pytorch_util as ptu
import pdb
import numpy as np
from policies.base_policy import *
from policies.MLP_policy import *
from infrastructure.logger import Logger
from soa_data import *
from PIL import Image
from torchvision.transforms import Resize
from training_utils import log_loss
import clip
from torchvision.models.resnet import ResNet

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def tsne_visualize(embeddings,embedding_dim):
    from sklearn.manifold import TSNE
    if isinstance(embeddings,torch.Tensor):
        sample_embeddings = embeddings.detach().cpu().numpy()  #[seq_len, dim]
        sample_embeddings = sample_embeddings.reshape((-1,embedding_dim))
    else:
        sample_embeddings = embeddings.reshape((-1,embedding_dim))
    perplexity = 100
    total_num_sample = sample_embeddings.shape[0]
    if perplexity >= total_num_sample:
        perplexity=total_num_sample - 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(sample_embeddings)  #[seq_len, 2]
    return embeddings_2d

def rgb_to_gray(tensor):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device).view(1, 1, 3, 1, 1)
    gray_tensor = (tensor * weights).sum(dim=2, keepdim=True)
    return gray_tensor

def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = 224
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    # print(f"{h} {w} {p}")
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 224
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

class DropoutMask(nn.Module):
    def __init__(self):
        super(DropoutMask, self).__init__()

    def forward(self, x, dropout_probs):
        """
        x: Input tensor of shape [batch_size, channels, height, width].
        dropout_probs: Tensor of shape [batch_size], each value is the dropout probability for the corresponding sample.
        """
        if not self.training or dropout_probs is None:
            return x
        
        # Expand dropout_probs to match the feature map shape
        batch_size, channels, height, width = x.shape
        # Reshape dropout_probs to [batch_size, 1, 1, 1] and broadcast
        #pdb.set_trace()
        dropout_probs = dropout_probs.view(batch_size, 1, 1, 1).to(x.device)
        
        # Generate dropout mask for each sample
        mask = torch.bernoulli(1 - dropout_probs).expand_as(x)
        
        # Apply mask and rescale
        return x * mask / (1 - dropout_probs + 1e-7)

from resnet_encoder import resnet_single

class ResNetEncoder(nn.Module):
    def __init__(self, params,model_name,input_is_depth,weight_path = None):
        super(ResNetEncoder, self).__init__()
        pretrained = (weight_path is not None)
        print(f'current image network is {model_name}')
        if model_name == 'resnet18':
            self.encoder = resnet_single.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.encoder = resnet_single.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.encoder = resnet_single.resnet50(pretrained=pretrained)
        elif model_name == 'resnet10':
            self.encoder = ResNet10()
            
        elif model_name == 'clip_resnet50':
            model, _ = clip.load("RN50", device="cpu")
            self.encoder = model.visual
            
        elif model_name == 'ViT':
            model, _ = clip.load("ViT-B/32", device="cpu") #ViT-B/32
            self.encoder = model.visual
        elif model_name == 'MAE':
            import timm
            self.encoder = timm.create_model('mae_base_patch16_224', pretrained=True)
        else:
            raise Exception(f"Wrong resnet type!{model_name}")
        #pdb.set_trace()
        if weight_path and model_name in ['resnet18','resnet34','resnet50'] :
            print("fffffffffffffffffffffff loaded!")
            self.encoder.load_state_dict(torch.load(weight_path,weights_only=True))
        
        
        # if weight_path and model_name == 'clip_resnet50':
        #    # self.encoder = torch.jit.load("pretrained_models/clip_resnet50_state_dict", map_location="cpu")
        #     self.encoder.load_state_dict(torch.load(weight_path,weights_only=True))
            
            #"pretrained_models/clip_resnet50_state_dict"
        # state_dict = self.encoder.state_dict()

        # torch.save(state_dict, "pretrained_models/clip_resnet50_state_dict.pth")
        # exit()
        if input_is_depth or params['color_num_channel']== 1:
            old_weights = self.encoder.conv1.weight
            new_weights = old_weights.mean(dim=1, keepdim=True)
            self.encoder.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=self.encoder.conv1.out_channels,
                kernel_size=self.encoder.conv1.kernel_size,
                stride=self.encoder.conv1.stride,
                padding=self.encoder.conv1.padding,
                bias=self.encoder.conv1.bias
            )
            self.encoder.conv1.weight.data = new_weights
        # else:
        #     nn.init.kaiming_normal_(self.encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
        if model_name in ['resnet18','resnet34','resnet50'] :
            self.encoder.fc = nn.Identity()
        self.encoder = self.encoder.cuda()


    def forward(self, x):
        return self.encoder(x)

    def forward_at(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        x = self.encoder.maxpool(x)

        g0 = self.encoder.layer1(x)
        g1 = self.encoder.layer2(g0)
        g2 = self.encoder.layer3(g1)
        g3 = self.encoder.layer4(g2)

        return [g.pow(2).mean(1) for g in (g0, g1, g2,g3)]

class ResNetAT(ResNetEncoder):
    """
    Overloaded ResNet model to return attention maps.
    """
    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        g0 = self.encoder.layer1(x)
        g1 = self.encoder.layer2(g0)
        g2 = self.encoder.layer3(g1)
        
        return [g.pow(2).mean(1) for g in (g0, g1, g2)]
    

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNet10(nn.Module):
    def __init__(self):
        super(ResNet10, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, 2, stride=2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward_with_feature_map(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.avgpool(x1)
        x3 = torch.flatten(x2, 1)
        return x3,x1

class OmniImageEncoderTransformer(nn.Module):
    def __init__(self):
        super(OmniImageEncoderTransformer, self).__init__()
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).cuda()
        self.image_encoder.eval()

    def forward(self,image,dropout_prob=None):
        batch_size, seq_len, channel,height,width = image.shape
        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        #pdb.set_trace()
        
        patch_embedding = self.image_encoder(image_input)
        patch_embedding = patch_embedding.reshape((batch_size,seq_len,-1))
        return patch_embedding
    
def build_multimae():
    import MultiMAE
    from functools import partial
    from MultiMAE.multimae.input_adapters import PatchedInputAdapter, SemSegInputAdapter
    from MultiMAE.multimae.output_adapters import SpatialOutputAdapter
    DOMAINS = ['rgb', 'depth', 'semseg']
    DOMAIN_CONF = {
        'rgb': {
            'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
            'output_adapter': partial(SpatialOutputAdapter, num_channels=3, stride_level=1),
        },
        'depth': {
            'input_adapter': partial(PatchedInputAdapter, num_channels=1, stride_level=1),
            'output_adapter': partial(SpatialOutputAdapter, num_channels=1, stride_level=1),
        },
        'semseg': {
            'input_adapter': partial(SemSegInputAdapter, num_classes=133,
                                    dim_class_emb=64, interpolate_class_emb=False, stride_level=4),
            'output_adapter': partial(SpatialOutputAdapter, num_channels=133, stride_level=4),
        },
    }
    input_adapters = {
        domain: dinfo['input_adapter'](
            patch_size_full=16,
        )
        for domain, dinfo in DOMAIN_CONF.items()
    }
    from MultiMAE.multimae import MultiMAE
    multimae = MultiMAE(
        input_adapters=input_adapters,
        output_adapters=None,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    CKPT_URL = 'https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth'
    ckpt = torch.hub.load_state_dict_from_url(CKPT_URL, map_location='cpu')
    multimae.load_state_dict(ckpt['model'], strict=False)
    return multimae

class ImageEncoder(nn.Module):
    def __init__(self,
                 params,
                 input_is_depth,
                 image_encoder_name, 
                 image_encoder_path,
                 ):
        super(ImageEncoder, self).__init__()
        self.image_encoder_name = image_encoder_name
        self.image_encoder_path = image_encoder_path
        if image_encoder_name == 'multimae':
            self.image_encoder = build_multimae()
        elif image_encoder_name == 'dinov2':
            self.image_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        else:
            self.image_encoder = ResNetEncoder(params,image_encoder_name,input_is_depth, image_encoder_path)
        self.image_encoder.to(ptu.device)
        
        if image_encoder_name == 'resnet18' or image_encoder_name == 'resnet34':
            self.image_embedding_dim = 512
        elif image_encoder_name == 'resnet50':
            self.image_embedding_dim = 2048
        elif image_encoder_name == 'resnet10':
            self.image_embedding_dim = 64
        elif image_encoder_name == 'clip_resnet50':
            self.image_embedding_dim = 1024
        elif image_encoder_name == 'ViT':
            self.image_embedding_dim = 512
        elif image_encoder_name == 'multimae':
            self.image_embedding_dim = 768
        elif image_encoder_name == 'dinov2':
            self.image_embedding_dim = 768
        else:
            raise Exception(f"Wrong resnet type! {image_encoder_name}")
        
        self.image_encoder_out_dim = params['image_encoder_out_dim']
        self.image_encoder_linear_projection = build_mlp(   input_size = self.image_embedding_dim,
                                                            output_size = self.image_encoder_out_dim,
                                                            n_layers = params['n_layers'], 
                                                            size = params['layer_size'] )
        self.image_encoder_linear_projection.to(ptu.device)

    def get_raw_image_embedding(self,image):
        batch_size, seq_len, channel,height,width = image.shape
        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        raw_image_embedding = self.image_encoder(image_input)
        raw_image_embedding = raw_image_embedding.reshape((batch_size,seq_len,-1))
        return raw_image_embedding
        
    def forward(self,image):
        batch_size, seq_len, channel,height,width = image.shape
        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        raw_image_embedding = self.image_encoder(image_input)
        raw_image_embedding = raw_image_embedding.reshape((batch_size,seq_len,-1))
        image_embedding = self.image_encoder_linear_projection(raw_image_embedding)
        #print(f'raw_image_embedding {raw_image_embedding.shape},image_embedding {image_embedding.shape}')
        return image_embedding

    def forward_multi_mae(self,depth,image):
        """ MultiMAE forward takes depth and image
        """
        batch_size, seq_len, channel,height,width = image.shape
        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        depth_input = depth.reshape((batch_size * seq_len,1,height,width))
        
        # Pre-process RGB, depth and semseg to the MultiMAE input format
        input_dict = {}
        #pdb.set_trace()
        # Normalize RGB
        input_dict['rgb'] = image_input

        # Normalize depth robustly
        trunc_depth = torch.sort(depth_input.flatten())[0]
        trunc_depth = trunc_depth[int(0.1 * trunc_depth.shape[0]): int(0.9 * trunc_depth.shape[0])]
        depth_input = (depth_input - trunc_depth.mean()[None,None,None]) / torch.sqrt(trunc_depth.var()[None,None,None] + 1e-6)
        input_dict['depth'] = depth_input

        input_dict = {k: v.to(ptu.device) for k,v in input_dict.items()}
        num_encoded_tokens = 199 # the number of visible tokens
        alphas = 1.0 # Dirichlet concentration parameter

        raw_image_embedding, masks = self.image_encoder.forward(
            input_dict, 
            mask_inputs=True, # True if forward pass should sample random masks
            num_encoded_tokens=num_encoded_tokens,
            alphas=alphas
        )
        #print(f"raw_image_embedding {raw_image_embedding.shape}")

        raw_image_embedding = raw_image_embedding.reshape((batch_size,seq_len,num_encoded_tokens + 1,-1)) ## Use CLS token
        #print(f"raw_image_embedding {raw_image_embedding.shape}")
        #pdb.set_trace()
        
        image_embedding = self.image_encoder_linear_projection(raw_image_embedding[:,:,-1,:])
        return image_embedding


    def get_feature_map(self,image):
        batch_size, seq_len, channel,height,width = image.shape
        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        raw_image_embedding,feature_map = self.image_encoder.encoder.forward_with_feature_map(image_input)
        raw_image_embedding = raw_image_embedding.reshape((batch_size,seq_len,-1))
        _,feature_channel,feature_h,feature_w = feature_map.shape
        feature_map = feature_map.reshape((batch_size,seq_len,feature_channel,feature_h,feature_w))
        #image_embedding = self.image_encoder_linear_projection(raw_image_embedding)
        
        #print(f'raw_image_embedding {raw_image_embedding.shape},image_embedding {image_embedding.shape}')
        return raw_image_embedding,feature_map

    def get_teacher_token(self, depth ,image):
        assert self.image_encoder_name == 'multimae'
        """ MultiMAE forward takes depth and image
            Return [Batch_size * Seq_len ,  Number_Token, Token_dim ]
        """
        batch_size, seq_len, channel,height,width = image.shape
        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        depth_input = depth.reshape((batch_size * seq_len,1,height,width))
        
        # Pre-process RGB, depth and semseg to the MultiMAE input format
        input_dict = {}
        #pdb.set_trace()
        # Normalize RGB
        input_dict['rgb'] = image_input

        # Normalize depth robustly
        trunc_depth = torch.sort(depth_input.flatten())[0]
        trunc_depth = trunc_depth[int(0.1 * trunc_depth.shape[0]): int(0.9 * trunc_depth.shape[0])]
        depth_input = (depth_input - trunc_depth.mean()[None,None,None]) / torch.sqrt(trunc_depth.var()[None,None,None] + 1e-6)
        input_dict['depth'] = depth_input

        input_dict = {k: v.to(ptu.device) for k,v in input_dict.items()}
        num_encoded_tokens = 199 # the number of visible tokens
        alphas = 1.0 # Dirichlet concentration parameter

        raw_image_token, masks = self.image_encoder.forward(
            input_dict, 
            mask_inputs=True, # True if forward pass should sample random masks
            num_encoded_tokens=num_encoded_tokens,
            alphas=alphas
        )
        #print(f"raw_image_token {raw_image_token.shape}")
        return raw_image_token

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'image_encoder_name': self.image_encoder_name,
            'image_encoder_path': self.image_encoder_path,
        }, path)
        print(f"Model saved to {path}")
        
class OmniImageEncoder(nn.Module):
    def __init__(self,params, image_encoder_name, image_encoder_path=None, load_resnet_from_checkpoint=False):
        super(OmniImageEncoder, self).__init__()
        self.dynamic_dropout = params['dynamic_dropout']
        
        if load_resnet_from_checkpoint and image_encoder_path is not None:
            #pdb.set_trace()
            self.image_encoder = ResNetEncoder(params,image_encoder_name, image_encoder_path).cuda()
            #self.image_encoder = ResNetEncoder(params,image_encoder_name).cuda()
        else:
            self.image_encoder = ResNetEncoder(params,image_encoder_name).cuda()
        self.image_encoder_name = image_encoder_name
        self.image_encoder_path = image_encoder_path
        self.block_size = 180
        self.omni_image_width = 1640
        num_block = ((self.omni_image_width + self.block_size - 1) // self.block_size)

        target_width = num_block * self.block_size
        print("target_width:",target_width)
        self.padding_width = target_width - self.omni_image_width
        self.num_patches = self.omni_image_width // self.padding_width

    def forward(self,image,dropout_prob=None):
        # if self.dynamic_dropout:
        #     assert dropout_prob is not None
        #pdb.set_trace()
        batch_size, seq_len, channel,height, width = image.shape

        image_input = image.reshape((batch_size * seq_len,channel,height,width))
        if False:  ## patchify
            image_padded = F.pad(image, (0, self.padding_width))
            patch_embeddings = []
            for i in range(self.num_patches):
                patch = image_padded[:, :, :,i * self.block_size:(i + 1) * self.block_size]
                #resized_batch = torch.stack([_transform(image) for image in patch])
                #print("resized_batch:",resized_batch.shape)
                
                #patch_embedding = self.image_encoder.encode_image(resized_batch.cuda())
                if self.dynamic_dropout and (dropout_prob is not None):
                    patch_embedding = self.image_encoder(patch,dropout_prob)
                else:
                    #print("patch shape:",patch.shape)
                    patch_embedding = self.image_encoder(patch)

                patch_embeddings.append(patch_embedding)
            #
            ## batch, seq_len * embedding_dim * num_patches
            patch_embedding = torch.stack(patch_embeddings).permute((1,0,2)).reshape((batch_size,seq_len,-1))
            ## batch, seq_len, embedding_dim * num_patches
            #pdb.set_trace()
        else:
            #pdb.set_trace()
            patch_embedding = self.image_encoder(image_input)
            patch_embedding = patch_embedding.reshape((batch_size,seq_len,-1))
        return patch_embedding

    # def forward(self,image):
    #     #batch_size, channel,height,width = image.shape
    #     #pdb.set_trace()
    #     patch_embedding = self.image_encoder(image)
    #     return patch_embedding
    
    def save(self, path):
        """
        Save the entire model, including the weights of the submodule ResNetEncoder
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'image_encoder_name': self.image_encoder_name,
            'image_encoder_path': self.image_encoder_path,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, params, path, load_resnet_from_checkpoint=False):
        """
        Load the model:
        1. If load_resnet_from_checkpoint=True, then load the ResNetEncoder checkpoint.
        2. Otherwise, directly load the entire OmniImageEncoder checkpoint.
        """
        checkpoint = torch.load(path)
        model = cls(
            params=params,
            image_encoder_name=checkpoint['image_encoder_name'],
            image_encoder_path=checkpoint['image_encoder_path'],
            load_resnet_from_checkpoint=load_resnet_from_checkpoint
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"OmniImageEncoder loaded from {path}")
        return model

class ImageContrastiveLearningLoss(nn.Module):
    def __init__(self, params, image_encoder_name, image_encoder_path):

        super(ImageContrastiveLearningLoss, self).__init__()
        self.image_encoder = ImageEncoder(    params,
                                                   input_is_depth=False,
                                                   image_encoder_name = image_encoder_name,
                                                   image_encoder_path = image_encoder_path     ).cuda()

        
        self.tau = params['tau']
        self.learning_rate = params['learning_rate']
        self.optimizer = optim.Adam(self.image_encoder.parameters(),self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5)

    def soft_nearest_neighbor_loss(self,anchor_embeddings, positive_embeddings, all_embeddings,tau=None):
        anchor_norm = F.normalize(anchor_embeddings, dim=-1)
        positive_norm = F.normalize(positive_embeddings, dim=-1)
        all_norm = F.normalize(all_embeddings, dim=-1)

        # Similarity between anchor and positive (numerator)
        sim_pos = torch.sum(anchor_norm * positive_norm, dim=-1)  # Shape: (batch_size,num_positive), sum is for the dot product
        sim_all = torch.sum(anchor_norm * all_norm, dim=-1)       # Shape: (batch_size, ) sum is for the dot product
        #pdb.set_trace()
        #print(tau[1,:])
        #exp_sim_pos = torch.exp(-sim_pos / tau[:,1:])
        #exp_sim_all = torch.exp(-sim_all / tau)
        
        
        exp_sim_pos = torch.exp(-sim_pos / 5)
        exp_sim_all = torch.exp(-sim_all / 5)
        
        
        nom = exp_sim_pos.sum(dim=-1)
        denom = exp_sim_all.sum(dim=-1)  # Shape: (batch_size,)

        # Compute SNN loss
        loss = -torch.log(nom / denom)  # Shape: (batch_size,)
        if torch.isinf(loss).any():
            pdb.set_trace()
        return loss.mean()

    def forward(self, image_anc, 
                      other_image, 
                      all_image,
                      all_image_aug,
                      tau = None):
        anchor_embeddings = self.image_encoder(image_anc.to(ptu.device))
        other_embeddings = self.image_encoder(other_image.to(ptu.device))
        all_embeddings = self.image_encoder(all_image.to(ptu.device))
        pdb.set_trace()
        inter_scene_loss = self.soft_nearest_neighbor_loss(anchor_embeddings,other_embeddings,all_embeddings,tau)
        intra_scene_loss = None
        return inter_scene_loss

    def update(self, image_anc, other_image,all_image,tau):
        loss = self(image_anc, other_image,all_image,tau)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'intra-scene Loss': ptu.to_numpy(loss)
        }

    def new_soft_nearest_neighbor_loss(self, embeddings, seq_len, T_pos=0.2, T_neg=0.9):
        batch_size, seq_len, embedding_dim = embeddings.shape
        N = batch_size * seq_len

        embeddings = embeddings.view(N, embedding_dim)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)  # (N, N)

        # Construct label mask
        labels = torch.arange(batch_size).repeat_interleave(seq_len).to(embeddings.device)
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Remove self-comparison
        mask_self = 1 - torch.eye(N, device=embeddings.device)
        label_mask = label_mask * mask_self

        # Create temperature matrix
        T_mat = T_pos * label_mask + T_neg * (1 - label_mask + 1e-10)  # avoid div 0

        # Apply adaptive temperature
        sim_scaled = similarity_matrix / T_mat
        exp_matrix = torch.exp(sim_scaled) * mask_self  # optional: zero out diagonal explicitly

        num = (exp_matrix * label_mask).sum(dim=1)
        denom = exp_matrix.sum(dim=1)

        loss = -torch.log(num / (denom + 1e-10)).mean()
        return loss

    def update_new(self,image,image_aug,tau = None):
        ## intra-scene
        image_embeddings = self.image_encoder(image)
    #    image_aug_embeddings = self.image_encoder(image_aug)
        
        intra_scene_loss = self.new_soft_nearest_neighbor_loss(image_embeddings,seq_len=5)
        #inter_scene_loss = self.new_soft_nearest_neighbor_loss(image_aug_embeddings,seq_len=5,temperature=tau)
        loss = intra_scene_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'intra-scene Loss': ptu.to_numpy(intra_scene_loss),
         #   'inter-scene Loss': ptu.to_numpy(inter_scene_loss),
            'total Loss': ptu.to_numpy(loss)
        }
        

def augment_images(image):
    """
    Apply random contrast and brightness augmentation to a batch of input images.

    Args:
    - image: torch.Tensor, shape (batch_size, seq_len, C, H, W)

    Returns:
    - augmented_image: torch.Tensor, same shape as input
    """
    batch_size, seq_len, C, H, W = image.shape
    import torchvision.transforms as T
    # Define data augmentation transforms
    transform = T.Compose([
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8),  # 80% chance to adjust brightness and contrast
    ])

    # Process each frame in the batch
    image_list = []
    for b in range(batch_size):
        seq_images = []
        for s in range(seq_len):
            img = image[b, s]  # (C, H, W)
            img = transform(img)  # Apply transform
            seq_images.append(img)
        image_list.append(torch.stack(seq_images))  # Stack into (seq_len, C, H, W)

    augmented_image = torch.stack(image_list)  # Shape (batch_size, seq_len, C, H, W)

    return augmented_image

def compuate_delta_pose(state_time_series):
    #print(state_time_series.shape)
    batch_size = state_time_series.shape[0]
    state_anchor = state_time_series[:,0,:]
    tau = torch.ones((batch_size,state_time_series.shape[1]))
    for i in range(1,state_time_series.shape[1]):
        state_ = state_time_series[:,i,:]
        delta_ = state_ - state_anchor
        delta_[:,2] /= torch.pi
        tau[:,i] = torch.norm(delta_[:,0:2],p=1) * torch.abs(delta_[:,2])
    return tau

def train_v_contrastive_learning(params):
    ptu.init_gpu()
    num_channel = params['color_num_channel']
    batch_size = params['batch_size']
    seq_len = params['seq_len']
    train_data_loader,test_data_loader,n_data = make_data_loader(params)
    
    
    ## timestamp   batch, seq_len
    ## image       batch, seq_len, channel, height, width
    ## action      batch, seq_len, action
    torch.set_printoptions(precision=16, sci_mode = False)
    
    smm_folder = path_utils.create_path_to_folder('./run_log/')
    model_path = path_utils.create_path_to_folder('./contrastive_models')
    logger = Logger(smm_folder)
    training_logs = []
    total_epochs = 1000
    image_encoder_name = params['test_image_encoder_name']
    image_encoder_path = params['pretrained_image_encoder_path']
    logger.log_text("Contrastive Experiment Parameters:",params)

    
    icl = ImageContrastiveLearningLoss(params, image_encoder_name, image_encoder_path)
    for current_epoch in range(total_epochs):
        for id,data in enumerate(train_data_loader):
            time_stamp,states,image,depth,image_original,actions,id = data
            if num_channel == 3: ## 3 channel
                image = image.to(ptu.device)
                #image_anc = image[:,0,:,:,:].unsqueeze(1)
                #image_other = image[:,1:,:,:,:]
            elif num_channel == 1: ## single channel
                image = rgb_to_gray(image)
                image = image.to(ptu.device)
                image_anc = image[:,0,:,:,:].unsqueeze(1)
                image_other = image[:,1:,:,:,:]
            else:
                raise Exception(f"channel error number of channle = {num_channel}")
            tau = compuate_delta_pose(states).to(ptu.device)
            image_aug = augment_images(image)
            contrastive_loss = icl.update_new(image,image_aug,tau)
        training_logs.append(contrastive_loss)
        print(f"epoch {current_epoch} contrastive loss {contrastive_loss}")
        if current_epoch % 5 == 0:
            # # Save the model periodically
            ckpt_path = f"{model_path}/contrastive_encoder_{current_epoch}.pth"
            icl.image_encoder.save(ckpt_path)
        #pdb.set_trace()
        log_loss(current_epoch,logger,training_logs)

def test_dropout():
    params = {'num_channel': 1}
    model = ResNetEncoder(params, model_name='resnet18')

    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 224, 224).cuda()
    dropout_probs = torch.tensor([0.1, 0.3, 0.5, 0.7]).cuda()

    # Forward
    output = model(input_tensor, dropout_probs)

def test_resnet_encoder(params):
    weight_path = 'pretrained_models/resnet18-5c106cde.pth'
    model = ResNetEncoder(params, model_name='resnet10',input_is_depth=False,weight_path=weight_path)

    #model = timm.create_model('resnet50', pretrained=True)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()
    output = model(input_tensor)
    print("shape of output:",output.shape)

def test_vit_encoder():
    model = timm.create_model('vit_base_patch16_224', pretrained=True).cuda() 
    model.eval() 
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).cuda() 
    
    with torch.no_grad():
        output = model(input_tensor)
        print("output shape:",output.shape)

def test_model_memory():
    from torchinfo import summary
    torch.cuda.empty_cache()   # Clear GPU memory cache (optional)

    before_memory = torch.cuda.memory_allocated()  # Record memory before model loading
    # model = ResNet10().cuda()
    model = resnet_single.resnet18(pretrained=False).cuda()

    after_memory = torch.cuda.memory_allocated()   # Record memory after model loading
    summary(model, input_size=(1, 3, 224, 224), device="cuda")
    print(f"Model GPU memory usage: {(after_memory - before_memory) / 1024**2:.2f} MB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--color_num_channel', type=int,default=3)
    parser.add_argument('--use_depth', type=int,default=0,help='0:rgb only | 1:depth only | 2:rgbd')
    parser.add_argument('--batch_size', type=int,default=16)
    parser.add_argument('--layer_size',type=int,default=512)
    parser.add_argument('--pro_state_dim',type=int,default=1024)
    parser.add_argument('--image_encoder_out_dim',type=int,default=1024)

    parser.add_argument('--n_layers',type=int,default=2)
    parser.add_argument('--seq_len', type=int,default=5)

    parser.add_argument('--tau', type=float,default=1e-01)
    parser.add_argument('--load_finetuned', type=bool,default=False)
    parser.add_argument('--dynamic_dropout', type=bool,default=False)
    parser.add_argument('--freeze_encoder', type=int,default=0)
    parser.add_argument('--freeze_action', type=int,default=0)
    
    parser.add_argument('--total_epochs', type=int,default=200)
    parser.add_argument('--teacher_image_encoder_name', type=str,default='resnet50')
    parser.add_argument('--student_image_encoder_name', type=str,default='resnet10')
    parser.add_argument('--test_image_encoder_name', type=str,default='resnet50')
    
    parser.add_argument('--pretrained_image_encoder_path', type=str,default='pretrained_models/resnet50-0676ba61.pth')
    #image_encoder_path = 'pretrained_models/resnet18-5c106cde.pth'
    #image_encoder_path = 'pretrained_models/resnet34-43635321.pth'
    #image_encoder_path = 'pretrained_models/resnet50-0676ba61.pth'
    
    parser.add_argument('--dump_train', type=int,default=1)
    parser.add_argument('--small_data', type=int,default=0)
    parser.add_argument('--save_model', type=int,default=1)
    parser.add_argument('--learning_rate', type=float,default=1e-06)
    
    args = parser.parse_args()
    # convert args to dictionary
    params = vars(args)
    #train_v_contrastive_learning(params)
    
    #test_dropout()
    #test_resnet_encoder(params)
    #test_vit_encoder()
    test_model_memory()