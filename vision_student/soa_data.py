
import torch
import torch.nn as nn
import numpy as np
import pickle
import pdb
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
import infrastructure.pytorch_util as ptu
import matplotlib.pyplot as plt
from data_utils import State_Observation_Action_Saliency
import os
import re
import cv2
import time
import h5py

def generate_teacher_gradient_labels(policy, states, observations, actions):
    observations = observations.to(ptu.device)
    actions = actions.to(ptu.device)
    states = states.to(ptu.device)
    
    observations.requires_grad_()
    sampled_actions = policy.forward_pro(states,observations)
    
    
    # Compute loss between sampled actions and target actions
    criterion_task = nn.MSELoss()
    loss_action = criterion_task(sampled_actions, actions[:,-1,:])
    
    policy.optimizer.zero_grad()
    loss_action.backward(retain_graph=True)  # Retain the graph for the next backward pass

    ###===================Use gradient=====================
    gradient_map = observations.grad
    # Normalize the gradient map if necessary (optional step for stability)
    batch_gradient_map = gradient_map / gradient_map.norm(p=2, dim=(2, 3, 4), keepdim=True)

    #======================================================
    return batch_gradient_map

class SOADataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str):
        with open(data_dir, 'rb') as f:
             self.soa_list = pickle.load(f)
        self.img_list = []
        self.state_list = []
        self.action_list = []
        for soa in self.soa_list:
            image_rot = np.transpose(soa.observation,(2,0,1))
            image = torch.FloatTensor(image_rot)
            state = torch.FloatTensor(soa.state)
            action = torch.FloatTensor(soa.action)
            self.img_list.append(image)
            self.state_list.append(state)
            self.action_list.append(action)

        if len(self.soa_list) > 0:
            self.observation_dim = self.soa_list[0].observation.shape
        else:
            self.observation_dim = None

    def __len__(self) -> int:
        return len(self.soa_list)
    
    def __getitem__(self, idx: int):
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        res = ( self.soa_list[idx].time_stamp,
                self.state_list[idx],
                self.img_list[idx],
                self.action_list[idx] )
        return res

from data_utils import State_Observation_Action

class SOA_MultiView_TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str,
                 seq_len: int):
        start_time = time.time()
        with h5py.File(data_dir, 'r') as f:
            self.img_0_list = f['img_0'][:]
            self.img_1_list = f['img_1'][:]
            self.img_2_list = f['img_2'][:]
            self.img_3_list = f['img_3'][:]
            
            self.depth_0_list = f['depth_0'][:]
            self.depth_1_list = f['depth_1'][:]
            self.depth_2_list = f['depth_2'][:]
            self.depth_3_list = f['depth_3'][:]

            self.state_list = f['states'][:]
            self.action_list = f['actions'][:]
            self.time_stamps = f['time_stamps'][:]
             
        end_time = time.time()
        print(f"{data_dir} Total time: {end_time - start_time:.4f} s")
        self.seq_len = seq_len
        self.data_len = self.img_resize_list.shape[0]
        if len(self.img_resize_list) > 0:
            self.observation_dim = self.img_resize_list[0].shape
        else:
            self.observation_dim = None

        self.max_time_gap = 0.5
        self.valid_indices = []
        for idx in range(self.data_len - self.seq_len + 1):
            window_ts = np.array(self.time_stamps[idx:idx + self.seq_len])
            diffs = np.diff(window_ts)
            if np.all(np.abs(diffs) <= self.max_time_gap):
                self.valid_indices.append(idx)

    def show_img_tensor(self,img_tensor):
        img_to_show = img_tensor.cpu().numpy().transpose(1, 2, 0)
        print(img_to_show.dtype)
        print(f'img shape:{img_to_show.shape}')
        if img_to_show.max() > 1:
            img_to_show = np.clip(img_to_show, 0, 255).astype(np.uint8)
        plt.matshow(img_to_show)
        plt.show()


    def __len__(self) -> int:
        return max(1, self.data_len - self.seq_len + 1)
        #return len(self.soa_list)
    
    
    def __getitem__(self, idx: int):
        """
        Returns a sequence of data, where idx is the start index for the window.
        """
        start_idx = self.valid_indices[idx]
        w_time_stamp_list = []
        ##----------------------
        w_img_0_list = []
        w_depth_0_list = []
        
        w_img_1_list = []
        w_depth_1_list = []
        
        w_img_2_list = []
        w_depth_2_list = []
        
        w_img_3_list = []
        w_depth_3_list = []
        ##----------------------
        w_state_list = []

        w_img_original_list = []
        w_action_list = []
        
        
        #print(self.data_len)
        # Get a sliding window of `seq_len`
        for i in range(self.seq_len):
            data_idx = start_idx + i
            if data_idx < self.data_len:
                w_time_stamp_list.append(self.time_stamps[data_idx])
                
                ##-------------------
                w_img_0_list.append(torch.FloatTensor(self.img_0_list[data_idx]))
                w_depth_0_list.append(torch.FloatTensor(self.depth_0_list[data_idx]))
                
                w_img_1_list.append(torch.FloatTensor(self.img_1_list[data_idx]))
                w_depth_1_list.append(torch.FloatTensor(self.depth_1_list[data_idx]))
                
                w_img_2_list.append(torch.FloatTensor(self.img_2_list[data_idx]))
                w_depth_2_list.append(torch.FloatTensor(self.depth_2_list[data_idx]))
                
                w_img_3_list.append(torch.FloatTensor(self.img_3_list[data_idx]))
                w_depth_3_list.append(torch.FloatTensor(self.depth_3_list[data_idx]))
                ##-------------------
                
                w_state_list.append(torch.FloatTensor(self.state_list[data_idx]))

                w_img_original_list.append(torch.FloatTensor(self.img_original_list[data_idx]))
                w_action_list.append(torch.FloatTensor(self.action_list[data_idx]))

        # Pad with zeros if the sequence ends early (only in the last window)
        while len(w_time_stamp_list) < self.seq_len:
            w_time_stamp_list.append(0.0)  # Or use np.nan for time stamps
            w_state_list.append(torch.zeros_like(self.state_list[0]))
            
            ##-----------------------------------
            w_img_0_list.append(torch.zeros_like(self.img_0_list[0]))
            w_depth_0_list.append(torch.zeros_like(self.depth_0_list[0]))
            
            w_img_1_list.append(torch.zeros_like(self.img_1_list[0]))
            w_depth_1_list.append(torch.zeros_like(self.depth_1_list[0]))
            
            w_img_2_list.append(torch.zeros_like(self.img_2_list[0]))
            w_depth_2_list.append(torch.zeros_like(self.depth_2_list[0]))
            
            w_img_3_list.append(torch.zeros_like(self.img_3_list[0]))
            w_depth_3_list.append(torch.zeros_like(self.depth_3_list[0]))
            ##-----------------------------------
            
            w_img_original_list.append(torch.zeros_like(self.img_original_list[0]))
            w_action_list.append(torch.zeros_like(self.action_list[0]))

        w_time_stamp = np.array(w_time_stamp_list, dtype=np.float32)
        
        ##----------------------
        w_img_0 = torch.stack(w_img_0_list)
        w_depth_0 = torch.stack(w_depth_0_list)
        
        w_img_1 = torch.stack(w_img_1_list)
        w_depth_1 = torch.stack(w_depth_1_list)
        
        w_img_2 = torch.stack(w_img_2_list)
        w_depth_2 = torch.stack(w_depth_2_list)
        
        w_img_3 = torch.stack(w_img_3_list)
        w_depth_3 = torch.stack(w_depth_3_list)
        ##----------------------

        w_state = torch.stack(w_state_list)
        w_action = torch.stack(w_action_list)
        
        ret = {
            'time_stamp': w_time_stamp,
            'img_0':      w_img_0,
            'img_1':      w_img_1,
            'img_2':      w_img_2,
            'img_3':      w_img_3,
            'depth_0':    w_depth_0,
            'depth_1':    w_depth_1,
            'depth_2':    w_depth_2,
            'depth_3':    w_depth_3,
            'state':      w_state,
            'action':     w_action,
            'idx':        idx
        }
        return ret

class SOATimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str,
                 seq_len: int):
        ##=================LOAD pickle data=======================
        # with open(data_dir, 'rb') as f:
        #      self.soa_list = pickle.load(f)
        # print("pickle data load ok")
        ##========================================================
        start_time = time.time()
        with h5py.File(data_dir, 'r') as f:
            print("Data keys:")
            print(list(f.keys()))
            self.img_resize_list = f['img_resize'][:]
            self.img_original_list = f['img_original'][:]
            self.state_list = f['states'][:]
            self.action_list = f['actions'][:]
            self.depth_img_list = f['depth_img'][:]
            self.time_stamps = f['time_stamps'][:]
            self.heatmap0_list = f['heatmaps_resize_0'][:]
            self.heatmap1_list = f['heatmaps_resize_1'][:]
            
             
        end_time = time.time()
        print(f"{data_dir} Total time: {end_time - start_time:.4f} s")
        

        self.seq_len = seq_len
        self.data_len = self.img_resize_list.shape[0]
        if len(self.img_resize_list) > 0:
            self.observation_dim = self.img_resize_list[0].shape
        else:
            self.observation_dim = None

        self.max_time_gap = 0.5
        self.valid_indices = []
        for idx in range(self.data_len - self.seq_len + 1):
            window_ts = np.array(self.time_stamps[idx:idx + self.seq_len])
            diffs = np.diff(window_ts)
            if np.all(np.abs(diffs) <= self.max_time_gap):
                self.valid_indices.append(idx)

    def show_img_tensor(self,img_tensor):
        img_to_show = img_tensor.cpu().numpy().transpose(1, 2, 0)
        print(img_to_show.dtype)
        print(f'img shape:{img_to_show.shape}')
        if img_to_show.max() > 1:
            img_to_show = np.clip(img_to_show, 0, 255).astype(np.uint8)
        plt.matshow(img_to_show)
        plt.show()

    def __len__(self) -> int:
        return max(1, self.data_len - self.seq_len + 1)
        #return len(self.soa_list)

    def __getitem__(self, idx: int):
        """
        Returns a sequence of data, where idx is the start index for the window.
        """
        w_time_stamp_list = []
        w_state_list = []
        w_img_resize_list = []
        w_depth_resize_list = []
        w_img_original_list = []
        w_action_list = []
        w_heatmap0_list = []
        w_heatmap1_list = []
        
        #print(self.data_len)
        # Get a sliding window of `seq_len`
        for i in range(self.seq_len):
            data_idx = idx + i
            if data_idx < self.data_len:
                w_time_stamp_list.append(self.time_stamps[data_idx])
                w_state_list.append(torch.FloatTensor(self.state_list[data_idx]))
                w_img_resize_list.append(torch.FloatTensor(self.img_resize_list[data_idx]))
                w_depth_resize_list.append(torch.FloatTensor(self.depth_img_list[data_idx]))
                w_heatmap0_list.append(torch.FloatTensor(self.heatmap0_list[data_idx]))
                w_heatmap1_list.append(torch.FloatTensor(self.heatmap1_list[data_idx]))
                w_img_original_list.append(torch.FloatTensor(self.img_original_list[data_idx]))
                w_action_list.append(torch.FloatTensor(self.action_list[data_idx]))

        # Pad with zeros if the sequence ends early (only in the last window)
        while len(w_time_stamp_list) < self.seq_len:
            w_time_stamp_list.append(0.0)  # Or use np.nan for time stamps
            w_state_list.append(torch.zeros_like(torch.FloatTensor(self.state_list[0])))
            w_img_resize_list.append(torch.zeros_like(torch.FloatTensor(self.img_resize_list[0])))
            w_depth_resize_list.append(torch.zeros_like(torch.FloatTensor(self.depth_img_list[0])))
            w_heatmap0_list.append(torch.zeros_like(torch.FloatTensor((self.heatmap0_list[0]))))
            w_heatmap1_list.append(torch.zeros_like(torch.FloatTensor((self.heatmap1_list[0]))))
            w_img_original_list.append(torch.zeros_like(torch.FloatTensor(self.img_original_list[0])))
            w_action_list.append(torch.zeros_like(torch.FloatTensor(self.action_list[0])))

        w_time_stamp = np.array(w_time_stamp_list, dtype=np.float32)
        w_state = torch.stack(w_state_list)
        w_img_resize = torch.stack(w_img_resize_list)
        w_depth_resize = torch.stack(w_depth_resize_list)
        w_heatmap0_resize = torch.stack(w_heatmap0_list)
        w_heatmap1_resize = torch.stack(w_heatmap1_list)
        w_img_original = torch.stack(w_img_original_list)
        w_action = torch.stack(w_action_list)
        return w_time_stamp, w_state, w_img_resize, w_depth_resize, w_heatmap0_resize, w_heatmap1_resize, w_img_original, w_action, idx

class SOATimeSeriesNoHeatMapsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir: str,
                 seq_len: int):
        ##=================LOAD pickle data=======================
        # with open(data_dir, 'rb') as f:
        #      self.soa_list = pickle.load(f)
        # print("pickle data load ok")
        ##========================================================
        start_time = time.time()
        with h5py.File(data_dir, 'r') as f:
            print("Data keys:")
            print(list(f.keys()))
            self.img_resize_list = f['img_resize'][:]
            self.img_original_list = f['img_original'][:]
            self.state_list = f['states'][:]
            self.action_list = f['actions'][:]
            self.depth_img_list = f['depth_img'][:]
            self.time_stamps = f['time_stamps'][:]

        end_time = time.time()
        print(f"{data_dir} Total time: {end_time - start_time:.4f} s")
        

        self.seq_len = seq_len
        self.data_len = self.img_resize_list.shape[0]
        if len(self.img_resize_list) > 0:
            self.observation_dim = self.img_resize_list[0].shape
        else:
            self.observation_dim = None

        self.max_time_gap = 0.5
        self.valid_indices = []
        for idx in range(self.data_len - self.seq_len + 1):
            window_ts = np.array(self.time_stamps[idx:idx + self.seq_len])
            diffs = np.diff(window_ts)
            if np.all(np.abs(diffs) <= self.max_time_gap):
                self.valid_indices.append(idx)

    def show_img_tensor(self,img_tensor):
        img_to_show = img_tensor.cpu().numpy().transpose(1, 2, 0)
        print(img_to_show.dtype)
        print(f'img shape:{img_to_show.shape}')
        if img_to_show.max() > 1:
            img_to_show = np.clip(img_to_show, 0, 255).astype(np.uint8)
        plt.matshow(img_to_show)
        plt.show()

    def __len__(self) -> int:
        return max(1, self.data_len - self.seq_len + 1)
        #return len(self.soa_list)

    def __getitem__(self, idx: int):
        """
        Returns a sequence of data, where idx is the start index for the window.
        """
        w_time_stamp_list = []
        w_state_list = []
        w_img_resize_list = []
        w_depth_resize_list = []
        w_img_original_list = []
        w_action_list = []
        w_heatmap0_list = []
        w_heatmap1_list = []
        
        #print(self.data_len)
        # Get a sliding window of `seq_len`
        for i in range(self.seq_len):
            data_idx = idx + i
            if data_idx < self.data_len:
                w_time_stamp_list.append(self.time_stamps[data_idx])
                w_state_list.append(torch.FloatTensor(self.state_list[data_idx]))
                w_img_resize_list.append(torch.FloatTensor(self.img_resize_list[data_idx]))
                w_depth_resize_list.append(torch.FloatTensor(self.depth_img_list[data_idx]))
                w_img_original_list.append(torch.FloatTensor(self.img_original_list[data_idx]))
                w_action_list.append(torch.FloatTensor(self.action_list[data_idx]))

        # Pad with zeros if the sequence ends early (only in the last window)
        while len(w_time_stamp_list) < self.seq_len:
            w_time_stamp_list.append(0.0)  # Or use np.nan for time stamps
            w_state_list.append(torch.zeros_like(torch.FloatTensor(self.state_list[0])))
            w_img_resize_list.append(torch.zeros_like(torch.FloatTensor(self.img_resize_list[0])))
            w_depth_resize_list.append(torch.zeros_like(torch.FloatTensor(self.depth_img_list[0])))
            w_img_original_list.append(torch.zeros_like(torch.FloatTensor(self.img_original_list[0])))
            w_action_list.append(torch.zeros_like(torch.FloatTensor(self.action_list[0])))

        w_time_stamp = np.array(w_time_stamp_list, dtype=np.float32)
        w_state = torch.stack(w_state_list)
        w_img_resize = torch.stack(w_img_resize_list)
        w_depth_resize = torch.stack(w_depth_resize_list)
        w_img_original = torch.stack(w_img_original_list)
        w_action = torch.stack(w_action_list)
        return w_time_stamp, w_state, w_img_resize, w_depth_resize, w_img_original, w_action, idx
    
class SOASTimeSeriesProcess():
    def __init__(self,
                 data_dir: str,
                 seq_len:  int):
        with open(data_dir, 'rb') as f:
            self.soa_list = pickle.load(f)
        self.seq_len = seq_len
        self.data_len = len(self.soa_list)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def get_saliency_data(self,teacher_policy):
        soas_list = []
        for idx,soa in enumerate(self.soa_list):
            
            w_time_stamp_list = []
            w_state_list = []
            w_img_resize_list = []
            w_img_original_list = []
            w_action_list = []
            #print(self.data_len)
            # Get a sliding window of `seq_len`
            for i in range(self.seq_len):
                data_idx = idx + i
                if data_idx < self.data_len:
                    w_time_stamp_list.append(self.soa_list[data_idx].time_stamp)
                    w_state_list.append(torch.FloatTensor(self.soa_list[data_idx].state))
                    w_img_original_list.append(torch.FloatTensor(self.soa_list[data_idx].observation))
                    w_action_list.append(torch.FloatTensor(self.soa_list[data_idx].action))
                    
                    ## resize image and put into buffer
                    pil_image = Image.fromarray(self.soa_list[data_idx].observation)
                    img_resize = self.transform(pil_image)
                    w_img_resize_list.append(img_resize)

            # Pad with zeros if the sequence ends early (only in the last window)
            while len(w_time_stamp_list) < self.seq_len:
                w_time_stamp_list.append(0.0)  # Or use np.nan for time stamps
                w_state_list.append(torch.zeros_like(w_state_list[0]))
                w_img_original_list.append(torch.zeros_like(w_img_original_list[0]))
                w_action_list.append(torch.zeros_like(w_action_list[0]))
                w_img_resize_list.append(torch.zeros_like(w_img_resize_list[0]))
            

            ## [Seq_len, state_dim]
            w_state = torch.stack(w_state_list)
            
            ## [Seq_len, channel, height, width]
            w_img_resize = torch.stack(w_img_resize_list)
            
            ## [Seq_len, action_dim]
            w_action = torch.stack(w_action_list)

            ## Add batch 1
            action_ = w_action.unsqueeze(0).clone()
            observation_ = w_img_resize.unsqueeze(0).clone()
            state_ = w_state.unsqueeze(0).clone()

            gradient_map = generate_teacher_gradient_labels(teacher_policy,state_,observation_,action_)
            #print(f"saliency map shape {gradient_map.shape}")
            gradient_map_numpy = gradient_map.squeeze(0).detach().cpu().numpy()
            gradient_map_current = gradient_map_numpy[0]
            
            
            time_stamp = soa.time_stamp
            observation = soa.observation
            state = soa.state
            action = soa.action
            #pdb.set_trace()
            
            soas = State_Observation_Action_Saliency(time_stamp, state, observation, action, gradient_map_current)
            soas_list.append(soas)
            #print(time_stamp)

        return soas_list

def load_multiple_data_no_saliency(params,data_path_list):
    seq_len = params['seq_len']
    datasets = []
    for data_path in data_path_list:
        dataset = SOATimeSeriesDataset(data_dir=data_path, seq_len=seq_len)
        datasets.append(dataset)
    # Concatenate all datasets
    combined_dataset = ConcatDataset(datasets)
    print(f'length of total data:{len(combined_dataset)}')
    return combined_dataset,len(combined_dataset)


def load_multiple_data_no_heatmaps(params,data_path_list):
    seq_len = params['seq_len']
    datasets = []
    for data_path in data_path_list:
        dataset = SOATimeSeriesNoHeatMapsDataset(data_dir=data_path, seq_len=seq_len)
        datasets.append(dataset)
    # Concatenate all datasets
    combined_dataset = ConcatDataset(datasets)
    print(f'length of total data:{len(combined_dataset)}')
    return combined_dataset,len(combined_dataset)

def load_seperate_data_no_heatmaps(params,data_path_list):
    """
    Load multiple datasets without heatmaps, each dataset is treated separately.
    """
    
    seq_len = params['seq_len']
    datasets = []
    for data_path in data_path_list:
        dataset = SOATimeSeriesNoHeatMapsDataset(data_dir=data_path, seq_len=seq_len)
        datasets.append(dataset)
    print(f'number of dataset:{len(datasets)}')
    return datasets,len(datasets)

def load_multiple_view_dataset(params,data_path_list):
    seq_len = params['seq_len']
    datasets = []
    datasets_len = []
    for data_path in data_path_list:
        dataset = SOA_MultiView_TimeSeriesDataset(data_dir=data_path, seq_len=seq_len)
        datasets.append(dataset)
        datasets_len.append(len(dataset))
    # Concatenate all datasets
    combined_dataset = ConcatDataset(datasets)
    print(f'length of total data:{len(combined_dataset)}')
    return combined_dataset,datasets_len


def generate_saliency_maps_for_multiple_data(params,data_path_list,teacher_policy,check_data = False):
    seq_len = params['seq_len']
    data_path = params['data_path']

    for i,data_path in enumerate(data_path_list):
        data_generator = SOASTimeSeriesProcess(data_dir=data_path, seq_len=seq_len)
        soas_list = data_generator.get_saliency_data(teacher_policy)
        dir_path = os.path.dirname(data_path)

        match = re.search(r"swarm(\d+)", data_path)
        if match:
            host_id = int(match.group(1))
        else:
            raise Exception(f"host id is not found")
    
        soas_file_name = f'{dir_path}/swarm{host_id}_soa_saliency.pkl'
        
        with open(soas_file_name, 'wb') as f:
            pickle.dump(soas_list, f)
        print(f"save {soas_file_name} OK")
        if check_data:
            with open(soas_file_name, 'rb') as f:
                loaded_list = pickle.load(f)
            for soas in loaded_list:
                soas.print_data()
                cv2.imshow("image",soas.observation)
                cv2.waitKey()
    return


def make_multiview_data_loader(params):
    batch_size = params['batch_size']
    host_id = 'vswarm301'
    data_folder = 'hdf5_test'
    
    ## list0 :0:4   list1:0:8  list2:0:8
    ## Test: 0:3
    train_data_path_list0 = \
    [f'/media/datadisk/data_space/vi_data/{data_folder}/cyber_zoo0_{i}/{host_id}/state/{host_id}_soa.h5' \
     for i in range(0,1)]


    train_dataset,train_dataset_len = load_multiple_view_dataset(params,train_data_path_list0)

    train_data_loader = torch.utils.data.DataLoader( dataset = train_dataset,
                                                     batch_size = batch_size,
                                                     shuffle = False, ## keep it False
                                                     num_workers = 1 )

    n_data = len(train_dataset)
    return train_data_loader,n_data


def make_data_loader(params):
    batch_size = params['batch_size']
    host_id = 'vswarm301'

    data_folder = params['data_folder']
    print(f"Current data folder: {data_folder}")
    
    train_data_path_list0 = \
    [f'/media/datadisk/data_space/vi_data/{data_folder}/cyber_zoo_{i}/{host_id}/state/{host_id}_soa.h5' \
     for i in range(0,1)]

    train_data_path_list1 = \
    [f'/media/datadisk/data_space/vi_data/{data_folder}/lab_{i}/{host_id}/state/{host_id}_soa.h5' \
     for i in range(0,1)]

    train_data_path_list2 = \
    [f'/media/datadisk/data_space/vi_data/{data_folder}/test_zone_{i}/{host_id}/state/{host_id}_soa.h5' \
     for i in range(0,1)]
    
    test_data_path_list0 = \
    [f'/media/datadisk/data_space/vi_data/{data_folder}/software_zone_{i}/{host_id}/state/{host_id}_soa.h5' \
     for i in range(0,1)]
            

    if params['small_data']:
        train_data_path_list = train_data_path_list0
    else:
        train_data_path_list = train_data_path_list0 + train_data_path_list1 + train_data_path_list2
    

    test_data_path_list = test_data_path_list0
    
    if params['with_heatmaps']:
        print("load data WITH heatmaps")
        train_dataset, train_dataset_len = load_multiple_data_no_saliency(params,train_data_path_list)
        test_dataset,test_dataset_len = load_multiple_data_no_saliency(params,test_data_path_list)
    else:
        print("load data WITHOUT heatmaps")
        train_dataset, train_dataset_len = load_multiple_data_no_heatmaps(params,train_data_path_list)
        test_dataset,test_dataset_len = load_multiple_data_no_heatmaps(params,test_data_path_list)

    train_data_loader = torch.utils.data.DataLoader( dataset = train_dataset,
                                                     batch_size = batch_size,
                                                     shuffle = False, ## keep it False
                                                     num_workers = 1 )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    train_dataset_len = train_dataset_len
    test_dataset_len = test_dataset_len
    ###################################################################
    
    return train_data_loader, test_data_loader,train_dataset_len,test_dataset_len
