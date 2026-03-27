import numpy as np
import rosbag
import numpy as np
from cv_bridge import CvBridge
import cv2
import tf
import os
import pdb
import copy
from scipy.spatial.transform import Rotation as R
from infrastructure.img_concatenater import ImageConcatenater
from infrastructure.cam_para_manager import *
import pickle
from dav2.depth_anything_v2.dpt import DepthAnythingV2
import torch
import h5py
import time
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def make_dav2_model():
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'dav2_models/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = model.to(DEVICE).eval()
    return model

class Car():
    def __init__(self):
        super(Car, self).__init__()
        self.image_vec = []
        self.odom_vec = []
        self.gazebo_state = []
        self.image_time_vec = []
        self.host_id = -1
        self.raw_img_list_list = []
        self.aligned_observation_and_action = {}

    def add_raw_image(self,raw_img_list):
        """
        add raw image
        """
        self.raw_img_list_list.append(raw_img_list)

    def add_image(self,image):
        """
        add concatenated omnidirectional image
        """
        self.image_vec.append(image)

    def add_image_time(self,img_time):
        """
        add image time
        """
        self.image_time_vec.append(img_time)

    def add_odom(self,odom):
        """
        add odom, one image -> odom
        """
        self.odom_vec.append(odom)
        
    def add_gazebo_state(self,gazebo_state):
        """
        add gazebo_state, one image -> gazebo_state
        """
        self.gazebo_state.append(gazebo_state)

    def get_data_num(self):
        assert len(self.image_vec) == len(self.odom_vec)
        return len(self.image_vec)

class State_Observation_Action:
    def __init__(self,time_stamp,gazebo_state,omni_image,action):
        super(State_Observation_Action, self).__init__()
        self.state = gazebo_state
        self.observation = omni_image
        self.action = action
        self.time_stamp = time_stamp
    def print_data(self):
        print(f'time stamp {self.time_stamp} | action {self.action} | state {self.state}\n')

class Car_Action():
    def __init__(self):
        super(Car_Action, self).__init__()
        self.omni_imgs = {}  # Dictionary to store time as key and image as value
        self.depth_imgs = {}
        self.actions = {}
        self.gazebo_state = {}
        self.host_id = -1
        self.aligned_observation_and_action = {}
        self.state_observation_action = []

    def check_soa(self):
        for soa in self.state_observation_action:
            soa.print_data()
            cv2.imshow("image",soa.observation)
            cv2.waitKey()
        
    def dump_soa(self,state_data_path,check_data = False):
        state_file_name = f'{state_data_path}/{self.host_id}_soa.pkl'
        with open(state_file_name, 'wb') as f:
            pickle.dump(self.state_observation_action, f, protocol=pickle.HIGHEST_PROTOCOL)
        if check_data:
            start_time = time.time()
            with open(state_file_name, 'rb') as f:
                loaded_list = pickle.load(f)
            end_time = time.time()
            for soa in loaded_list:
                soa.print_data()
                print("read ")
                temp = soa.observation[:,:,:3].astype(np.uint8)
                cv2.imshow("image",temp)
                cv2.waitKey()
    
    def dump_hdf5_tensor_data(self, state_data_path, check_data=False):
        file_name = f'{state_data_path}/{self.host_id}_soa.h5'
        img_resize_list = []
        state_list = []
        action_list = []
        img_original_list = []
        depth_img_list = []
        time_stamp_list = []
        self.data_len = len(self.state_observation_action)
        
        transform_depth = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 90, img.width, img.height))),  # Crop out the top 90 rows
            transforms.Resize((224, 224)),  # Resize to (224, 224)
            transforms.ToTensor(),  # Convert to PyTorch Tensor, ensuring format (1, H, W)
            transforms.Lambda(lambda t: torch.log(t + 1e-6)),  # Log normalization, add small constant to avoid -inf
            transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min()) if (t.max() - t.min()) > 0 else t)  # Normalize linearly to [0, 1]
        ])
        
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 90, img.width, img.height))),
            # transforms.ColorJitter(
            #     brightness=0.4,   # Brightness adjustment range (0.6, 1.4)  (default: 1 ± 0.4)
            #     contrast=0.4,     # Contrast adjustment range (0.6, 1.4)
            #     saturation=0.4,   # Saturation adjustment range (0.6, 1.4)
            #     hue=0.1           # Hue adjustment range (-0.1, 0.1)
            # ),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_crop = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 90, img.width, img.height))),
            transforms.ToTensor()
        ])

        for soa in self.state_observation_action:
            ## soa.observation shape:180 x 1640 x 3
            ## after resize:224 x 224 x 3
            pil_image = Image.fromarray(soa.observation[:,:,0:3].astype(np.uint8))
            pil_image_depth = Image.fromarray(soa.observation[:,:,-1])

            ##############
            # im_to_show = image_rot.transpose((1,2,0))
            # plt.matshow(im_to_show)
            # plt.show()
            
            img_crop = transform_crop(pil_image)
            img_resize = transform(pil_image)
            depth_resize = transform_depth(pil_image_depth)
            image_resize = torch.FloatTensor(img_resize)
            depth_resize_tensor = torch.FloatTensor(depth_resize)
            state = torch.FloatTensor(soa.state)
            action = torch.FloatTensor(soa.action)
            image_original = torch.FloatTensor(img_crop)
            time_stamp = torch.FloatTensor(np.array(soa.time_stamp))
            
            #pdb.set_trace()
            ##====== check images ========================
            # im_to_show = img_crop.cpu().numpy().transpose((1,2,0))
            
            # dp_to_show = depth_resize_tensor.cpu().numpy().transpose((1,2,0))
            # if im_to_show.max() > 1:
            #     im_to_show = np.clip(im_to_show, 0, 255).astype(np.uint8)  # 限制范围到 [0, 255] 并转换为整数

            # plt.matshow(im_to_show)
            # plt.matshow(dp_to_show)
            # plt.show()
            ##============================================

            img_resize_list.append(image_resize)
            img_original_list.append(image_original)
            depth_img_list.append(depth_resize_tensor)
            state_list.append(state)
            action_list.append(action)
            time_stamp_list.append(time_stamp)
    
        img_resize_np = np.stack([t.numpy() for t in img_resize_list])
        img_original_np = np.stack([t.numpy() for t in img_original_list])
        depth_img_np = np.stack([t.numpy() for t in depth_img_list])
        state_np = np.stack([t.numpy() for t in state_list])
        action_np = np.stack([t.numpy() for t in action_list])
        time_stamp_np = np.stack([t.numpy() for t in time_stamp_list])

        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('img_resize', data=img_resize_np, compression='gzip')
            hf.create_dataset('img_original', data=img_original_np, compression='gzip')
            hf.create_dataset('depth_img', data=depth_img_np, compression='gzip')
            
            hf.create_dataset('states', data=state_np, compression='gzip')
            hf.create_dataset('actions', data=action_np, compression='gzip')
            hf.create_dataset('time_stamps', data=time_stamp_np, compression='gzip')

            hf.attrs['data_len'] = len(img_resize_list)
            hf.attrs['description'] = "State-Observation-Action dataset with processed images"
        print(f"save hdf5 {file_name} OK")
        if check_data:
            start_time = time.time()
            with h5py.File(file_name, 'r') as f:
                end_time = time.time()
                img_resize = f['img_resize'][:]
                img_original_ = f['img_original'][:]
                
                states_ = f['states'][:]
                actions_ = f['actions'][:]
                depth_ = f['depth_img'][:]
                time_stamp_ = f['time_stamps'][:]

                for id,img in enumerate(img_resize):
                    im_to_show = np.transpose(img,(1,2,0))
                    dp_to_show = np.transpose(depth_[id],(1,2,0))
                    im_or_to_show = np.transpose(img_original_[id],(1,2,0))
                    print(f"current time {time_stamp_[id]} state {states_[id]} action {actions_[id]}")

                    plt.matshow(im_to_show)
                    plt.matshow(im_or_to_show)
                    plt.matshow(dp_to_show)
                    plt.show()


    def dump_hdf5_individual_img_tensor_data(self,state_data_path,check_data = False):
        file_name = f'{state_data_path}/{self.host_id}_soa.h5'
        img_list = [[] for _ in range(4)]
        depth_list = [[] for _ in range(4)]

        state_list = []
        action_list = []
        time_stamp_list = []
        self.data_len = len(self.state_observation_action)
        transform_depth = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: torch.log(t + 1e-6)),
            transforms.Lambda(lambda t: (t - t.min()) / (t.max() - t.min()) if (t.max() - t.min()) > 0 else t)
        ])
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for soa in self.state_observation_action:
            ## soa.observation shape:180 x 1640 x 3
            ## after resize:224 x 224 x 3
            for k in range(4):
                pil_image_0 = Image.fromarray(soa.observation[k,:,:,0:3].astype(np.uint8))
                pil_image_depth_0 = Image.fromarray(soa.observation[k,:,:,-1])
                
                img_0 = transform(pil_image_0)
                depth_0 = transform_depth(pil_image_depth_0)
            
                img_0 = torch.FloatTensor(img_0)
                depth_0 = torch.FloatTensor(depth_0)
                img_list[k].append(img_0)
                depth_list[k].append(depth_0)

            state = torch.FloatTensor(soa.state)
            action = torch.FloatTensor(soa.action)
            time_stamp = torch.FloatTensor(np.array(soa.time_stamp))
            ##====== check images ========================
            # im_to_show = soa.observation[0,:,:,0:3] #img_0.cpu().numpy().transpose((1,2,0))
            
            # dp_to_show = depth_0.cpu().numpy().transpose((1,2,0))
            # if im_to_show.max() > 1:
            #     im_to_show = np.clip(im_to_show, 0, 255).astype(np.uint8)

            # plt.matshow(im_to_show)
            # plt.matshow(dp_to_show)
            # plt.show()
            ##============================================
            
            state_list.append(state)
            action_list.append(action)
            time_stamp_list.append(time_stamp)
    
        img_0_np = np.stack([t.numpy() for t in img_list[0]])
        depth_img_0_np = np.stack([t.numpy() for t in depth_list[0]])
        
        img_1_np = np.stack([t.numpy() for t in img_list[1]])
        depth_img_1_np = np.stack([t.numpy() for t in depth_list[1]])

        img_2_np = np.stack([t.numpy() for t in img_list[2]])
        depth_img_2_np = np.stack([t.numpy() for t in depth_list[2]])

        img_3_np = np.stack([t.numpy() for t in img_list[3]])
        depth_img_3_np = np.stack([t.numpy() for t in depth_list[3]])
        
        
        state_np = np.stack([t.numpy() for t in state_list])
        action_np = np.stack([t.numpy() for t in action_list])
        time_stamp_np = np.stack([t.numpy() for t in time_stamp_list])

        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('img_0', data=img_0_np, compression='gzip')
            hf.create_dataset('depth_0', data=depth_img_0_np, compression='gzip')
    
            hf.create_dataset('img_1', data=img_1_np, compression='gzip')
            hf.create_dataset('depth_1', data=depth_img_1_np, compression='gzip')
            
            hf.create_dataset('img_2', data=img_2_np, compression='gzip')
            hf.create_dataset('depth_2', data=depth_img_2_np, compression='gzip')
            
            hf.create_dataset('img_3', data=img_3_np, compression='gzip')
            hf.create_dataset('depth_3', data=depth_img_3_np, compression='gzip')

            hf.create_dataset('states', data=state_np, compression='gzip')
            hf.create_dataset('actions', data=action_np, compression='gzip')
            hf.create_dataset('time_stamps', data=time_stamp_np, compression='gzip')

            hf.attrs['data_len'] = len(img_0_np)
            hf.attrs['description'] = "State-Observation-Action dataset with processed images"
        
        print(f"save hdf5 {file_name} OK")
        if check_data:
            start_time = time.time()
            with h5py.File(file_name, 'r') as f:
                end_time = time.time()
                img_list0 = f['img_0'][:]
                img_list1 = f['img_1'][:]
                img_list2 = f['img_2'][:]
                img_list3 = f['img_3'][:]
                depth_list_0 = f['depth_0'][:]
                depth_list_1 = f['depth_1'][:]
                depth_list_2 = f['depth_2'][:]
                depth_list_3 = f['depth_3'][:]

                states_ = f['states'][:]
                actions_ = f['actions'][:]
                time_stamp_ = f['time_stamps'][:]
                print(f"Total time: {end_time - start_time:.4f} s")
                
                for id in range(len(img_list0)):
                    im_to_show0 = np.transpose(img_list0[id],(1,2,0))
                    dp_to_show0 = np.transpose(depth_list_0[id],(1,2,0))
                    print(f"current time {time_stamp_[id]} state {states_[id]} action {actions_[id]}")

                    im_to_show1 = np.transpose(img_list1[id],(1,2,0))
                    dp_to_show1 = np.transpose(depth_list_1[id],(1,2,0))
                    
                    im_to_show2 = np.transpose(img_list2[id],(1,2,0))
                    dp_to_show2 = np.transpose(depth_list_2[id],(1,2,0))
                    
                    im_to_show3 = np.transpose(img_list3[id],(1,2,0))
                    dp_to_show3 = np.transpose(depth_list_3[id],(1,2,0))
                    
                    plt.matshow(im_to_show0)
                    plt.matshow(dp_to_show0)

                    # plt.matshow(im_to_show1)
                    # plt.matshow(dp_to_show1)
                    
                    # plt.matshow(im_to_show2)
                    # plt.matshow(dp_to_show2)
                    
                    # plt.matshow(im_to_show3)
                    # plt.matshow(dp_to_show3)
                    plt.show()


    def dump_hdf5_soa(self,state_data_path,check_data = False):
        fiel_name = f'{state_data_path}/{self.host_id}_soa.h5'
        with h5py.File(fiel_name, 'w') as f:
            states = np.array([soa.state for soa in self.state_observation_action])
            observations = np.array([soa.observation for soa in self.state_observation_action])
            actions = np.array([soa.action for soa in self.state_observation_action])
            timestamps = np.array([soa.time_stamp for soa in self.state_observation_action])
            
            f.create_dataset('states', data=states)
            f.create_dataset('observations', data=observations)
            f.create_dataset('actions', data=actions)
            f.create_dataset('timestamps', data=timestamps)
        
        print(f"hdf5 data file {fiel_name} save ok!!")
        if check_data:
            start_time = time.time()
            with h5py.File(fiel_name, 'r') as f:
                end_time = time.time()
                print(f"Total time: {end_time - start_time:.4f} s")
                states = f['states'][:]
                observations = f['observations'][:]
                actions = f['actions'][:]
                timestamps = f['timestamps'][:]

                reconstructed_list = [
                    State_Observation_Action(ts, s, o, a)
                    for ts, s, o, a in zip(timestamps, states, observations, actions)]
                for soa in reconstructed_list:
                    soa.print_data()
                   # pdb.set_trace()
                    temp = soa.observation[:,:,:3].astype(np.uint8)
                    cv2.imshow("image",temp)
                    cv2.waitKey()
        
        
    def add_image(self, t, image):
        """
        Add concatenated omnidirectional image.
        :param t: Time (key for the dictionary)
        :param image: Image (value for the dictionary)
        """
        self.omni_imgs[t] = image

    def add_action(self, t, action):
        """
        Add action corresponding to a specific time.
        :param t: Time (key for the dictionary)
        :param action: Action (value for the dictionary)
        """
        self.actions[t] = action

    def add_gazebo_state(self, t, state):
        """
        Add gazebo state corresponding to a specific time.
        :param t: Time (key for the dictionary)
        :param state: Gazebo state (value for the dictionary)
        """
        self.gazebo_state[t] = state

    def convert_time(self,ros_time):
        t = ros_time.secs * 1e09 + ros_time.nsecs
        return t

    def get_closest_value(self, dict_query, t):
        """
        Find the value in the dictionary with the closest timestamp to t.
        :param dict_query: Dictionary with timestamps as keys and corresponding states as values.
        :param t: Target timestamp.
        :return: The value corresponding to the closest timestamp.
        """
        closest_key = None
        closest_diff = float('inf')

        for key in dict_query.keys():
            #key_sec = key.to_sec()  # Convert ROS time to seconds
            #pdb.set_trace()
            diff = abs(key - t)
            if diff < closest_diff:
                closest_diff = diff
                closest_key = key

        return (closest_key,dict_query[closest_key]) if closest_key is not None else (None,None)

    def key_query_value(self,dict_keys,dict_query):
        value_vec = []
        for t in dict_keys.keys():
            closest_time, value = self.get_closest_value(dict_query,t)
            #print(closest_time - t,type(value))
            value_vec.append(value)
        return value_vec

    def merge_all_values_action_key(self,dict_keys_action,state_vec,image_vec):
        assert len(dict_keys_action.keys()) == len(state_vec)
        assert len(dict_keys_action.keys()) == len(image_vec)
        soa_vec = []  ## state observation action vector
        for i,time_stamp in enumerate(dict_keys_action.keys()):
            soa = State_Observation_Action( time_stamp = time_stamp,
                                            action = dict_keys_action[time_stamp],
                                            gazebo_state = state_vec[i],
                                            omni_image = image_vec[i] )
            soa_vec.append(soa)
        return soa_vec

    def merge_all_values_img_key(self,dict_keys_img,state_vec,action_vec):
        assert len(dict_keys_img.keys()) == len(state_vec)
        assert len(dict_keys_img.keys()) == len(action_vec)
        soa_vec = []  ## state observation action vector
        for i,time_stamp in enumerate(dict_keys_img.keys()):
            soa = State_Observation_Action( time_stamp = time_stamp,
                                            action = action_vec[i],
                                            gazebo_state = state_vec[i],
                                            omni_image = dict_keys_img[time_stamp] )
            soa_vec.append(soa)
        return soa_vec
    
    def merge_state_ob_action(self):
        assert len(self.gazebo_state) > 0
        assert len(self.omni_imgs) > 0
        assert len(self.actions) > 0

        ## assume states have the highest frequency
        if len(self.omni_imgs) > len(self.actions):
            image_vec = self.key_query_value ( dict_keys = self.actions, dict_query = self.omni_imgs)
            state_vec = self.key_query_value ( dict_keys = self.actions, dict_query = self.gazebo_state)
            self.state_observation_action = \
                self.merge_all_values_action_key(dict_keys_action = self.actions,
                                                 state_vec = state_vec, 
                                                 image_vec = image_vec)
        else:
            ## TODO: what if action frequency is higher?
            action_vec = self.key_query_value(dict_keys = self.omni_imgs, dict_query = self.actions)
            state_vec = self.key_query_value ( dict_keys = self.omni_imgs, dict_query = self.gazebo_state)
            self.state_observation_action = \
                self.merge_all_values_img_key(dict_keys_img = self.omni_imgs,
                                              state_vec = state_vec, 
                                              action_vec = action_vec)
            
        
        return self.state_observation_action

def getRandomIndex(n, x):
    index = np.random.choice(np.arange(n), size=x, replace=False)
    return index

def generate_random_training_idx(total_data_num, training_num):
    ret = getRandomIndex(total_data_num,training_num)
    ret.sort()
    return ret

def generate_even_training_idx(total_data_num, interval,start_from = 0):
    ret = range(start_from, total_data_num, interval) 
    return ret

def gaussian_k(x0,y0,sigma, width, height):
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
 
def generate_hm(height, width ,landmarks,s=3):
    Nlandmarks = landmarks.shape[0]
    hm = np.zeros((height, width, Nlandmarks), dtype = np.float32)
    for i in range(Nlandmarks):
        if not np.array_equal(landmarks[i], [-1,-1]):
            
            hm[:,:,i] = gaussian_k(landmarks[i][0],
                                    landmarks[i][1],
                                    s,width,height)
        else:
            hm[:,:,i] = np.zeros((height,width))
    return hm

def convert_time(ros_time):
    t = ros_time.secs * 1e09 + ros_time.nsecs
    return t

def get_gazebo_state(gazebo_state_vec, t):
    def compare_timestamp(gazebo_state):
        return (gazebo_state[0] - t).to_sec()
    left = 0
    right = len(gazebo_state_vec) - 1
    closest_state = None
    while left <= right:
        mid = (left + right) // 2
        diff = compare_timestamp(gazebo_state_vec[mid])
        if diff == 0:
            return gazebo_state_vec[mid]
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
        if closest_state is None or abs(diff) < abs(compare_timestamp(closest_state)):
            closest_state = gazebo_state_vec[mid]
    return closest_state

def get_odom(odom_vec, t):
    def compare_timestamp(odom):
        return (odom.header.stamp - t).to_sec()
    left = 0
    right = len(odom_vec) - 1
    closest_odom = None
    while left <= right:
        mid = (left + right) // 2
        diff = compare_timestamp(odom_vec[mid])
        if diff == 0:
            return odom_vec[mid]
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
        if closest_odom is None or abs(diff) < abs(compare_timestamp(closest_odom)):
            closest_odom = odom_vec[mid]
    return closest_odom

def get_img_time(img_time_vec, t):
    def compare_timestamp(img_time):
        return (img_time - t).to_sec()
    left = 0
    right = len(img_time_vec) - 1
    closest_img_t = None
    while left <= right:
        mid = (left + right) // 2
        diff = compare_timestamp(img_time_vec[mid])
        if diff == 0:
            return img_time_vec[mid]
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
        if closest_img_t is None or abs(diff) < abs(compare_timestamp(closest_img_t)):
            closest_img_t = img_time_vec[mid]
    return closest_img_t


def read_odom(bag_dir,car_id):
    bag = rosbag.Bag(bag_dir)
    topic_info = bag.get_type_and_topic_info()
    print(topic_info)
    print(bag_dir)
    odom_car = []
    gazebostate_car = []
    ######## use robot odom  ##############
    for topic, msg, t in bag.read_messages(topics=['/robot/odom']):
        #print(f"Received message on topic {topic} at time {t}")
        #print("odom time diff:",convert_time(t - msg.header.stamp)/1e06)
        #print(msg.header.stamp)
        msg_offset = msg
        msg_offset.header.stamp = t
        odom_car.append(msg_offset)

   # pdb.set_trace()
    ######## use gazebo state  #############
    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
        print(t)
        pose_id = msg.name.index(f'swarm{car_id}')
        gazebostate_car.append((t,msg.pose[pose_id]))

def read_one_car(bag_dir,car_id,check_image=False):
    bag = rosbag.Bag(bag_dir)
    topic_info = bag.get_type_and_topic_info()
    print(topic_info)
    print(bag_dir)

    img_0 = []
    img_1 = []
    img_2 = []
    img_3 = []
    img_time = []

    ## read images
    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[ '/cam0/image_raw',
                                                    '/cam1/image_raw',
                                                    '/cam2/image_raw',
                                                    '/cam3/image_raw'] ):
        # print(f"Received message on topic {topic} at time {t}")
        # print(f"Received message on topic {topic} at time {msg.header}")
        #pdb.set_trace()
        cnt = cnt + 1
        if cnt < 200:
            continue
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        if 'cam0' in topic:
            img_0.append(cv_image)
            #print(f"Received message on topic {topic} at time {t}")
            #print("image time diff:",convert_time(t - msg.header.stamp)/1e06)
            img_time.append(t)  ## use camera 0 as baseline
        elif 'cam1' in topic:
            img_1.append(cv_image)
        elif 'cam2' in topic:
            img_2.append(cv_image)
        elif 'cam3' in topic:
            img_3.append(cv_image)
    
    #total_img = topic_info.topics['/cam0/image_raw'].message_count
    ## read odom
    assert (len(img_0) == len(img_1), "total_img == len(img_0)") and \
           (len(img_1) == len(img_2), "total_img == len(img_1)") and \
           (len(img_2) == len(img_3), "total_img == len(img_2)") 

    total_img = np.min([len(img_0),len(img_1),len(img_2),len(img_3)])
    print(f"total_img {total_img}")
    
    def Reverse(lst):
        return [ele for ele in reversed(lst)]
    
    car = Car()
    car.host_id = car_id
    odom_car_0 = []
    gazebostate_car_0 = []
    
    ######## use robot odom
    for topic, msg, t in bag.read_messages(topics=['/robot/odom']):
        #print(f"Received message on topic {topic} at time {t}")
        #print("odom time diff:",convert_time(t - msg.header.stamp)/1e06)
        #print(msg.header.stamp)
        msg_offset = msg
        msg_offset.header.stamp = t
        odom_car_0.append(msg_offset)
        car.add_odom(msg_offset)
   # pdb.set_trace()
    ######## use gazebo state
    for topic, msg, t in bag.read_messages(topics=['/gazebo/model_states']):
        #print(t)
        pose_id = msg.name.index(f'swarm{car_id}')
        gazebostate_car_0.append((t,msg.pose[pose_id]))
        car.add_gazebo_state((t,msg.pose[pose_id]))

    ic = ImageConcatenater()
    cutted_img_height = ic.get_cutted_img_height()
    start_id,_ = ic.compute_crop_size()
    original_img_width = ic.get_original_img_width()

    for i in range(total_img):
        img0 = img_0[i][-cutted_img_height:, start_id:original_img_width-start_id : ]
        img1 = img_1[i][-cutted_img_height:, start_id:original_img_width-start_id : ]
        img2 = img_2[i][-cutted_img_height:, start_id:original_img_width-start_id : ]
        img3 = img_3[i][-cutted_img_height:, start_id:original_img_width-start_id : ]

        img_list = [img0,img1,img2,img3]
        result = cv2.hconcat((img_list))
        car.add_image(result)
        car.add_image_time(img_time[i])
        car.add_raw_image(img_list)
        ####### Check single agent time stamps ###########
        odom = get_odom(odom_car_0,img_time[i])
        gazebo_state = get_gazebo_state(gazebostate_car_0,img_time[i])
       # pdb.set_trace()
        #print("odom and image time diff (ms):",convert_time(img_time[i] - odom.header.stamp)/ 1e06)
        #print("gazebo state and image time diff (ms):",convert_time(img_time[i] - gazebo_state[0])/ 1e06)
        
        ##################################################
        if check_image:
            cv2.imshow("omni_image",result)
            ##plt.matshow(result)
            #plt.show()
            cv2.waitKey()
    return car

def read_real_world_bag(bag_dir,car_id,check_image=False):
    bag = rosbag.Bag(bag_dir)
    topic_info = bag.get_type_and_topic_info()
    print(topic_info)

    img_0 = []
    img_1 = []
    img_2 = []
    img_3 = []
    img_time = []

    ## read images
    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[ '/cam0/image_raw',
                                                    '/cam1/image_raw',
                                                    '/cam2/image_raw',
                                                    '/cam3/image_raw'] ):
        #print(f"Received message on topic {topic} at time {t}")
        #print(f"Received message on topic {topic} at time {msg.header}")
        
        cnt = cnt + 1
        cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        if 'cam0' in topic:
            img_0.append(cv_image)
            #print(f"Received message on topic {topic} at time {t}")
           # print(f"image time diff:{convert_time(t - msg.header.stamp)/1e06} ms")
            img_time.append(msg.header.stamp)  ## use camera 0 as baseline
        elif 'cam1' in topic:
            img_1.append(cv_image)
        elif 'cam2' in topic:
            img_2.append(cv_image)
        elif 'cam3' in topic:
            img_3.append(cv_image)
    #total_img = topic_info.topics['/cam0/image_raw'].message_count
    ## read odom
    assert (len(img_0) == len(img_1), "total_img == len(img_0)") and \
           (len(img_1) == len(img_2), "total_img == len(img_1)") and \
           (len(img_2) == len(img_3), "total_img == len(img_2)") 

    total_img = np.min([len(img_0),len(img_1),len(img_2),len(img_3)])
    print(f"total_img {total_img}")
   # pdb.set_trace()
    
    def Reverse(lst):
        return [ele for ele in reversed(lst)]
    
    car = Car()
    car.host_id = car_id
    odom_car_0 = []
    print(f"total_img {total_img}")

    ######## use vicon  ################
    for topic, msg, t in bag.read_messages(topics=[f'/vicon/VSWARM{car_id}/VSWARM{car_id}']):
        odom_car_0.append(msg)
        car.add_odom(msg)

    ########################
    ## image undistortion ##
    ########################

    ic = ImageConcatenater()
    cutted_img_height = ic.get_cutted_img_height()
    start_id,_ = ic.compute_crop_size()
    original_img_width = ic.get_original_img_width()

    cam_para = Car_param(car_id,'cali_file')
    for i in range(total_img):
        map1,map2 = get_map_from_intrinsics(0,cam_para)
        img_0[i] = cv2.remap(img_0[i], map1, map2, cv2.INTER_LINEAR)
        map1,map2 = get_map_from_intrinsics(1,cam_para)
        img_1[i] = cv2.remap(img_1[i], map1, map2, cv2.INTER_LINEAR)
        map1,map2 = get_map_from_intrinsics(2,cam_para)
        img_2[i] = cv2.remap(img_2[i], map1, map2, cv2.INTER_LINEAR)
        map1,map2 = get_map_from_intrinsics(3,cam_para)
        img_3[i] = cv2.remap(img_3[i], map1, map2, cv2.INTER_LINEAR)

        img0 = img_0[i][90:img_0[i].shape[0]-90, : ]
        img1 = img_1[i][90:img_0[i].shape[0]-90, : ]
        img2 = img_2[i][90:img_0[i].shape[0]-90, : ]
        img3 = img_3[i][90:img_0[i].shape[0]-90, : ]
        #print(img0.shape)
        img_list = [img0,img1,img2,img3]
        result = cv2.hconcat((img_list))
        car.add_image(result)
        car.add_image_time(img_time[i])
        car.add_raw_image(img_list)
        ####### Check single agent time stamps ###########
        #odom = get_odom(odom_car_0,img_time[i])
        #pdb.set_trace()
        #print("odom and image time diff (ms):",convert_time(img_time[i] - odom.header.stamp)/ 1e06)
        
        ##################################################
        if check_image:
            cv2.imshow("omni_image",result)
            cv2.waitKey(1)
    return car

def convert_3D_pose_to_2D(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    return x, y, yaw

def convert_pose_msg_to_mat(pose):
    translation = [pose.position.x, pose.position.y, pose.position.z]
    rotation = [pose.orientation.x, pose.orientation.y, pose.orientation.z ,pose.orientation.w]
    rotation_ = R.from_quat(rotation)
    translation_matrix = tf.transformations.translation_matrix(translation)
    translation_matrix[0:3,0:3] = rotation_.as_matrix()
    return translation_matrix

def convert_Transform_to_mat(pose):
    translation = [pose.translation.x, pose.translation.y, pose.translation.z]
    rotation = [pose.rotation.x, pose.rotation.y, pose.rotation.z ,pose.rotation.w]
    rotation_ = R.from_quat(rotation)
    translation_matrix = tf.transformations.translation_matrix(translation)
    translation_matrix[0:3,0:3] = rotation_.as_matrix()
    return translation_matrix


def project_onto_image(p_in_camera):
    ic = ImageConcatenater()
    focal_len = ic.get_cropped_focal()
    cx = ic.get_cropped_cx()
    cy = ic.get_cropped_cy()
   # pdb.set_trace()
    u = focal_len * p_in_camera[0] / p_in_camera[2] + cx
    v = focal_len * p_in_camera[1] / p_in_camera[2] + cy
    return (u,v)

def project_onto_image2(p_in_camera,cam_param):
    cam_K = cam_param.get_undistort_K()
    focal_len = cam_K[0][0]
    cx = cam_K[0][2]
    cy = cam_K[1][2]
    #print("focal_len,cx,cy:",focal_len,cx,cy)
    #pdb.set_trace()
    u = focal_len * p_in_camera[0] / p_in_camera[2] + cx
    v = focal_len * p_in_camera[1] / p_in_camera[2] + cy
    return (u,v)

def project_cam(img_h,single_img_w,p_in_body,cam_param):
    camera_body_3 = get_cam_body_extrinsics(3,cam_param)
    p_in_camera_3 = camera_body_3 @ p_in_body

    camera_body_2 = get_cam_body_extrinsics(2,cam_param)
    p_in_camera_2 = camera_body_2 @ p_in_body

    camera_body_1 =  get_cam_body_extrinsics(1,cam_param)
    p_in_camera_1 = camera_body_1 @ p_in_body

    camera_body_0 = get_cam_body_extrinsics(0,cam_param)
    p_in_camera_0 = camera_body_0 @ p_in_body

    p_candidates = [p_in_camera_0,p_in_camera_1,p_in_camera_2,p_in_camera_3]
    #print(p_candidates)
   # pdb.set_trace()
    res = []
    for id,p in enumerate(p_candidates):
        if p[2] < 0:
            continue
        #print(p)
        #u,v = project_onto_image2(p,cam_param)
        u,v = project_onto_image(p)
        #print((u,v,id),p)
        
        #pdb.set_trace()
        if u >= 0 and u < single_img_w and v >=0 and v < img_h:
            u = u + id * single_img_w
            res.append((u,v,id))
        elif abs(u - single_img_w) < 2:
            u = single_img_w * (id + 1)
            res.append((u,v,id))
            
    if len(res) != 1 and len(res) != 0:
        #print("len(res):",len(res))
        x = 0
        y = 0
        for k in range(len(res)):
            x = x + res[k][0]
            y = y + res[k][1]
        x = x / len(res)
        y = y / len(res)
        
        #pdb.set_trace()
        res = [(x,y,res[0][2])]
    #assert len(res) == 1 or len(res) == 0
    return res

def generate_key_point_labels(img_h,img_w,relative_pose):
    """
    generate semantic key point labels for ONE target
    """
    pass

def generate_v_attention_labels(img_h,img_w,relative_pose,cam_param):
    """
    generate visual attention labels for ONE target
    """
    p_in_body =  relative_pose @ np.array([0,0,0,1]).T
    single_img_wid = ImageConcatenater().compute_cropped_width()
    p = project_cam(img_h,single_img_wid,p_in_body,cam_param)
    p_np = np.array(p)
    body_depth = np.linalg.norm(p_in_body[0:2])
    #print(p_np)
    hm = generate_hm(img_h,img_w,p_np,s = int(20 / body_depth))
    return hm

ic = ImageConcatenater()
def generate_motion_compensation_img(img_current,img_previsous,relative_pose,camera_body,show=False):
    """
    generate motion compensated image.[img_h,img_w,2] -> (gray,motion_flow)
    relative_pose: relative pose of body frame
    camera_body: extrinsics between camera and body
    """
    ex = camera_body @ relative_pose @ np.linalg.inv(camera_body)
    focal_len = ic.get_cropped_focal()
    cx = ic.get_cropped_cx()
    cy = ic.get_cropped_cy()
    K_mat = np.array([[focal_len,     0,      cx], 
                      [0,        focal_len,   cy],
                      [0,             0,       1]])

    H = K_mat @ ex[0:3,0:3] @ np.linalg.inv(K_mat)  ## current -> last
    w = img_current.shape[1]
    h = img_current.shape[0]
    warped_current = cv2.warpPerspective(img_current, H, (w, h))
    
    #### compute optical flow ##########################
    hsv = np.zeros_like(img_previsous)
    if warped_current.shape[2] == 3:
        warped_current_gray = cv2.cvtColor(warped_current, cv2.COLOR_BGR2GRAY)
    if img_current.shape[2] == 3:
        current_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    if img_previsous.shape[2] == 3:
        pre_gray = cv2.cvtColor(img_previsous, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(pre_gray,warped_current_gray, None, **fb_params)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    motion_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #motion_flow = cv2.copyTo(motion_flow,bbox_mask)
    motion_flow = cv2.cvtColor(motion_flow, cv2.COLOR_BGR2GRAY)
    #motion_flow = np.expand_dims(motion_flow, axis=2)

    if show:
        overlapping1 = cv2.addWeighted(img_current, 0.5, img_previsous, 0.5, 0)
        overlapping2 = cv2.addWeighted(warped_current, 0.5, img_previsous, 0.5, 0)
        overlap_list = [overlapping1,overlapping2]
        result = cv2.hconcat((overlap_list))
        cv2.imshow("result",result)
        cv2.imshow("motion_flow",motion_flow)
        cv2.imshow("current_gray",current_gray)
        cv2.waitKey(1)
        
    return current_gray, motion_flow

def is_homogeneous(lst):
    if not lst:
        return True
    
    first_shape = np.shape(lst[0])
    for item in lst:
        if np.shape(item) != first_shape:
            return False
    return True

def generate_bbox_labels(img_h,img_w,relative_pose,cam_param):
    """
    generate bounding box labels for ONE target
    """
    robot_radius = 0.15
    points_on_car = []
    for theta in np.linspace(0,2*np.pi,20):
        p_on_car_circle = [robot_radius * np.cos(theta),
                           robot_radius * np.sin(theta),
                           0.0,  1 ]
        points_on_car.append(p_on_car_circle)
    points_on_car.append([0,       0,      0.1,           1])
    points_on_car = np.array(points_on_car)

    p_in_body_all =  relative_pose @ points_on_car.T
    bbox_pro_pts = []
    single_img_wid = ImageConcatenater().compute_cropped_width()
    
    for i in range(p_in_body_all.shape[1]):
        p_ = project_cam(img_h,single_img_wid,p_in_body_all[:,i],cam_param)
        #print()
        if len(p_) > 0:
            bbox_pro_pts.append(p_)
    print(np.array(bbox_pro_pts).astype(np.float32))
    #pdb.set_trace()
    p_np = None
    
    if len(bbox_pro_pts) > 0 and is_homogeneous(bbox_pro_pts):
        p_np = np.array(bbox_pro_pts).squeeze(1)
    return p_np

def generate_bbox_mask(pts_list_np, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for pts in pts_list_np:
        x_min, y_min, x_max, y_max = pts[1], pts[2], pts[3], pts[4]
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
    return mask

fb_params = dict(pyr_scale = 0.5,
                 levels = 3,
                 winsize = 10,
                 iterations = 1,
                 poly_n = 5,
                 poly_sigma = 3,
                 flags = 0)

def check_and_save_img(img,data_path,img_cnt, mono=True, show=False):
    """
    check and save image
    """
    if mono:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{data_path}/{img_cnt:05d}.png', img)
    if show:
        cv2.imshow('gray image',img)
        cv2.waitKey()


def check_and_save_img_motion(pts_list_np,cated_img,data_path,img_cnt,show=False):
    """
    check and save image, save optical flow
    """
    ################# gernerate mask for motion flow ################
    bbox_mask = generate_bbox_mask(pts_list_np,cated_img.shape)
    # bbox_mask_flip = cv2.flip(bbox_mask,-1)

    #################################################################
    motion_flow = cv2.copyTo(cated_img[:,:,1],bbox_mask)
    cated_img[:,:,1] = copy.deepcopy(motion_flow)
    #print(data_path,cated_img.shape)
    np.save(f'{data_path}/{img_cnt:05d}.npy',cated_img)
    # temp = np.load(f'{data_path}/{img_cnt:05d}.npy')
    # print("save shape:",temp.shape)

    if show:
        cv2.imshow('gray image',cated_img[:,:,0])
        cv2.imshow('motion image masked',cated_img[:,:,1])
        cv2.waitKey()

def check_and_save_bbox(img,pts_vec,data_path,img_cnt,show=False):
    """
    check and save bbox label
    """
    if len(pts_vec) == 0:
        print(f"no target in this image {img_cnt}")
        return
    
    import copy
    img_check = copy.deepcopy(img)
    width = img.shape[1]
    height = img.shape[0]
    
    pts_list = []
    pts_list_scale = []
    
    for id,pts_on_one_car in enumerate(pts_vec):
        same_cam = np.all(pts_on_one_car[:, 2] == pts_on_one_car[0, 2])
        if same_cam == False and \
        (np.all(np.logical_or(pts_on_one_car[:, 2] == 3, pts_on_one_car[:, 2] == 0))):
            continue
       # print(pts_on_one_car[:, 2])
       # print("same_cam:",same_cam)
        points = pts_on_one_car[:,0:2].astype(np.int64)
        #print("pts_on_one_car\n",pts_on_one_car)
        offset = 0
        p_x_max = np.max(points[:,0]) + offset
        p_x_min = np.min(points[:,0]) - offset
        p_y_max = np.max(points[:,1]) + offset
        p_y_min = np.min(points[:,1]) - offset
        ####  original format ####
        pts_list.append([id,p_x_min,p_y_min,p_x_max,p_y_max])
        ####  yolo format ####
        pts_list_scale.append([0,
                         (p_x_min + p_x_max) / 2 / width,   ## x_center
                         (p_y_min + p_y_max) / 2 / height,  ## y_center
                         (p_x_max - p_x_min ) / width,
                         (p_y_max - p_y_min ) / height]) 

        if show:
            cv2.rectangle(img_check,(p_x_min,p_y_min),(p_x_max,p_y_max),(0,255,0))
    pts_list_np = np.array(pts_list)
    pts_list_scale_np = np.array(pts_list_scale)
    #np.save(f'{data_path}/{img_cnt:05d}.npy',pts_list_np)
    #temp = np.load(f'{data_path}/{img_cnt:05d}.npy')
    if len(pts_list_scale) == 0:
        return
    fmt = ['%d'] + ['%.6f'] * (pts_list_scale_np.shape[1] - 1)
    np.savetxt(f'{data_path}/{img_cnt:05d}.txt', pts_list_scale_np, fmt=fmt) 
    #pdb.set_trace()

    if show:
        img_flip = cv2.flip(img_check,-1)
        cv2.imshow("omni_image_bbx_check",img_flip)
        cv2.waitKey()
    return pts_list_np

def check_and_save_vattention(img,hm_total_v_att,v_att_label_path,img_cnt,show=False):
    """
    check and save visual attention data
    """
    print("hm_total shape:",hm_total_v_att.shape)
    valid_target_num = hm_total_v_att.shape[2]
    print("valid_target_num:",valid_target_num)
    if show:
        hm_total_v_att = ((255* hm_total_v_att)).astype(dtype=np.uint8)
        if valid_target_num != 0:
            heat_map_total = np.array(hm_total_v_att)
            heat_map_total = cv2.cvtColor(heat_map_total, cv2.COLOR_BGR2RGB)
        else:
            heat_map_total = np.zeros((hm_total_v_att.shape[0],hm_total_v_att.shape[1],1),dtype=np.uint8)
            
        heat_map_trans = cv2.applyColorMap(heat_map_total, cv2.COLORMAP_JET)
        overlapping = cv2.addWeighted(img, 0.5, heat_map_trans, 0.5, 0)
        overlapping_flip = cv2.flip(overlapping,-1)
        cv2.imshow("overlap",overlapping_flip)
        cv2.waitKey()
        
    #hm_total_v_att = cv2.flip(hm_total_v_att,-1)
    print(f"v_att_label_path {v_att_label_path}")
    np.save(f'{v_att_label_path}/{img_cnt:05d}.npy',hm_total_v_att)
    # temp = np.load(f'{v_att_label_path}/{img_cnt:05d}.npy')
    # print(temp.shape)

def preprocess(img,scale):
    """
    preprocess image for v attention inference
    """
    img = cv2.resize(img,dsize=None,fx=scale,fy=scale,interpolation = cv2.INTER_NEAREST)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # if (img > 1).any():
    #     img = img / 255.0
    img = img / 255
    return img

def convert_np(img_tensor):
    """
    convert v attention prediction tensor to numpy array
    """
    return img_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) / (max_val - min_val)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    return normalized_image

def normalize_array(arr):
    if len(arr) == 0:
        return arr
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = ((np.array(arr) - min_val) / (max_val - min_val)) * 1
    return normalized_arr
