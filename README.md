# Robot Laser Tag 🤖

**[Update]**  This repository hosts the code for the paper "Learning Visuomotor Policy for Multi-Robot Laser Tag Game", which has been accepted by ICRA2026.

## ⚙️ Environment Setup

### Offline
The code is designed to run on **Ubuntu 20.04** with **ROS Noetic**.

To set up the Python environment for training and offline running, follow these steps:
```bash
conda env create -f environment.yaml
conda activate vi
```
### Onboard
The robot uses a Jetson Orin NX with JetPack 5.1.3.

## 🧠 Policy Training

### Teacher policy training
The state-based MARL teacher policy is implemented in the state_teacher/ directory. To train the teacher policy:
```bash
cd state_teacher
python train_teacher.py
```

### Student policy training
The vision-based student policy is located in the vision_student/ directory.       


## 💻 Offline Test
The offline test code is in the vision_student/ directory.       
To test it, first start a rosbag

```bash
rosbag play [PATH_TO_THE_BAG]
```
Offline test nodes are in the vision_student/offline folder. The Gazebo simulation environment is NOT included in this repo.     


## 🚀 Onboard Deployment
We also provide the onboard running code, which is in the onboard folder.        
🛠️ **Note:** This part is currently under active organization. A complete version of the code will be uploaded upon acceptance of the paper.

---
To run the code on a real robot, all vision-based models must be converted to [TensorRT](https://github.com/NVIDIA/TensorRT) in advance. For tutorials on TensorRT and Triton, please refer to Nvidia's official resources: [TensorRT](https://github.com/NVIDIA/TensorRT) and [Triton](https://github.com/triton-inference-server). These details will not be covered in this project.               
To start the module inference server,
```bash
cd onboard
./start_triton_server.sh
```

To start the depth estimation module,
```bash
python3 onboard_depth.py
```

To start the detection module,
```bash
python3 onboard_detection.py
```

To start the policy module,
```bash
python3 onboard_vi.py
```

## 🙏 Acknowledgment
We would like to thank the open-source projects that served as references for comparison and implementation in our work.
* [Multi-UAV-pursuit-evasion](https://github.com/thu-uav/Multi-UAV-pursuit-evasion)
* [Quad-swarm-rl](https://github.com/Zhehui-Huang/quad-swarm-rl)
* [vision-based-pursuit
](https://github.com/abajcsy/vision-based-pursuit)
* [Bearing-Angle Estimation](https://github.com/WindyLab/Bearing-angle)
* [STT Estimation](https://github.com/WindyLab/spatial-temporal-triangulation-for-bearing-only-cooperative-motion-estimation)
* [GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER)
* [Multi-particle-envs](https://github.com/openai/multiagent-particle-envs)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{li2024lasertag,
  title={Learning Visuomotor Policy for Multi-Robot Laser Tag Game},
  author={Kai Li, Shiyu Zhao},
  year={2026},
  booktitle={Proceedings of IEEE International Conference on Robotics and Automation (ICRA)}
}