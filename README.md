# VLA-from-FastVLM
This repository builds a new Vision-Language-Action (VLA) framework using Apple’s FastVLM as the vision-language backbone, with an Action Expert module for robot control.

folder tree 
```bash
VLA-from-FastVLM/
 ├─ original.py        # step1: FastVLMが動くことの確認用（固定）
 ├─ latent_expert.py   # step2: 潜在ベクトル→Action Expertに渡す
 ├─ train_expert.py    # step3: DataLoaderで行動模倣学習
 ├─ ros2_node.py       # step4: ROS2統合（実機動作用）
 └─ models/            # Action Expertの定義など
```




# ros2 (一旦無視)
```bash

#### 例: パラメータはros2の--ros-argsで上書き可
ros2 run your_pkg ros2_node.py --ros-args \
  -p ckpt:=runs/exp1/best.pt \
  -p joint_names:="[joint1,joint2,joint3,joint4,joint5,joint6,gripper,aux]" \
  -p scaler_json:=data/scaler.json \
  -p rate_hz:=20.0 -p delta_limit:=0.05 -p vel_limit:=0.5
```
