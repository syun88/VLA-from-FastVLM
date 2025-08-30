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