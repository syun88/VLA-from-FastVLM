#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2
import torch
from PIL import Image as PILImage
from transformers import AutoTokenizer, AutoModelForCausalLM

from train_expert import ActionExpertMLP, build_inputs, encode_latent

JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6","gripper","aux"]  # 例
DT = 0.05  # 20Hz
VEL_LIMIT = 0.5  # rad/s 相当の限界例
DELTA_LIMIT = 0.1  # 1ステップのΔ角制限

class VLAAgent(Node):
    def __init__(self, mid="apple/FastVLM-0.5B", revision="main", ckpt="runs/exp1/best.pt"):
        super().__init__("vla_agent")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 1)
        self.pub = self.create_publisher(JointTrajectory, "/joint_trajectory", 1)

        # load VLM (frozen)
        self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            mid,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            revision=revision,
        )
        for p in self.model.parameters(): p.requires_grad_(False)
        self.model.eval()

        # load Expert
        ck = torch.load(ckpt, map_location=self.model.device)
        self.expert = ActionExpertMLP(d_model=ck["d_model"], action_dim=ck["action_dim"]).to(self.model.device)
        self.expert.load_state_dict(ck["expert"]); self.expert.eval()

        # 状態（現在角度）はここでは持たず、Δのみ出す（実システムではサブスクして積分推奨）
        self.prompt = "Act for this observation."

        self.get_logger().info("VLAAgent ready.")

    @torch.no_grad()
    def on_image(self, msg: Image):
        # Image -> PIL
        cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)

        # latent
        ids, attn, px = build_inputs(self.tok, self.model, pil, self.prompt)
        z = encode_latent(self.model, ids, attn, px)  # [1, d_model]

        # action (Δq)
        act = self.expert(z).detach().cpu().numpy()[0]  # [-1,1]
        # 逆正規化する場合はここで inverse_scale() を呼ぶ

        # 安全クリップ
        import numpy as np
        act = np.clip(act, -1.0, 1.0)
        act = act * DELTA_LIMIT  # [-Δmax, Δmax] へ縮小

        # JointTrajectory publish（Δをそのまま目標速度っぽく解釈も可）
        jt = JointTrajectory()
        jt.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        # ここでは Δを「短時間で到達すべき目標角」扱い（本番は現在角に加算して目標角とするのが良い）
        pt.positions = [float(x) for x in act]  # 実運用は q_t + Δ を計算して positions に入れる
        pt.velocities = [float(max(min(x/DT, VEL_LIMIT), -VEL_LIMIT)) for x in act]
        pt.time_from_start = rclpy.duration.Duration(seconds=DT).to_msg()
        jt.points.append(pt)
        self.pub.publish(jt)

def main():
    rclpy.init()
    node = VLAAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
