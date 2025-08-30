#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2, threading, json, numpy as np, torch
from PIL import Image as PILImage
from transformers import AutoTokenizer, AutoModelForCausalLM

# 自作モジュール（あなたのリポジトリ構成に合わせて）
from train_expert import ActionExpertMLP, build_inputs, encode_latent

def load_scaler(path: str|None):
    if not path:
        return None
    with open(path, "r") as f:
        sc = json.load(f)
    return sc

def denorm(x: np.ndarray, sc: dict|None):
    if not sc: return x
    if "min" in sc and "max" in sc:
        mn, mx = np.array(sc["min"]), np.array(sc["max"])
        return (x + 1.0) * 0.5 * (mx - mn) + mn
    if "mean" in sc and "std" in sc:
        mu, st = np.array(sc["mean"]), np.array(sc["std"])
        return x * st + mu
    return x

class VLAAgent(Node):
    def __init__(self):
        super().__init__("vla_agent")
        # ===== パラメータ =====
        self.declare_parameter("model_id", "apple/FastVLM-0.5B")
        self.declare_parameter("revision", "main")
        self.declare_parameter("ckpt", "runs/exp1/best.pt")
        self.declare_parameter("prompt", "Act for this observation.")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("traj_topic", "/joint_trajectory")
        self.declare_parameter("joint_names", ["joint1","joint2","joint3","joint4","joint5","joint6","gripper","aux"])
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("delta_limit", 0.1)   # 1ステップのΔ最大（rad）
        self.declare_parameter("vel_limit", 0.5)     # 速度上限（rad/s相当）
        self.declare_parameter("scaler_json", "")    # 逆正規化に使用（任意）
        self.declare_parameter("clip_norm", 1.0)     # Expert出力のクリップ

        p = lambda k: self.get_parameter(k).get_parameter_value()
        self.mid        = p("model_id").string_value
        self.revision   = p("revision").string_value
        self.ckpt_path  = p("ckpt").string_value
        self.prompt     = p("prompt").string_value
        self.img_topic  = p("image_topic").string_value
        self.js_topic   = p("joint_states_topic").string_value
        self.traj_topic = p("traj_topic").string_value
        self.joint_names= list(p("joint_names").string_array_value)
        self.rate_hz    = p("rate_hz").double_value
        self.delta_limit= p("delta_limit").double_value
        self.vel_limit  = p("vel_limit").double_value
        self.scaler     = load_scaler(p("scaler_json").string_value)
        self.clip_norm  = p("clip_norm").double_value

        # ===== 購読/配布 =====
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, self.img_topic, self.on_image, 1)
        self.sub_js  = self.create_subscription(JointState, self.js_topic, self.on_joint_state, 1)
        self.pub_traj= self.create_publisher(JointTrajectory, self.traj_topic, 1)

        # ===== モデル読み込み（VLM凍結 + Expert）=====
        tok = AutoTokenizer.from_pretrained(self.mid, trust_remote_code=True, revision=self.revision)
        model = AutoModelForCausalLM.from_pretrained(
            self.mid,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # macOS/MPSはfp32推奨
            device_map="auto",
            trust_remote_code=True,
            revision=self.revision,
        )
        for p_ in model.parameters(): p_.requires_grad_(False)
        model.eval()
        self.tok = tok
        self.model = model

        ckpt = torch.load(self.ckpt_path, map_location=self.model.device)
        expert = ActionExpertMLP(d_model=ckpt["d_model"], action_dim=ckpt["action_dim"]).to(self.model.device)
        expert.load_state_dict(ckpt["expert"]); expert.eval()
        self.expert = expert

        # ===== 状態 =====
        self.lock = threading.Lock()
        self.last_pil = None
        self.q = np.zeros(len(self.joint_names), dtype=np.float32)  # 現在角度（購読できないときは0基準）
        self.have_q = False

        # Publishタイマー
        self.timer = self.create_timer(1.0/self.rate_hz, self.on_timer)

        self.get_logger().info("VLAAgent ready.")

    def on_image(self, msg: Image):
        cv = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)
        with self.lock:
            self.last_pil = pil

    def on_joint_state(self, msg: JointState):
        # joint_names の順序で q を更新（名前でマッピング）
        name2idx = {n:i for i,n in enumerate(self.joint_names)}
        with self.lock:
            for n, pos in zip(msg.name, msg.position):
                if n in name2idx:
                    self.q[name2idx[n]] = float(pos)
                    self.have_q = True

    @torch.no_grad()
    def predict_delta(self, pil: PILImage.Image) -> np.ndarray:
        ids, attn, px = build_inputs(self.tok, self.model, pil, self.prompt)
        z = encode_latent(self.model, ids, attn, px)          # [1, d_model]
        a = self.expert(z).detach().cpu().numpy()[0]          # [-1,1] 期待
        # クリップ保険
        if self.clip_norm > 0:
            a = np.clip(a, -self.clip_norm, self.clip_norm)
        # 逆正規化（学習時に正規化していた場合）
        a = denorm(a, self.scaler)
        # Δ上限（物理安全）
        a = np.clip(a, -self.delta_limit, self.delta_limit)
        return a

    def on_timer(self):
        with self.lock:
            pil = self.last_pil
            q   = self.q.copy()
            have_q = self.have_q
        if pil is None:
            return  # 画像まだ

        dq = self.predict_delta(pil)  # Δq
        # 現在角が取れていれば加算、無ければΔをそのまま小目標として送る
        if have_q:
            q_next = q + dq
        else:
            q_next = dq
            self.get_logger().warn_once("No /joint_states yet; publishing Δ as absolute small target.")

        # 速度推定（粗い近似）
        dt = 1.0 / max(1e-6, self.rate_hz)
        vel = np.clip(dq / dt, -self.vel_limit, self.vel_limit)

        jt = JointTrajectory()
        jt.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions  = [float(x) for x in q_next]
        pt.velocities = [float(v) for v in vel]
        pt.time_from_start = Duration(seconds=dt).to_msg()
        jt.points.append(pt)
        self.pub_traj.publish(jt)

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
