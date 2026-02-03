#!/usr/bin/env python3
"""
LNN Inference Node for ROS2 Humble
Trained LNN model ကို သုံးပြီး Image -> cmd_vel prediction

Usage:
    ros2 run <package_name> lnn_inference_node.py
    
Or standalone:
    python3 lnn_inference_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import numpy as np
import cv2
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP


# ============== Model Definition (train_lnn.py နဲ့ တူညီရမယ်) ==============
class CNNFeatureExtractor(nn.Module):
    """Image ကနေ feature vector ထုတ်မယ်"""
    
    def __init__(self, output_dim=64):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(batch_size, seq_len, -1)
        return x


class LNNModel(nn.Module):
    """CNN + LNN Model"""
    
    def __init__(self, feature_dim=64, lnn_units=64, output_size=2, use_cfc=True):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor(output_dim=feature_dim)
        self.wiring = AutoNCP(units=lnn_units, output_size=output_size)
        
        if use_cfc:
            self.lnn = CfC(feature_dim, self.wiring)
        else:
            self.lnn = LTC(feature_dim, self.wiring)
        
        self.hidden = None
        
    def forward(self, x, return_sequences=True):
        features = self.cnn(x)
        output, self.hidden = self.lnn(features, self.hidden)
        
        if return_sequences:
            return output
        else:
            return output[:, -1, :]
    
    def reset_hidden(self):
        self.hidden = None
    
    def predict_single(self, x):
        """Single image prediction (real-time inference)"""
        # x: [C, H, W] -> [1, 1, C, H, W]
        x = x.unsqueeze(0).unsqueeze(0)
        features = self.cnn(x)  # [1, 1, feature_dim]
        output, self.hidden = self.lnn(features, self.hidden)
        return output[0, 0]  # [2] - linear.x, angular.z


# ============== ROS2 Inference Node ==============
class LNNInferenceNode(Node):
    """
    LNN Inference ROS2 Node
    
    Subscribes:
        /rgb (sensor_msgs/Image) or /rgb/compressed (CompressedImage)
    
    Publishes:
        /cmd_vel (geometry_msgs/Twist)
    """
    
    def __init__(self):
        super().__init__('lnn_inference_node')
        
        # Parameters
        self.declare_parameter('model_path', './lnn_model.pth')
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('img_width', 160)
        self.declare_parameter('img_height', 120)
        self.declare_parameter('max_linear_vel', 0.5)  # m/s
        self.declare_parameter('max_angular_vel', 1.0)  # rad/s
        self.declare_parameter('inference_rate', 10.0)  # Hz
        self.declare_parameter('device', 'cuda')  # cuda or cpu
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.img_width = self.get_parameter('img_width').value
        self.img_height = self.get_parameter('img_height').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.inference_rate = self.get_parameter('inference_rate').value
        device_param = self.get_parameter('device').value
        
        # Device setup
        if device_param == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.get_logger().info(f'Using device: {self.device}')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load model
        self.model = self._load_model()
        
        # Subscriber
        if self.use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.image_topic + '/compressed',
                self.image_callback,
                10
            )
        else:
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                10
            )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10
        )
        
        # State
        self.latest_image = None
        self.is_running = True
        
        # Inference timer (rate limiting)
        timer_period = 1.0 / self.inference_rate
        self.inference_timer = self.create_timer(timer_period, self.inference_callback)
        
        self.get_logger().info('LNN Inference Node initialized')
        self.get_logger().info(f'  Model: {self.model_path}')
        self.get_logger().info(f'  Image topic: {self.image_topic}')
        self.get_logger().info(f'  Cmd vel topic: {self.cmd_vel_topic}')
        self.get_logger().info(f'  Inference rate: {self.inference_rate} Hz')
    
    def _load_model(self):
        """Trained model ကို load လုပ်မယ်"""
        self.get_logger().info(f'Loading model from: {self.model_path}')
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Model configuration from checkpoint
            config = checkpoint.get('config', {})
            lnn_units = config.get('LNN_UNITS', 64)
            output_size = config.get('OUTPUT_SIZE', 2)
            
            # Create model
            model = LNNModel(
                feature_dim=64,
                lnn_units=lnn_units,
                output_size=output_size,
                use_cfc=True
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            self.get_logger().info('Model loaded successfully!')
            return model
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
    
    def image_callback(self, msg):
        """Image message ကို receive လုပ်မယ်"""
        try:
            if self.use_compressed:
                # CompressedImage
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # sensor_msgs/Image
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Store latest image
            self.latest_image = cv_image
            
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')
    
    def inference_callback(self):
        """Timer callback - inference လုပ်ပြီး cmd_vel publish မယ်"""
        if self.latest_image is None:
            return
        
        try:
            # Preprocess image
            img = self._preprocess_image(self.latest_image)
            
            # Inference
            with torch.no_grad():
                prediction = self.model.predict_single(img)
            
            # Convert to cmd_vel
            linear_x = float(prediction[0].cpu()) * self.max_linear_vel
            angular_z = float(prediction[1].cpu()) * self.max_angular_vel
            
            # Training data မှာ linear velocity က positive only ဖြစ်တာကြောင့်
            # negative value ထွက်လာရင် 0 အဖြစ် clamp လုပ်မယ် (backward မသွားစေရန်)
            if linear_x < 0:
                linear_x = 0.0
            
            # Clamp values
            linear_x = np.clip(linear_x, 0, self.max_linear_vel)  # 0 to max only
            angular_z = np.clip(angular_z, -self.max_angular_vel, self.max_angular_vel)
            
            # Publish
            cmd_msg = Twist()
            khh_const = 50.0  # Increased for better linear velocity
            ang_const = 3.0
            cmd_msg.linear.x = linear_x * khh_const
            cmd_msg.angular.z = angular_z / ang_const
            self.cmd_vel_pub.publish(cmd_msg)
            
            self.get_logger().debug(
                f'Predicted: linear.x={linear_x:.3f}, angular.z={angular_z:.3f}'
            )
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
    
    def _preprocess_image(self, cv_image):
        """Image preprocessing"""
        # Resize
        img = cv2.resize(cv_image, (self.img_width, self.img_height))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # [H, W, C] -> [C, H, W]
        img = np.transpose(img, (2, 0, 1))
        
        # To tensor
        img_tensor = torch.from_numpy(img).to(self.device)
        
        return img_tensor
    
    def reset_hidden_state(self):
        """Hidden state reset (new episode စရင်)"""
        if self.model is not None:
            self.model.reset_hidden()
            self.get_logger().info('Hidden state reset')
    
    def stop(self):
        """Stop publishing and reset"""
        self.is_running = False
        
        # Publish zero velocity
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)
        
        self.get_logger().info('Stopped - publishing zero velocity')


# ============== Standalone Inference (ROS2 မလိုဘဲ) ==============
class StandaloneInference:
    """
    ROS2 မပါဘဲ inference လုပ်ချင်ရင် သုံးမယ်
    Camera/Video file ကနေ inference
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image settings
        self.img_width = 160
        self.img_height = 120
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        config = checkpoint.get('config', {})
        model = LNNModel(
            feature_dim=64,
            lnn_units=config.get('LNN_UNITS', 64),
            output_size=config.get('OUTPUT_SIZE', 2),
            use_cfc=True
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print('Model loaded!')
        return model
    
    def preprocess(self, cv_image):
        img = cv2.resize(cv_image, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).to(self.device)
    
    def predict(self, cv_image):
        """Single image prediction"""
        img_tensor = self.preprocess(cv_image)
        
        with torch.no_grad():
            prediction = self.model.predict_single(img_tensor)
        
        linear_x = float(prediction[0].cpu())
        angular_z = float(prediction[1].cpu())
        
        return linear_x, angular_z
    
    def run_camera(self, camera_id=0):
        """Camera ကနေ real-time inference"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f'Cannot open camera {camera_id}')
            return
        
        print('Press Q to quit')
        self.model.reset_hidden()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            linear_x, angular_z = self.predict(frame)
            
            # Display
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f'Linear: {linear_x:.3f} m/s',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                f'Angular: {angular_z:.3f} rad/s',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Steering visualization
            h, w = display_frame.shape[:2]
            center_x = w // 2
            center_y = h - 50
            arrow_len = int(angular_z * 100)
            cv2.arrowedLine(
                display_frame,
                (center_x, center_y),
                (center_x + arrow_len, center_y),
                (255, 0, 0),
                3
            )
            
            cv2.imshow('LNN Inference', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        """Video file ကနေ inference"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f'Cannot open video: {video_path}')
            return
        
        print('Press Q to quit, SPACE to pause')
        self.model.reset_hidden()
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                linear_x, angular_z = self.predict(frame)
                
                # Display
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f'Linear: {linear_x:.3f} | Angular: {angular_z:.3f}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('LNN Inference', display_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()


# ============== Main ==============
def main(args=None):
    rclpy.init(args=args)
    
    node = LNNInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt - shutting down')
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--standalone':
        # Standalone mode (ROS2 မလို)
        import argparse
        
        parser = argparse.ArgumentParser(description='LNN Standalone Inference')
        parser.add_argument('--standalone', action='store_true')
        parser.add_argument('--model', type=str, default='./lnn_model.pth', help='Model path')
        parser.add_argument('--camera', type=int, default=None, help='Camera ID')
        parser.add_argument('--video', type=str, default=None, help='Video file path')
        parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
        
        args = parser.parse_args()
        
        inference = StandaloneInference(args.model, args.device)
        
        if args.camera is not None:
            inference.run_camera(args.camera)
        elif args.video is not None:
            inference.run_video(args.video)
        else:
            print('Please specify --camera <id> or --video <path>')
    else:
        # ROS2 mode
        main()
