#!/usr/bin/env python3
"""
LNN Inference Node with CNN + LNN Attention Visualization
Trained LNN model ကို သုံးပြီး Image -> cmd_vel prediction
+ CNN Attention Mask (spatial attention)
+ LNN Attention (temporal/neuron attention)
+ Linear vel, Angular vel, Hz ကို ဖေါ်ပြမယ်

Usage:
    ros2 run <package_name> lnn_inference_node_mask.py
    
Or standalone:
    python3 lnn_inference_node_mask.py --standalone --camera 0
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import time
from collections import deque
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP


# ============== Model Definition with CNN + LNN Attention ==============
class CNNFeatureExtractorWithAttention(nn.Module):
    """Image ကနေ feature vector ထုတ်မယ် + Attention Map"""
    
    def __init__(self, output_dim=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)
        
        self.attention_map = None
        self.feature_maps = {}
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        
        x = F.relu(self.bn1(self.conv1(x)))
        self.feature_maps['conv1'] = x.detach()
        
        x = F.relu(self.bn2(self.conv2(x)))
        self.feature_maps['conv2'] = x.detach()
        
        x = F.relu(self.bn3(self.conv3(x)))
        self.feature_maps['conv3'] = x.detach()
        
        x = F.relu(self.bn4(self.conv4(x)))
        self.feature_maps['conv4'] = x.detach()
        
        self.attention_map = torch.mean(x, dim=1, keepdim=True)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(batch_size, seq_len, -1)
        
        return x
    
    def get_attention_map(self, original_size):
        if self.attention_map is None:
            return None
        
        attn = self.attention_map[0, 0]
        attn = attn - attn.min()
        if attn.max() > 0:
            attn = attn / attn.max()
        
        attn_np = attn.cpu().numpy()
        attn_resized = cv2.resize(attn_np, (original_size[1], original_size[0]))
        
        return attn_resized


class LNNModelWithAttention(nn.Module):
    """CNN + LNN Model with Both Attention Types"""
    
    def __init__(self, feature_dim=64, lnn_units=64, output_size=2, use_cfc=True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.lnn_units = lnn_units
        self.output_size = output_size
        
        self.cnn = CNNFeatureExtractorWithAttention(output_dim=feature_dim)
        self.wiring = AutoNCP(units=lnn_units, output_size=output_size)
        
        if use_cfc:
            self.lnn = CfC(feature_dim, self.wiring)
        else:
            self.lnn = LTC(feature_dim, self.wiring)
        
        self.hidden = None
        
        # LNN Attention tracking
        self.hidden_history = deque(maxlen=50)  # Last 50 hidden states
        self.current_hidden_activations = None
        self.neuron_importance = None
        
    def forward(self, x, return_sequences=True):
        features = self.cnn(x)
        output, self.hidden = self.lnn(features, self.hidden)
        
        if return_sequences:
            return output
        else:
            return output[:, -1, :]
    
    def reset_hidden(self):
        self.hidden = None
        self.hidden_history.clear()
        self.current_hidden_activations = None
        self.neuron_importance = None
    
    def predict_single(self, x):
        """Single image prediction with LNN attention tracking"""
        x = x.unsqueeze(0).unsqueeze(0)
        features = self.cnn(x)
        output, self.hidden = self.lnn(features, self.hidden)
        
        # Track hidden state for LNN attention
        if self.hidden is not None:
            hidden_np = self.hidden.detach().cpu().numpy().flatten()
            self.current_hidden_activations = hidden_np
            self.hidden_history.append(hidden_np.copy())
            
            # Calculate neuron importance (absolute activation strength)
            self.neuron_importance = np.abs(hidden_np)
            
        return output[0, 0]
    
    def get_cnn_attention_map(self, original_size):
        """Get CNN spatial attention map"""
        return self.cnn.get_attention_map(original_size)
    
    def get_lnn_attention_info(self):
        """
        Get LNN attention information:
        - current_activations: Current hidden state activations
        - neuron_importance: Absolute activation strength per neuron
        - temporal_pattern: Hidden state history
        - wiring_info: NCP wiring structure info
        """
        if self.current_hidden_activations is None:
            return None
        
        info = {
            'current_activations': self.current_hidden_activations,
            'neuron_importance': self.neuron_importance,
            'temporal_history': np.array(list(self.hidden_history)) if len(self.hidden_history) > 0 else None,
            'num_neurons': len(self.current_hidden_activations),
            'wiring': {
                'sensory_neurons': self.wiring.sensory_size,
                'inter_neurons': self.wiring.inter_size,
                'command_neurons': self.wiring.command_size,
                'motor_neurons': self.wiring.motor_size,
            }
        }
        
        return info
    
    def load_from_original(self, checkpoint, device):
        """Load weights from original model checkpoint"""
        from collections import OrderedDict
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        
        for key, value in state_dict.items():
            if key.startswith('cnn.cnn.'):
                parts = key.split('.')
                idx = int(parts[2])
                rest = '.'.join(parts[3:]) if len(parts) > 3 else ''
                
                layer_mapping = {
                    0: 'conv1', 1: 'bn1',
                    3: 'conv2', 4: 'bn2',
                    6: 'conv3', 7: 'bn3',
                    9: 'conv4', 10: 'bn4',
                }
                
                if idx in layer_mapping:
                    new_key = f'cnn.{layer_mapping[idx]}'
                    if rest:
                        new_key += f'.{rest}'
                    new_state_dict[new_key] = value
            elif key.startswith('cnn.fc.'):
                new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        
        self.load_state_dict(new_state_dict)
        self.to(device)
        self.eval()


# ============== Visualization Helper ==============
class AttentionVisualizer:
    """CNN + LNN Attention Visualization"""
    
    def __init__(self, display_width=960, display_height=720):
        self.display_width = display_width
        self.display_height = display_height
        
        # Layout sizes
        self.main_view_width = 640
        self.main_view_height = 480
        self.lnn_panel_width = 320
        self.info_panel_height = 240
    
    def create_visualization(self, original_image, cnn_attention, lnn_info, 
                            linear_x, angular_z, fps, max_linear_vel, max_angular_vel):
        """
        Create combined CNN + LNN attention visualization
        
        Layout:
        +------------------+-------------+
        |                  |   LNN       |
        |  CNN Attention   |  Neurons    |
        |  (Image+Heatmap) |  Activity   |
        +------------------+-------------+
        |     Info Panel + Temporal History     |
        +---------------------------------------+
        """
        # === Left: CNN Attention on Image ===
        main_img = cv2.resize(original_image, (self.main_view_width, self.main_view_height))
        
        if cnn_attention is not None:
            cnn_attn_resized = cv2.resize(cnn_attention, (self.main_view_width, self.main_view_height))
            cnn_attn_colored = cv2.applyColorMap(
                (cnn_attn_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            main_view = cv2.addWeighted(main_img, 0.5, cnn_attn_colored, 0.5, 0)
        else:
            main_view = main_img
        
        # Add CNN label
        cv2.putText(main_view, 'CNN Spatial Attention', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # === Right: LNN Neuron Activity ===
        lnn_panel = self._create_lnn_neuron_panel(lnn_info)
        
        # Combine top row
        top_row = np.hstack([main_view, lnn_panel])
        
        # === Bottom: Info + Temporal History ===
        info_panel = self._create_info_panel(
            lnn_info, linear_x, angular_z, fps, max_linear_vel, max_angular_vel
        )
        
        # Final combine
        combined = np.vstack([top_row, info_panel])
        
        return combined
    
    def _create_lnn_neuron_panel(self, lnn_info):
        """Create LNN neuron activity visualization"""
        panel = np.zeros((self.main_view_height, self.lnn_panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        cv2.putText(panel, 'LNN Neuron Activity', (10, 30), font, 0.7, (255, 255, 255), 2)
        
        if lnn_info is None:
            cv2.putText(panel, 'Waiting for data...', (10, 100), font, 0.5, (150, 150, 150), 1)
            return panel
        
        activations = lnn_info['current_activations']
        importance = lnn_info['neuron_importance']
        wiring = lnn_info['wiring']
        
        # Wiring info
        y_offset = 60
        cv2.putText(panel, f"Neurons: {lnn_info['num_neurons']}", (10, y_offset), 
                    font, 0.5, (200, 200, 200), 1)
        y_offset += 25
        cv2.putText(panel, f"Sensory: {wiring['sensory_neurons']} | Inter: {wiring['inter_neurons']}", 
                    (10, y_offset), font, 0.4, (150, 200, 150), 1)
        y_offset += 20
        cv2.putText(panel, f"Command: {wiring['command_neurons']} | Motor: {wiring['motor_neurons']}", 
                    (10, y_offset), font, 0.4, (150, 150, 200), 1)
        
        # Neuron activation bars
        y_offset += 30
        cv2.putText(panel, 'Neuron Activations:', (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 10
        
        num_neurons = len(activations)
        bar_height = min(8, (self.main_view_height - y_offset - 50) // num_neurons)
        bar_max_width = self.lnn_panel_width - 80
        
        # Normalize activations for display
        act_max = np.max(np.abs(activations)) + 1e-6
        
        for i, (act, imp) in enumerate(zip(activations, importance)):
            y = y_offset + i * (bar_height + 2)
            if y > self.main_view_height - 30:
                break
            
            # Neuron index
            cv2.putText(panel, f'{i:2d}', (5, y + bar_height - 1), font, 0.3, (150, 150, 150), 1)
            
            # Bar background
            cv2.rectangle(panel, (30, y), (30 + bar_max_width, y + bar_height), (60, 60, 60), -1)
            
            # Activation bar (green for positive, red for negative)
            bar_len = int(abs(act) / act_max * bar_max_width * 0.5)
            center_x = 30 + bar_max_width // 2
            
            if act >= 0:
                color = (0, int(200 * imp / act_max + 55), 0)  # Green
                cv2.rectangle(panel, (center_x, y), (center_x + bar_len, y + bar_height), color, -1)
            else:
                color = (0, 0, int(200 * imp / act_max + 55))  # Red
                cv2.rectangle(panel, (center_x - bar_len, y), (center_x, y + bar_height), color, -1)
            
            # Center line
            cv2.line(panel, (center_x, y), (center_x, y + bar_height), (100, 100, 100), 1)
        
        # Show top important neurons
        y_offset = self.main_view_height - 60
        top_neurons = np.argsort(importance)[-5:][::-1]
        cv2.putText(panel, 'Top Active Neurons:', (10, y_offset), font, 0.5, (255, 200, 0), 1)
        y_offset += 20
        top_str = ', '.join([f'N{n}' for n in top_neurons])
        cv2.putText(panel, top_str, (10, y_offset), font, 0.4, (200, 200, 100), 1)
        
        return panel
    
    def _create_info_panel(self, lnn_info, linear_x, angular_z, fps, max_linear_vel, max_angular_vel):
        """Create info panel with velocity info and temporal history"""
        total_width = self.main_view_width + self.lnn_panel_width
        panel = np.zeros((self.info_panel_height, total_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # === Left side: Velocity Info ===
        info_x = 20
        
        # Linear Velocity
        linear_color = (0, 255, 0) if linear_x > 0.1 else (0, 200, 200)
        cv2.putText(panel, f'Linear Vel:  {linear_x:.3f} m/s', 
                    (info_x, 35), font, 0.8, linear_color, 2)
        
        # Angular Velocity
        if angular_z > 0.05:
            ang_color = (255, 150, 0)
            direction = "LEFT"
        elif angular_z < -0.05:
            ang_color = (0, 150, 255)
            direction = "RIGHT"
        else:
            ang_color = (200, 200, 200)
            direction = "STRAIGHT"
        
        cv2.putText(panel, f'Angular Vel: {angular_z:.3f} rad/s ({direction})', 
                    (info_x, 75), font, 0.8, ang_color, 2)
        
        # FPS
        cv2.putText(panel, f'Inference:   {fps:.1f} Hz', 
                    (info_x, 115), font, 0.8, (0, 255, 255), 2)
        
        # Velocity bars
        bar_x = 350
        bar_width = 180
        bar_height = 25
        
        # Linear bar
        cv2.rectangle(panel, (bar_x, 15), (bar_x + bar_width, 15 + bar_height), (80, 80, 80), -1)
        linear_bar_len = int((linear_x / max_linear_vel) * bar_width)
        cv2.rectangle(panel, (bar_x, 15), (bar_x + linear_bar_len, 15 + bar_height), (0, 255, 0), -1)
        cv2.rectangle(panel, (bar_x, 15), (bar_x + bar_width, 15 + bar_height), (200, 200, 200), 1)
        
        # Angular bar
        cv2.rectangle(panel, (bar_x, 55), (bar_x + bar_width, 55 + bar_height), (80, 80, 80), -1)
        center = bar_x + bar_width // 2
        ang_bar_len = int((angular_z / max_angular_vel) * (bar_width // 2))
        if ang_bar_len > 0:
            cv2.rectangle(panel, (center, 55), (center + ang_bar_len, 55 + bar_height), (255, 150, 0), -1)
        else:
            cv2.rectangle(panel, (center + ang_bar_len, 55), (center, 55 + bar_height), (0, 150, 255), -1)
        cv2.line(panel, (center, 55), (center, 55 + bar_height), (255, 255, 255), 2)
        cv2.rectangle(panel, (bar_x, 55), (bar_x + bar_width, 55 + bar_height), (200, 200, 200), 1)
        
        # === Right side: Temporal History Graph ===
        if lnn_info is not None and lnn_info['temporal_history'] is not None:
            history = lnn_info['temporal_history']
            
            graph_x = 580
            graph_y = 20
            graph_width = total_width - graph_x - 20
            graph_height = self.info_panel_height - 40
            
            # Graph background
            cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                          (50, 50, 50), -1)
            cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), 
                          (100, 100, 100), 1)
            
            # Title
            cv2.putText(panel, 'LNN Temporal Activity', (graph_x + 10, graph_y - 5), 
                        font, 0.5, (255, 255, 255), 1)
            
            if len(history) > 1:
                # Plot mean activation over time
                mean_activations = np.mean(np.abs(history), axis=1)
                
                # Normalize
                max_val = np.max(mean_activations) + 1e-6
                normalized = mean_activations / max_val
                
                # Draw line graph
                points = []
                for i, val in enumerate(normalized):
                    x = graph_x + int(i / len(normalized) * graph_width)
                    y = graph_y + graph_height - int(val * (graph_height - 10)) - 5
                    points.append((x, y))
                
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(panel, points[i], points[i + 1], (0, 255, 255), 2)
                
                # Also plot top 3 neuron activities with different colors
                top_neurons = np.argsort(np.mean(np.abs(history), axis=0))[-3:]
                colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
                
                for neuron_idx, color in zip(top_neurons, colors):
                    neuron_history = history[:, neuron_idx]
                    max_val = np.max(np.abs(neuron_history)) + 1e-6
                    normalized = (neuron_history - neuron_history.min()) / (np.max(neuron_history) - neuron_history.min() + 1e-6)
                    
                    points = []
                    for i, val in enumerate(normalized):
                        x = graph_x + int(i / len(normalized) * graph_width)
                        y = graph_y + graph_height - int(val * (graph_height - 10)) - 5
                        points.append((x, y))
                    
                    if len(points) > 1:
                        for i in range(len(points) - 1):
                            cv2.line(panel, points[i], points[i + 1], color, 1)
            
            # Legend
            cv2.putText(panel, 'Mean', (graph_x + graph_width - 80, graph_y + 15), 
                        font, 0.35, (0, 255, 255), 1)
            cv2.putText(panel, 'Top Neurons', (graph_x + graph_width - 80, graph_y + 30), 
                        font, 0.35, (200, 200, 200), 1)
        
        # Keyboard shortcuts
        cv2.putText(panel, 'Q:Quit  R:Reset  V:View Mode', (info_x, self.info_panel_height - 15), 
                    font, 0.5, (150, 150, 150), 1)
        
        return panel


# ============== ROS2 Inference Node ==============
class LNNInferenceNodeWithMask(Node):
    """LNN Inference ROS2 Node with CNN + LNN Attention"""
    
    def __init__(self):
        super().__init__('lnn_inference_node_mask')
        
        # Parameters
        self.declare_parameter('model_path', './lnn_model.pth')
        self.declare_parameter('image_topic', '/rgb')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('img_width', 160)
        self.declare_parameter('img_height', 120)
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('show_visualization', True)
        
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
        self.show_visualization = self.get_parameter('show_visualization').value
        
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
        
        # Visualizer
        self.visualizer = AttentionVisualizer()
        
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
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # State
        self.latest_image = None
        self.is_running = True
        
        # FPS
        self.last_time = time.time()
        self.fps = 0.0
        self.fps_alpha = 0.9
        
        # Timer
        timer_period = 1.0 / self.inference_rate
        self.inference_timer = self.create_timer(timer_period, self.inference_callback)
        
        self.get_logger().info('LNN Inference Node with CNN+LNN Attention initialized')
    
    def _load_model(self):
        self.get_logger().info(f'Loading model from: {self.model_path}')
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint.get('config', {})
            
            model = LNNModelWithAttention(
                feature_dim=64,
                lnn_units=config.get('LNN_UNITS', 64),
                output_size=config.get('OUTPUT_SIZE', 2),
                use_cfc=True
            )
            
            model.load_from_original(checkpoint, self.device)
            self.get_logger().info('Model loaded!')
            return model
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
    
    def image_callback(self, msg):
        try:
            if self.use_compressed:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.latest_image = cv_image
            
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')
    
    def inference_callback(self):
        if self.latest_image is None:
            return
        
        try:
            # FPS
            current_time = time.time()
            instant_fps = 1.0 / (current_time - self.last_time + 1e-6)
            self.fps = self.fps_alpha * self.fps + (1 - self.fps_alpha) * instant_fps
            self.last_time = current_time
            
            # Preprocess
            img = self._preprocess_image(self.latest_image)
            
            # Inference
            with torch.no_grad():
                prediction = self.model.predict_single(img)
            
            # Get values
            linear_x = float(prediction[0].cpu()) * self.max_linear_vel
            angular_z = float(prediction[1].cpu()) * self.max_angular_vel
            
            if linear_x < 0:
                linear_x = 0.0
            
            linear_x = np.clip(linear_x, 0, self.max_linear_vel)
            angular_z = np.clip(angular_z, -self.max_angular_vel, self.max_angular_vel)
            
            # Publish
            cmd_msg = Twist()
            khh_const = 50.0
            ang_const = 3.0
            cmd_msg.linear.x = linear_x * khh_const
            cmd_msg.angular.z = angular_z / ang_const
            self.cmd_vel_pub.publish(cmd_msg)
            
            # Visualization
            if self.show_visualization:
                cnn_attn = self.model.get_cnn_attention_map((self.img_height, self.img_width))
                lnn_info = self.model.get_lnn_attention_info()
                
                display = self.visualizer.create_visualization(
                    self.latest_image, cnn_attn, lnn_info,
                    linear_x, angular_z, self.fps,
                    self.max_linear_vel, self.max_angular_vel
                )
                
                cv2.imshow('LNN Attention Visualization', display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.is_running = False
                    rclpy.shutdown()
                elif key == ord('r'):
                    self.model.reset_hidden()
                    self.get_logger().info('Hidden state reset')
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
    
    def _preprocess_image(self, cv_image):
        img = cv2.resize(cv_image, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).to(self.device)
    
    def stop(self):
        self.is_running = False
        cmd_msg = Twist()
        self.cmd_vel_pub.publish(cmd_msg)
        cv2.destroyAllWindows()


# ============== Standalone Inference ==============
class StandaloneInferenceWithMask:
    """Standalone inference with CNN + LNN attention"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.model = self._load_model(model_path)
        self.visualizer = AttentionVisualizer()
        
        self.img_width = 160
        self.img_height = 120
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0
        
        self.last_time = time.time()
        self.fps = 0.0
        self.fps_alpha = 0.9
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        model = LNNModelWithAttention(
            feature_dim=64,
            lnn_units=config.get('LNN_UNITS', 64),
            output_size=config.get('OUTPUT_SIZE', 2),
            use_cfc=True
        )
        
        model.load_from_original(checkpoint, self.device)
        print('Model loaded!')
        return model
    
    def preprocess(self, cv_image):
        img = cv2.resize(cv_image, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).to(self.device)
    
    def predict(self, cv_image):
        img_tensor = self.preprocess(cv_image)
        
        with torch.no_grad():
            prediction = self.model.predict_single(img_tensor)
        
        linear_x = float(prediction[0].cpu())
        angular_z = float(prediction[1].cpu())
        
        if linear_x < 0:
            linear_x = 0.0
        
        return linear_x, angular_z
    
    def run_camera(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f'Cannot open camera {camera_id}')
            return
        
        print('Press Q to quit, R to reset hidden state')
        self.model.reset_hidden()
        
        while True:
            current_time = time.time()
            instant_fps = 1.0 / (current_time - self.last_time + 1e-6)
            self.fps = self.fps_alpha * self.fps + (1 - self.fps_alpha) * instant_fps
            self.last_time = current_time
            
            ret, frame = cap.read()
            if not ret:
                break
            
            linear_x, angular_z = self.predict(frame)
            
            cnn_attn = self.model.get_cnn_attention_map((self.img_height, self.img_width))
            lnn_info = self.model.get_lnn_attention_info()
            
            display = self.visualizer.create_visualization(
                frame, cnn_attn, lnn_info,
                linear_x, angular_z, self.fps,
                self.max_linear_vel, self.max_angular_vel
            )
            
            cv2.imshow('LNN Attention Visualization', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.model.reset_hidden()
                print('Hidden state reset')
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f'Cannot open video: {video_path}')
            return
        
        print('Press Q to quit, SPACE to pause, R to reset')
        self.model.reset_hidden()
        
        paused = False
        
        while True:
            if not paused:
                current_time = time.time()
                instant_fps = 1.0 / (current_time - self.last_time + 1e-6)
                self.fps = self.fps_alpha * self.fps + (1 - self.fps_alpha) * instant_fps
                self.last_time = current_time
                
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.model.reset_hidden()
                    continue
                
                linear_x, angular_z = self.predict(frame)
                
                cnn_attn = self.model.get_cnn_attention_map((self.img_height, self.img_width))
                lnn_info = self.model.get_lnn_attention_info()
                
                display = self.visualizer.create_visualization(
                    frame, cnn_attn, lnn_info,
                    linear_x, angular_z, self.fps,
                    self.max_linear_vel, self.max_angular_vel
                )
                
                cv2.imshow('LNN Attention Visualization', display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                self.model.reset_hidden()
                print('Hidden state reset')
        
        cap.release()
        cv2.destroyAllWindows()


# ============== Main ==============
def main(args=None):
    rclpy.init(args=args)
    node = LNNInferenceNodeWithMask()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down')
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--standalone':
        import argparse
        
        parser = argparse.ArgumentParser(description='LNN Inference with CNN+LNN Attention')
        parser.add_argument('--standalone', action='store_true')
        parser.add_argument('--model', type=str, default='./lnn_model.pth')
        parser.add_argument('--camera', type=int, default=None)
        parser.add_argument('--video', type=str, default=None)
        parser.add_argument('--device', type=str, default='cuda')
        
        args = parser.parse_args()
        
        inference = StandaloneInferenceWithMask(args.model, args.device)
        
        if args.camera is not None:
            inference.run_camera(args.camera)
        elif args.video is not None:
            inference.run_video(args.video)
        else:
            print('Usage:')
            print('  python3 lnn_inference_node_mask.py --standalone --camera 0 --model ./lnn_model.pth')
            print('  python3 lnn_inference_node_mask.py --standalone --video test.mp4 --model ./lnn_model.pth')
    else:
        main()
