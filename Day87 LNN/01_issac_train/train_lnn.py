#!/usr/bin/env python3
"""
LNN Training Script for ROS2 Bag Data
Image -> cmd_vel (linear.x, angular.z) prediction

ROS2 bag record /image /cmd_vel_unstamped ကနေ data ကို train မယ်
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP, NCP
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_types_from_msg, Stores, get_typestore
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


# ============== Configuration ==============
class Config:
    # Data
    BAG_PATH = "/home/mr_robot/square"  # ROS2 bag folder path
    IMAGE_TOPIC = "/rgb"
    CMD_VEL_TOPIC = "/cmd_vel"
    
    # Image preprocessing
    IMG_WIDTH = 160
    IMG_HEIGHT = 120
    IMG_CHANNELS = 3
    
    # Sequence
    SEQ_LENGTH = 16  # Time steps for LNN
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    TRAIN_SPLIT = 0.8
    
    # Model
    LNN_UNITS = 64  # Inter neurons
    OUTPUT_SIZE = 2  # linear.x, angular.z
    
    # Paths
    MODEL_SAVE_PATH = "./lnn_model.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============== Dataset ==============
class ROS2BagDataset(Dataset):
    """ROS2 bag ကနေ image နဲ့ cmd_vel ကို load လုပ်မယ်"""
    
    def __init__(self, bag_path, seq_length=16, img_size=(160, 120)):
        self.seq_length = seq_length
        self.img_size = img_size
        
        # Data containers
        self.images = []
        self.cmd_vels = []
        self.timestamps_img = []
        self.timestamps_cmd = []
        
        # Load data from bag
        self._load_bag(bag_path)
        
        # Synchronize timestamps
        self._sync_data()
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _load_bag(self, bag_path):
        """ROS2 bag ကနေ data ကို ဖတ်မယ်"""
        print(f"Loading bag from: {bag_path}")
        
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        
        with Reader(bag_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == Config.IMAGE_TOPIC:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    # Image message ကို numpy array အဖြစ်ပြောင်း
                    img = self._decode_image(msg)
                    if img is not None:
                        self.images.append(img)
                        self.timestamps_img.append(timestamp)
                        
                elif connection.topic == Config.CMD_VEL_TOPIC:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    # Twist message ကနေ linear.x, angular.z
                    cmd = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)
                    self.cmd_vels.append(cmd)
                    self.timestamps_cmd.append(timestamp)
        
        print(f"Loaded {len(self.images)} images, {len(self.cmd_vels)} cmd_vel messages")
    
    def _decode_image(self, msg):
        """Image message ကို numpy array အဖြစ်ပြောင်း"""
        try:
            # CompressedImage or Image
            if hasattr(msg, 'data'):
                if hasattr(msg, 'format'):  # CompressedImage
                    img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:  # sensor_msgs/Image
                    img = np.array(msg.data, dtype=np.uint8)
                    img = img.reshape((msg.height, msg.width, -1))
                    if img.shape[2] == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Resize
                img = cv2.resize(img, self.img_size)
                return img
        except Exception as e:
            print(f"Image decode error: {e}")
            return None
    
    def _sync_data(self):
        """Image နဲ့ cmd_vel ကို timestamp အရ sync လုပ်မယ်"""
        if len(self.images) == 0 or len(self.cmd_vels) == 0:
            print("Warning: No data to sync!")
            return
            
        synced_images = []
        synced_cmds = []
        
        cmd_idx = 0
        for i, img_ts in enumerate(self.timestamps_img):
            # Nearest cmd_vel ကို ရှာမယ်
            while cmd_idx < len(self.timestamps_cmd) - 1:
                if abs(self.timestamps_cmd[cmd_idx + 1] - img_ts) < abs(self.timestamps_cmd[cmd_idx] - img_ts):
                    cmd_idx += 1
                else:
                    break
            
            # 100ms အတွင်း ဆိုရင် sync လုပ်မယ်
            if abs(self.timestamps_cmd[cmd_idx] - img_ts) < 100_000_000:  # 100ms in ns
                synced_images.append(self.images[i])
                synced_cmds.append(self.cmd_vels[cmd_idx])
        
        self.synced_images = np.array(synced_images)
        self.synced_cmds = np.array(synced_cmds)
        print(f"Synced data: {len(self.synced_images)} pairs")
    
    def _create_sequences(self):
        """Sequence data ဖန်တီးမယ်"""
        sequences = []
        
        if len(self.synced_images) < self.seq_length:
            print("Warning: Not enough data for sequences!")
            return sequences
        
        for i in range(len(self.synced_images) - self.seq_length):
            seq_imgs = self.synced_images[i:i + self.seq_length]
            seq_cmds = self.synced_cmds[i:i + self.seq_length]
            sequences.append((seq_imgs, seq_cmds))
        
        print(f"Created {len(sequences)} sequences")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        imgs, cmds = self.sequences[idx]
        
        # Normalize images [0, 1]
        imgs = imgs.astype(np.float32) / 255.0
        # [T, H, W, C] -> [T, C, H, W]
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        
        return torch.tensor(imgs), torch.tensor(cmds)


# ============== Model ==============
class CNNFeatureExtractor(nn.Module):
    """Image ကနေ feature vector ထုတ်မယ်"""
    
    def __init__(self, output_dim=64):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # [B, 3, 120, 160] -> [B, 32, 60, 80]
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # [B, 32, 60, 80] -> [B, 64, 30, 40]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # [B, 64, 30, 40] -> [B, 128, 15, 20]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # [B, 128, 15, 20] -> [B, 128, 7, 10]
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1]
        )
        
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        batch_size, seq_len = x.shape[:2]
        
        # Flatten batch and time
        x = x.view(-1, *x.shape[2:])  # [B*T, C, H, W]
        
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # [B*T, 128]
        x = self.fc(x)  # [B*T, output_dim]
        
        # Reshape back
        x = x.view(batch_size, seq_len, -1)  # [B, T, output_dim]
        
        return x


class LNNModel(nn.Module):
    """
    CNN Feature Extractor + Liquid Neural Network
    Image sequence -> cmd_vel prediction
    """
    
    def __init__(self, 
                 feature_dim=64,
                 lnn_units=64, 
                 output_size=2,
                 use_cfc=False):
        super().__init__()
        
        # CNN for feature extraction
        self.cnn = CNNFeatureExtractor(output_dim=feature_dim)
        
        # LNN Wiring (NCP structure)
        # Sensory -> Inter -> Command -> Motor
        self.wiring = AutoNCP(units=lnn_units, output_size=output_size)
        
        # LNN Cell (LTC or CfC)
        if use_cfc:
            # Closed-form Continuous-time (faster)
            self.lnn = CfC(feature_dim, self.wiring)
        else:
            # Liquid Time Constant
            self.lnn = LTC(feature_dim, self.wiring)
        
        # Hidden state
        self.hidden = None
        
    def forward(self, x, return_sequences=True):
        """
        x: [B, T, C, H, W] image sequence
        returns: [B, T, 2] or [B, 2] cmd_vel predictions
        """
        # Extract features
        features = self.cnn(x)  # [B, T, feature_dim]
        
        # LNN forward
        output, self.hidden = self.lnn(features, self.hidden)
        
        if return_sequences:
            return output  # [B, T, 2]
        else:
            return output[:, -1, :]  # [B, 2] last timestep only
    
    def reset_hidden(self, batch_size=1):
        """Hidden state reset (new sequence အတွက်)"""
        self.hidden = None


class LNNModelNCP(nn.Module):
    """
    Detailed NCP Wiring နဲ့ LNN Model
    """
    
    def __init__(self, 
                 feature_dim=64,
                 inter_neurons=32,
                 command_neurons=16,
                 motor_neurons=2):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor(output_dim=feature_dim)
        
        # Detailed NCP Wiring
        self.wiring = NCP(
            inter_neurons=inter_neurons,      # Processing layer
            command_neurons=command_neurons,  # Decision layer
            motor_neurons=motor_neurons,      # Output (linear.x, angular.z)
            sensory_fanout=4,                 # Each sensory connects to 4 inter
            inter_fanout=4,                   # Each inter connects to 4 command
            recurrent_command_synapses=6,     # Command layer recurrence
            motor_fanin=4                     # Each motor receives from 4 command
        )
        
        self.lnn = LTC(feature_dim, self.wiring)
        self.hidden = None
        
    def forward(self, x, return_sequences=True):
        features = self.cnn(x)
        output, self.hidden = self.lnn(features, self.hidden)
        
        if return_sequences:
            return output
        else:
            return output[:, -1, :]
    
    def reset_hidden(self, batch_size=1):
        self.hidden = None


# ============== Training ==============
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, cmd_vels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        cmd_vels = cmd_vels.to(device)
        
        # Reset hidden state for each batch
        model.reset_hidden(images.size(0))
        
        optimizer.zero_grad()
        
        # Forward
        predictions = model(images, return_sequences=True)
        
        # Loss
        loss = criterion(predictions, cmd_vels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping (LNN အတွက် important)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, cmd_vels in dataloader:
            images = images.to(device)
            cmd_vels = cmd_vels.to(device)
            
            model.reset_hidden(images.size(0))
            
            predictions = model(images, return_sequences=True)
            loss = criterion(predictions, cmd_vels)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(config: Config):
    """Main training function"""
    
    print(f"Using device: {config.DEVICE}")
    
    # Dataset
    dataset = ROS2BagDataset(
        bag_path=config.BAG_PATH,
        seq_length=config.SEQ_LENGTH,
        img_size=(config.IMG_WIDTH, config.IMG_HEIGHT)
    )
    
    if len(dataset) == 0:
        print("Error: No data loaded!")
        return
    
    # Train/Val split
    train_size = int(len(dataset) * config.TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = LNNModel(
        feature_dim=64,
        lnn_units=config.LNN_UNITS,
        output_size=config.OUTPUT_SIZE,
        use_cfc=True  # CfC is faster
    ).to(config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.__dict__
            }, config.MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LNN Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    
    print(f"\nTraining complete! Best model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Best validation loss: {best_val_loss:.6f}")


# ============== Dummy Data for Testing ==============
def create_dummy_dataset(num_samples=1000, seq_length=16):
    """Testing အတွက် dummy data ဖန်တီးမယ်"""
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, seq_length):
            self.num_samples = num_samples
            self.seq_length = seq_length
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random images [T, C, H, W]
            imgs = torch.randn(self.seq_length, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
            # Random cmd_vel [T, 2]
            cmds = torch.randn(self.seq_length, 2) * 0.5
            return imgs, cmds
    
    return DummyDataset(num_samples, seq_length)


def test_model():
    """Model architecture ကို test လုပ်မယ်"""
    print("Testing LNN Model...")
    
    config = Config()
    device = config.DEVICE
    
    # Dummy data
    dataset = create_dummy_dataset(100, config.SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Model
    model = LNNModel(
        feature_dim=64,
        lnn_units=config.LNN_UNITS,
        output_size=config.OUTPUT_SIZE,
        use_cfc=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Wiring: {model.wiring}")
    
    # Test forward pass
    for imgs, cmds in dataloader:
        imgs = imgs.to(device)
        model.reset_hidden(imgs.size(0))
        
        output = model(imgs)
        print(f"Input shape: {imgs.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0, -1]}")
        break
    
    print("Test passed!")


# ============== Main ==============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LNN Training for ROS2 Bag Data")
    parser.add_argument("--test", action="store_true", help="Test model with dummy data")
    parser.add_argument("--bag", type=str, default="/home/mr_robot/square", help="ROS2 bag path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    if args.test:
        test_model()
    else:
        config = Config()
        config.BAG_PATH = args.bag
        config.EPOCHS = args.epochs
        config.BATCH_SIZE = args.batch_size
        config.LEARNING_RATE = args.lr
        
        train(config)
