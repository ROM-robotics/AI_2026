### record data for 10 minutes
```
ros2 bag record /image /cmd_vel_unstamped
```

### train 
```bash
# Install dependencies
pip install ncps torch torchvision rosbags opencv-python tqdm matplotlib
pip install "numpy<2"

# Test model (dummy data နဲ့)
python train_lnn.py --test

# Train with ROS2 bag
python train_lnn.py --bag /home/mr_robot/square --epochs 100 --batch_size 32
```

### inference
```bash
# ROS2 Node အနေနဲ့ run
python3 lnn_inference_node.py


# Parameters နဲ့ run
ros2 run <package> lnn_inference_node.py --ros-args \
    -p model_path:=./lnn_model.pth \
    -p image_topic:=/image \
    -p cmd_vel_topic:=/cmd_vel_unstamped \
    -p inference_rate:=10.0 \
    -p max_linear_vel:=0.5 \
    -p max_angular_vel:=1.0

# Standalone mode (ROS2 မလို - camera/video test)
python3 lnn_inference_node.py --standalone --model ./lnn_model.pth --camera 0
python3 lnn_inference_node.py --standalone --model ./lnn_model.pth --video test.mp4
```
or
```bash
python3 lnn_inference_node_mask.py --standalone --camera 0 --model ./lnn_model.pth
```