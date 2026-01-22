### how to train g1
```
cd /home/mr_robot/ISAAAC/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1
```
copy Isaac-Velocity-Rough-G1-v0
```
cd /home/mr_robot/ISAAAC/IsaacLab/scripts/reinforcement_learning/rsl_rl
```

#### train
```
python train.py --task=Isaac-Velocity-Rough-G1-v0 --num_envs=5 --headless
```