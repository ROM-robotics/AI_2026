# NVIDIA Physical AI Installation Guide

## အဆင့် ၁ - GPU စစ်ဆေးခြင်း

အရင်ဆုံး GPU ရှိမရှိ စစ်ဆေးပါ:

```bash
# NVIDIA GPU ရှိမရှိ စစ်ဆေးရန်
nvidia-smi

# CUDA version ကြည့်ရန်
nvcc --version
```

**လိုအပ်ချက်များ:**
- NVIDIA GPU (RTX series recommended)
- CUDA 11.8 or later
- Ubuntu 20.04 or 22.04

---

## အဆင့် ၂ - Isaac Sim / Isaac Lab ထည့်သွင်းခြင်း

### နည်းလမ်း (က) - Isaac Sim Installer သုံးခြင်း

**၁. Isaac Sim Download လုပ်ရန်:**
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) သို့သွားပါ
- Account အသစ်ဖွင့်ပါ (အခမဲ့)
- Isaac Sim (Latest Version) download လုပ်ပါ

**၂. Installation လုပ်ရန်:**

```bash
# Download လုပ်ထားတဲ့ folder သို့သွားပါ
cd ~/Downloads

# Execute permission ပေးပါ
chmod +x IsaacSim-*.AppImage

# Run the installer
./IsaacSim-*.AppImage
```

---

### နည်းလမ်း (ခ) - Docker Container သုံးခြင်း (အလွယ်ဆုံး)

**၁. Docker Install လုပ်ရန်:**

```bash
# Docker ထည့်သွင်းရန်
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# NVIDIA Container Toolkit ထည့်သွင်းရန်
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**၂. Isaac Sim Container Pull လုပ်ရန်:**

```bash
# NGC Container ကို pull လုပ်ပါ
docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```

**၃. Container Run လုပ်ရန်:**

```bash
docker run --name isaac-sim --entrypoint bash --gpus all \
  -e "ACCEPT_EULA=Y" -it --rm \
  nvcr.io/nvidia/isaac-sim:4.0.0
```

---

## အဆင့် ၃ - Isaac Lab ထည့်သွင်းခြင်း

### ၃.၁ Conda Environment ပြင်ဆင်ရန်

```bash
# Conda install လုပ်ထားမရှိရင် miniconda ထည့်ပါ
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Environment အသစ်ဖန်တီးပါ
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

pip install --upgrade pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

pip install -e source/isaaclab
pip install git+https://github.com/leggedrobotics/rsl_rl.git
pip install h5py
pip install hydra-core
pip install tensorboard
```

### ၃.၂ Isaac Lab Repository Clone လုပ်ရန်

```bash
# Isaac Lab ကို clone လုပ်ပါ
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Dependencies ထည့်သွင်းပါ
./isaaclab.sh --install
```

### ၃.၃ Isaac Sim Path သတ်မှတ်ရန်

```bash
# .bashrc ဖိုင်ကို ပြင်ပါ
nano ~/.bashrc

# အောက်ပါ line ထည့်ပါ (သင့် Isaac Sim path နဲ့ပြောင်းပါ)
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.0.0"

# Reload bashrc
source ~/.bashrc
```

---

## သတိပြုရန်များ

- **GPU Memory:** အနည်းဆုံး 8GB VRAM လိုအပ်ပါတယ်
- **Disk Space:** အနည်းဆုံး 50GB လွတ်နေရပါမယ်
- **Internet:** Download အတွက် stable internet လိုပါတယ်

---