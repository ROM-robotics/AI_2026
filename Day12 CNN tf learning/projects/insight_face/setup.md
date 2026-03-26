# InsightFace Setup Guide

## 1. Conda Environment ဆောက်ခြင်း

```bash
conda create -n insightface python=3.9 -y
conda activate insightface
```

---

## 2. PyTorch & CUDA Install

CUDA version ကိုကြည့်ပြီး သင့်တော်တဲ့ version ရွေးပါ။

```bash
# CUDA 11.8 အတွက်
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.6 အတွက်
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip install torch torchvision torchaudio
```

CUDA ရှိမရှိ စစ်ဆေးရန်:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

---

## 3. InsightFace Dependencies Install

```bash
pip install numpy opencv-python-headless Pillow scikit-learn
pip install onnx onnxruntime-gpu   # GPU မရှိရင် onnxruntime သာ သုံးပါ
```

> **Note:** `onnxruntime-gpu` သည် CUDA ရှိမှသာ အလုပ်လုပ်သည်။ CPU only ဆိုရင် `onnxruntime` ကိုသာ install လုပ်ပါ။

---

## 4. InsightFace Install

```bash
pip install insightface
```

Build from source လုပ်လိုပါက:

```bash
pip install cython
git clone https://github.com/deepinsight/insightface.git
cd insightface/python-package
pip install -e .
```

---

## ⚠️ Windows Error: "Microsoft Visual C++ 14.0 or greater is required"

### ဘာကြောင့် ဖြစ်တာ?

InsightFace ၏ `face3d` Cython extension (`mesh_core_cython.pyx`) ကို compile လုပ်ရန် **MSVC C++ compiler** လိုအပ်သည်။ Windows မှာ default အနေဖြင့် မပါဘဲ error ဖြစ်သည်။

---

### ✅ Step A — Microsoft C++ Build Tools Install (Recommended)

1. [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/) သို့ ဝင်ပါ
2. **"Build Tools for Visual Studio"** ကို download လုပ်ပါ
3. Install လုပ်ရာတွင် **"Desktop development with C++"** workload ကို ရွေးပါ
4. Install ပြီးလျှင် terminal ကို restart ပြီး ထပ်ကြိုးစားပါ:

```bash
conda activate insightface
pip install insightface
```

---


### ✅ Step B — Pre-built Wheel သုံး

insightface ၏ specific version အတွက် pre-built wheel ကို download ပြီး install လုပ်နိုင်သည်:

```bash
# version 0.7.3 wheel (Python 3.9, Windows x64)
pip install insightface==0.7.3 --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Pretrained Model Download

InsightFace သည် model ကို run time မှာ auto-download လုပ်သည်။ Default path: `~/.insightface/models/`

Manual download လုပ်လိုပါက [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) မှ ရယူနိုင်သည်။

> **Note:** `inswapper_128.onnx` မော်ဒယ်ကို [insightface releases](https://github.com/deepinsight/insightface/releases) မှ ရယူပါ။

---

## Environment Export / Share

```bash
# environment.yml ထုတ်ရန်
conda env export > environment.yml

# တခြားစက်မှာ restore လုပ်ရန်
conda env create -f environment.yml
```

## Folder Structure (Recommended)

```
insight_face/
├── setup.md
├── environment.yml
├── models/              ← onnx model files
├── data/                ← test images / videos
└── scripts/
    ├── detect.py
    ├── recognize.py
    └── swap.py
```
