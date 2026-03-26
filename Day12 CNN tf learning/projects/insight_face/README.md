# InsightFace — Usage Guide

Face Detection, Recognition, Face Swap ကို step by step အသုံးပြုနည်း။

---

## Folder Structure

```
insight_face/
├── README.md
├── setup.md             ← environment setup အသေးစိတ်
├── environment.yml      ← conda env restore
├── models/              ← inswapper_128.onnx ထားမည့် နေရာ
├── data/
│   ├── images/          ← test images
│   ├── videos/          ← test videos
│   └── gallery/         ← face registration images
│       ├── Alice/
│       └── Bob/
├── scripts/
│   ├── detect.py        ← face detection
│   ├── recognize.py     ← face recognition (1:N)
│   └── swap.py          ← face swap
└── similarity_search/   ← vector similarity demos
```

---

## Step 1 — Environment Setup

```bash
# Conda environment ဆောက်ပါ
conda create -n insightface python=3.9 -y
conda activate insightface
```

သို့မဟုတ် environment.yml မှ restore လုပ်ပါ:

```bash
conda env create -f environment.yml
conda activate insightface
```

> ⚡ Setup အသေးစိတ်နှင့် Windows C++ error fix → [setup.md](setup.md) တွင် ကြည့်ပါ

---

## Step 2 — Dependencies Install

```bash
pip install numpy opencv-python-headless Pillow scikit-learn onnx
pip install onnxruntime-gpu    # CPU only ဆိုရင် → onnxruntime
```

---

## Step 3 — InsightFace Install

```bash
# Windows (Cython build skip)
$env:SKIP_CYTHON = "1"
pip install insightface

# Linux / Mac
pip install insightface
```

---

## Step 4 — Model Download (Auto)

Script ကို run လိုက်ပါက `buffalo_l` model ကို **auto-download** လုပ်သည်။  
Path: `C:\Users\<you>\.insightface\models\buffalo_l\`

Manual download လိုပါက → [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)

**Face Swap** အတွက် `inswapper_128.onnx` ကို download ပြီး `models/` ထဲ ထည့်ပါ:  
Download → [InsightFace Releases](https://github.com/deepinsight/insightface/releases)

```
models/
└── inswapper_128.onnx
```

---

## Step 5 — scripts/ ထဲ ဝင်ပါ

```bash
cd scripts
```

---

## Step 6 — Face Detection

Image, Video, Webcam မှ မျက်နှာများ detect လုပ်ပြီး Bounding Box / Landmarks / Age / Gender ကို ထုတ်သည်။

```bash
# Image (result ကို screen ပေါ် ပြသည်)
python detect.py --source ../data/images/test.jpg

# Image (result ကို file အနေဖြင့် save လုပ်သည်)
python detect.py --source ../data/images/test.jpg --save

# Video file
python detect.py --source ../data/videos/test.mp4 --save

# Webcam (index 0)
python detect.py --source 0
```

**Output:**
- Console: bbox, landmarks, age, gender, detection score
- `--save`: `test_detected.jpg` / `test_detected.mp4`

---

## Step 7 — Face Recognition

### 7-A. Gallery Register

`data/gallery/` ထဲမှာ person တစ်ယောက်အတွက် subfolder တစ်ခု ဆောက်ပြီး ဓာတ်ပုံများ ထည့်ပါ:

```
data/gallery/
├── Alice/
│   ├── alice_01.jpg
│   └── alice_02.jpg
└── Bob/
    ├── bob_01.jpg
    └── bob_02.jpg
```

Register လုပ်ပါ (embedding extract + `gallery_embeddings.pkl` save):

```bash
python recognize.py --mode register --gallery ../data/gallery
```

### 7-B. Recognize from Image

```bash
python recognize.py --mode recognize \
    --query ../data/images/test.jpg \
    --gallery ../data/gallery
```

Threshold ညှိချင်ပါက (default: 0.5):

```bash
python recognize.py --mode recognize \
    --query ../data/images/test.jpg \
    --gallery ../data/gallery \
    --threshold 0.6
```

**Output:**
- Console: matched name + cosine similarity score
- `test_recognized.jpg` — bounding box + name label ပါသော image

### 7-C. Real-time Webcam Recognition

```bash
python recognize.py --mode webcam --gallery ../data/gallery
```

> `q` နှိပ်ပါက ပိတ်သည်

---

## Step 8 — Face Swap

> ⚠️ `models/inswapper_128.onnx` မပါဘဲ run မရပါ

### 8-A. Image Swap

```bash
python swap.py \
    --source ../data/images/source.jpg \
    --target ../data/images/target.jpg
```

**Output:** `target_swapped.jpg` + side-by-side preview (Source | Target | Swapped)

### 8-B. Video Swap

```bash
python swap.py \
    --source ../data/images/source.jpg \
    --target ../data/videos/target.mp4
```

**Output:** `target_swapped.mp4`

### 8-C. Real-time Webcam Swap

```bash
python swap.py --source ../data/images/source.jpg --target 0
```

### 8-D. Swap All Faces (default: highest confidence face only)

```bash
python swap.py \
    --source ../data/images/source.jpg \
    --target ../data/images/group.jpg \
    --all-faces
```

---

## Quick Reference

| Task | Command |
|---|---|
| Env activate | `conda activate insightface` |
| Detect (image) | `python detect.py --source img.jpg --save` |
| Detect (webcam) | `python detect.py --source 0` |
| Register gallery | `python recognize.py --mode register --gallery ../data/gallery` |
| Recognize (image)| `python recognize.py --mode recognize --query img.jpg --gallery ../data/gallery` |
| Recognize (webcam)| `python recognize.py --mode webcam --gallery ../data/gallery` |
| Swap (image) | `python swap.py --source src.jpg --target tgt.jpg` |
| Swap (webcam) | `python swap.py --source src.jpg --target 0` |

---

## Similarity Score 참고

| Method | Same Person | Different |
|---|---|---|
| Cosine Similarity | ≥ 0.5 | < 0.5 |
| ArcFace threshold | 0.5 ~ 0.6 (recommended) | — |

> Score 가 낮을수록 → Unknown 처리됩니다. `--threshold` 로 조정하세요.
