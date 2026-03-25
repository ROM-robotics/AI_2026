# CNN Pretrained Models — အသေးစိတ် မှတ်တမ်း

> **Source:** `cnn_pretrained_models.ipynb`
> **Framework:** PyTorch (`torchvision.models`, `facenet-pytorch`)

---

## Table of Contents

1. [ResNet Family](#1-resnet-family)
2. [EfficientNet Family](#2-efficientnet-family)
3. [MobileNet Family](#3-mobilenet-family)
4. [ConvNeXt Family](#4-convnext-family)
5. [InceptionResNetV1 (FaceNet)](#5-inceptionresnetv1-facenet)
6. [All Models Comparison](#6-all-models-comparison)

---

## Overview Table

| Model | Key Innovation | ImageNet Top-1 (approx) |
|-------|---------------|------------------------|
| ResNet-18 / 50 / 152 | Skip Connections (Residual Learning) | 69.8% / 76.1% / 78.3% |
| EfficientNet-B0 / B7 | Compound Scaling (width × depth × resolution) | 77.7% / 84.1% |
| MobileNetV2 / V3-Large | Depthwise Separable Conv, Inverted Residuals | 71.9% / 75.3% |
| ConvNeXt-Tiny / Base | Modernized ConvNet (ViT-inspired design) | 82.1% / 84.1% |
| InceptionResNetV1 (FaceNet) | Inception modules + Residuals + Triplet Loss | Face Embedding |

---

## 1. ResNet Family

### Core Idea — Residual (Skip) Connections

ResNet ရဲ့ အဓိက innovation မှာ **skip connection** (shortcut connection) ပါ။ Deep network တွေမှာ ဖြစ်တတ်တဲ့ **vanishing gradient problem** ကို ဖြေရှင်းဖို့ layer တွေကို bypass လုပ်တဲ့ shortcut path တစ်ခု ထည့်ထားတယ်။

$$y = F(x, \{W_i\}) + x \quad \text{(Residual Connection)}$$

- $x$ = input (identity / skip path)
- $F(x)$ = layer တွေကတဆင့် learned residual
- Layer တွေဟာ $F(x)$ ကိုတိုက်ရိုက် learn မလုပ်ဘဲ **residual** $H(x) - x$ ကိုသာ learn လုပ်ရတာ training ကို ပိုလွယ်ကူစေတယ်

---

### ResNet-18

| Property | Value |
|----------|-------|
| Parameters | ~11.7M |
| Input Size | 224 × 224 |
| Block Type | Basic Block |
| Depth | 18 layers |
| Pretrained Data | ImageNet-1K |

**Architecture Structure:**

```
conv1        → 7×7, stride 2, 64 filters          (Stem)
bn1          → BatchNorm2d(64)
relu         → ReLU
maxpool      → 3×3, stride 2

layer1       → 2× BasicBlock (64 channels)
layer2       → 2× BasicBlock (128 channels, stride 2)
layer3       → 2× BasicBlock (256 channels, stride 2)
layer4       → 2× BasicBlock (512 channels, stride 2)

avgpool      → AdaptiveAvgPool2d(1×1)
fc           → Linear(512, 1000)
```

**Basic Block (ResNet-18/34 use this):**
```
Input
  ↓
Conv2d(3×3) → BN → ReLU
  ↓
Conv2d(3×3) → BN
  ↓ + skip connection (identity)
ReLU
  ↓
Output
```

---

### ResNet-50

| Property | Value |
|----------|-------|
| Parameters | ~25.6M |
| Input Size | 224 × 224 |
| Block Type | Bottleneck Block |
| Depth | 50 layers |
| Pretrained Data | ImageNet-1K |

**Architecture Structure:**

```
conv1        → 7×7, stride 2, 64 filters
bn1 + relu + maxpool

layer1       → 3× Bottleneck (256 channels)
layer2       → 4× Bottleneck (512 channels)
layer3       → 6× Bottleneck (1024 channels)
layer4       → 3× Bottleneck (2048 channels)

avgpool → fc(2048, 1000)
```

**Bottleneck Block (ResNet-50/101/152 use this):**
```
Input (256ch)
  ↓
Conv2d(1×1, 64)   → BN → ReLU    ← dimension reduce
  ↓
Conv2d(3×3, 64)   → BN → ReLU    ← spatial processing
  ↓
Conv2d(1×1, 256)  → BN           ← dimension expand
  ↓ + skip connection
ReLU
  ↓
Output (256ch)
```

> ResNet-18/34 ကြည့်တဲ့ **Basic Block** နဲ့ ကွာချက် — Bottleneck သည် `1×1 → 3×3 → 1×1` pattern ကိုသုံးပြီး parameter ကိုစတိုချ၊ ခြားနားချက် ပိုတွက်နိုင်ဖော်ပေးတယ်။

---

### ResNet-152

| Property | Value |
|----------|-------|
| Parameters | ~60.2M |
| Input Size | 224 × 224 |
| Block Type | Bottleneck Block |
| Depth | 152 layers |
| Pretrained Data | ImageNet-1K |

**Layer counts:**

```
layer1 → 3×  Bottleneck
layer2 → 8×  Bottleneck
layer3 → 36× Bottleneck   ← most layers here
layer4 → 3×  Bottleneck
```

---

### BatchNorm2d နားလည်ချက်

ResNet တွေမှာ BatchNorm2d ကိုကြည့်ရင် ဒီ parameters တွေပါ:

```
BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

| Parameter | ရှင်းလင်းချက် |
|-----------|--------------|
| `eps=1e-05` | Standard deviation ၀ မဖြစ်အောင် denominator မှာ ပေါင်းထည့်တဲ့ ငယ်တဲ့ constant |
| `momentum=0.1` | Running mean/variance ကို update လုပ်တဲ့ rate — batch အသစ်ကို 10%, အဟောင်းကို 90% သိမ်းတယ် |
| `affine=True` | Learnable parameters γ (scale) နဲ့ β (shift) ပါမပါဆုံးဖြတ်ပေး — `True` ဆိုရင် model ကိုယ်တိုင် rescale/shift သင်ယူနိုင်တယ် |
| `track_running_stats=True` | Training မှာ running mean/variance ကိုသိမ်းထားပြီး inference မှာ သုံးတယ် |

---

### ResNet Family Comparison

| Model | Params (M) | Conv+FC Layers | Block Type |
|-------|-----------|----------------|------------|
| ResNet-18 | 11.7M | 20 | Basic Block (2 conv each) |
| ResNet-50 | 25.6M | 53 | Bottleneck (3 conv each) |
| ResNet-152 | 60.2M | 155 | Bottleneck (3 conv each) |

---

## 2. EfficientNet Family

### Core Idea — Compound Scaling

EfficientNet ရဲ့ innovation မှာ model scale လုပ်တဲ့အခါ **width, depth, resolution** သုံးမျိုးကို တစ်ပြိုင်နက် balanced ဖြစ်အောင် scale လုပ်တာပါ (compound scaling)။

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

$$\text{subject to: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \geq 1, \beta \geq 1, \gamma \geq 1$$
```
ဒီနေရာမှာ $\beta$ နဲ့ $\gamma$ ကို ဘာလို့ square ($^2$) တင်ထားတာလဲဆိုတာ စိတ်ဝင်စားဖို့ကောင်းပါတယ်။Depth ($d$): Layer အရေအတွက် ၂ ဆတိုးရင် တွက်ချက်မှု (FLOPS) က ၂ ဆပဲ တိုးပါတယ်။Width ($w$): Channel အရေအတွက် ၂ ဆတိုးရင် တွက်ချက်မှုက ၄ ဆ ($2^2$) တိုးပါတယ်။Resolution ($r$): ပုံရဲ့ pixels အရေအတွက် ၂ ဆတိုးရင်လည်း တွက်ချက်မှုက ၄ ဆ ($2^2$) တိုးပါတယ်။
```
- $\phi$ = compound coefficient (B0=0, B7=6)
- $\alpha, \beta, \gamma$ = NAS ဖြင့်ရှာထားသော optimal ratio

---

### MBConv Block (Mobile Inverted Bottleneck)

EfficientNet ု MobileNet family နှစ်ခုစလုံးမှာ သုံးတာပါ:

```
Input
  ↓
Expand Conv (1×1, pointwise)   ← channels တိုး (expansion factor 1-6×)
  ↓
Depthwise Conv (3×3 or 5×5)    ← per-channel spatial filtering
  ↓
Squeeze-and-Excitation (SE)    ← channel attention mechanism
  ↓
Project Conv (1×1, pointwise)  ← channels လျှော့ပြန်
  ↓ + skip connection (input channels == output channels ဆိုရင်သာ)
Output
```

---

### EfficientNet-B0 (Baseline)

| Property | Value |
|----------|-------|
| Parameters | ~5.3M |
| Input Size | 224 × 224 |
| Top-1 Acc | ~77.7% |
| Scale (φ) | 0 (baseline) |

**Feature Stages:**

```
stage[0]  → Conv2d stem (3→32, stride 2)
stage[1]  → 1× MBConv1  (32→16)
stage[2]  → 2× MBConv6  (16→24, stride 2)
stage[3]  → 2× MBConv6  (24→40, stride 2, 5×5)
stage[4]  → 3× MBConv6  (40→80, stride 2)
stage[5]  → 3× MBConv6  (80→112)
stage[6]  → 4× MBConv6  (112→192, stride 2, 5×5)
stage[7]  → 1× MBConv6  (192→320)
stage[8]  → Conv2d head (320→1280)

classifier → Linear(1280, 1000)
```

---

### EfficientNet-B7 (Largest)

| Property | Value |
|----------|-------|
| Parameters | ~66M |
| Input Size | 600 × 600 |
| Top-1 Acc | ~84.1% |
| Scale (φ) | 6 |

---

### EfficientNet B0 → B7 Scaling Summary

| Model | Input Size | Params (M) | Top-1 Acc | Δ Input (vs prev) | Δ Top-1 (vs prev) |
|-------|-----------|-----------|-----------|-------------------|-------------------|
| B0 | 224 | 5.3M | 77.7% | — | — |
| B1 | 240 | 7.8M | 78.8% | +16 | +1.1% |
| B2 | 260 | 9.1M | 79.8% | +20 | +1.0% |
| B3 | 300 | 12.2M | 81.6% | +40 | +1.8% |
| B4 | 380 | 19.3M | 83.4% | +80 | +1.8% |
| B5 | 456 | 30.4M | 83.7% | +76 | +0.3% |
| B6 | 528 | 43.0M | 84.0% | +72 | +0.3% |
| B7 | 600 | 66.4M | 84.1% | +72 | +0.1% |

**Summary (B0 vs B7):**
- Input Size: 224 → 600 (2.68× larger)
- Params: 5.3M → 66M (~12.5× more)
- Top-1 Accuracy: +6.4% gain

---

### Compound Scaling vs Traditional Scaling

| Scaling Type | ဘာလုပ်တာ | ကန့်သတ်ချက် |
|-------------|---------|------------|
| Depth only | Layer တိုး (ResNet-50→101) | Vanishing gradient ဖြစ်နိုင် |
| Width only | Channel တိုး | ကြီးတဲ့ model ရဖို့ memory မကျေပနာ |
| Resolution only | Input ကြီးစေ | Accuracy gain နည်းတဲ့ဆီမှာ saturate ဖြစ်တယ် |
| **Compound** | သုံးမျိုးကို balanced ဖြစ်အောင် | **EfficientNet ရဲ့နည်းလမ်း** |

---

## 3. MobileNet Family

### Core Idea — Depthwise Separable Convolution

Standard convolution ကို **depthwise conv + pointwise conv** နှစ်ဆင့်ခွဲပြီး computation cost ကို ၈-၉ ဆ လျှော့ချနိုင်တယ်။

**Standard Conv computation cost:**
$$D_K^2 \cdot M \cdot N \cdot D_F^2$$

**Depthwise Separable Conv cost:**
$$D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2$$

**Cost reduction ratio:**
$$\frac{1}{N} + \frac{1}{D_K^2} \approx \frac{1}{8} \text{ to } \frac{1}{9} \quad \text{(3×3 kernels)}$$

Where:
- $D_K$ = kernel size
- $M$ = input channels
- $N$ = output channels  
- $D_F$ = feature map spatial size

---

### MobileNetV2

| Property | Value |
|----------|-------|
| Parameters | ~3.4M |
| Input Size | 224 × 224 |
| Top-1 Acc | ~71.9% |
| Block Type | Inverted Residual (MBConv) |

**Inverted Residual Block:**

ResNet ရဲ့ Bottleneck နဲ့ ပြောင်းပြန်ဖြစ်တယ်:

```
ResNet Bottleneck (Wide→Narrow→Wide):
  256ch → 64ch → 256ch

MobileNet Inverted Residual (Narrow→Wide→Narrow):
  16ch → 96ch → 24ch   ← expand 6×, then compress
```

```
Input (narrow, e.g. 16 channels)
  ↓
Expand Conv (1×1, pointwise)    → 16→96 channels (6× expansion)
  ↓
Depthwise Conv (3×3)            → per-channel, 96 channels
  ↓
Project Conv (1×1, pointwise)   → 96→24 channels (compress)
  ↓ + skip connection (identity, only if stride=1, in_ch==out_ch)
Output
```

**Architecture:**
```
features[0]  → Conv2d stem (3→32, stride 2)
features[1]  → 1× InvertedResidual (32→16, t=1)
features[2-3]  → 2× InvertedResidual (16→24, t=6, stride 2)
features[4-6]  → 3× InvertedResidual (24→32, t=6, stride 2)
features[7-10] → 4× InvertedResidual (32→64, t=6, stride 2)
features[11-13]→ 3× InvertedResidual (64→96, t=6)
features[14-16]→ 3× InvertedResidual (96→160, t=6, stride 2)
features[17]   → 1× InvertedResidual (160→320, t=6)
features[18]   → Conv2d (320→1280, 1×1)

classifier → Dropout → Linear(1280, 1000)
```

---

### MobileNetV3-Large

| Property | Value |
|----------|-------|
| Parameters | ~5.5M |
| Input Size | 224 × 224 |
| Top-1 Acc | ~75.3% |
| Block Type | MBConv + SE + h-swish |

**V3 ကို V2 ထက် ဘာသာ update ဖြစ်တာ:**

| Feature | V2 | V3 |
|---------|----|----|
| Block type | InvertedResidual | InvertedResidual + SE |
| Activation | ReLU6 | h-swish (last layers) |
| Architecture search | Manual | NAS (Neural Architecture Search) |
| Classifier head | Simple | Efficient (h-swish + reduced layers) |

**Squeeze-and-Excitation (SE) Module:**
```
Feature Map (H×W×C)
  ↓
Global Average Pool → (1×1×C)     ← "squeeze"
  ↓
FC → ReLU → FC → Sigmoid          ← channel importance ကို learn
  ↓
Scale original feature map         ← "excitation"
```

**h-swish activation:**
$$h\text{-swish}(x) = x \cdot \frac{ReLU6(x+3)}{6}$$

---

### MobileNet V2 vs V3 Comparison

| Model | Params (M) | Top-1 Acc | Key Addition |
|-------|-----------|-----------|-------------|
| MobileNetV2 | 3.4M | 71.9% | Inverted Residuals |
| MobileNetV3-Small | 2.5M | 67.7% | SE + h-swish + NAS |
| MobileNetV3-Large | 5.5M | 75.3% | SE + h-swish + NAS |

---

## 4. ConvNeXt Family

### Core Idea — Modernized Pure ConvNet

Vision Transformer (ViT) ရဲ့ design idea တွေကို pure ConvNet ထဲ ပြန်ထည့်ပြီး modernize လုပ်ထားတဲ့ architecture ။ 2022 မှာ Meta AI မှ publish လုပ်တဲ့ "A ConvNet for the 2020s" paper ကနေ လာတာ။

---

### ResNet → ConvNeXt: Key Design Changes

| Design Choice | ResNet | ConvNeXt |
|--------------|--------|---------|
| Stem | 7×7 conv, stride 2 + 3×3 maxpool | 4×4 conv, stride 4 (patchify stem, like ViT) |
| Downsampling | Inside residual block | Separate downsampling layer between stages |
| Depthwise conv | — | 7×7 depthwise conv (large kernel) |
| Bottleneck direction | Wide→Narrow→Wide | Narrow→Wide→Narrow (inverted, like MobileNet) |
| Normalization | BatchNorm | LayerNorm |
| Activation | ReLU (multiple per block) | GELU (once per block) |
| Activation placement | After each conv | Fewer activations (ViT inspired) |
| Block ratio | Balanced across stages | More blocks in stage 3 (ViT-like) |

---

### ConvNeXt Block

```
Input
  ↓
Depthwise Conv (7×7)       ← large kernel, same channels
  ↓
LayerNorm
  ↓
Pointwise Conv (1×1)       ← channels ×4 expand
  ↓
GELU activation
  ↓
Pointwise Conv (1×1)       ← channels ÷4 compress
  ↓ + skip connection (with LayerScale)
Output
```

---

### ConvNeXt-Tiny

| Property | Value |
|----------|-------|
| Parameters | ~28.6M |
| Input Size | 224 × 224 |
| Top-1 Acc | ~82.1% |

**Architecture:**
```
features[0]  → Patchify Stem (4×4 conv, stride 4, 3→96ch)

features[1]  → Stage 1: 3× ConvNeXt Block (96ch)
features[2]  → Downsample (LayerNorm + 2×2 conv, 96→192ch)
features[3]  → Stage 2: 3× ConvNeXt Block (192ch)
features[4]  → Downsample (LayerNorm + 2×2 conv, 192→384ch)
features[5]  → Stage 3: 9× ConvNeXt Block (384ch)   ← most blocks
features[6]  → Downsample (LayerNorm + 2×2 conv, 384→768ch)
features[7]  → Stage 4: 3× ConvNeXt Block (768ch)

avgpool → LayerNorm → flatten
classifier → Linear(768, 1000)
```

---

### ConvNeXt-Base

| Property | Value |
|----------|-------|
| Parameters | ~88.6M |
| Input Size | 224 × 224 |
| Top-1 Acc | ~84.1% |

**Channels (Tiny vs Base):**

| Stage | ConvNeXt-Tiny | ConvNeXt-Base |
|-------|--------------|--------------|
| Stage 1 | 96 | 128 |
| Stage 2 | 192 | 256 |
| Stage 3 | 384 | 512 |
| Stage 4 | 768 | 1024 |

---

### ConvNeXt Family Full Lineup

| Model | Channels | Blocks per stage | Params (M) |
|-------|---------|-----------------|-----------|
| ConvNeXt-Tiny | 96/192/384/768 | 3/3/9/3 | 28.6M |
| ConvNeXt-Small | 96/192/384/768 | 3/3/27/3 | 50.2M |
| ConvNeXt-Base | 128/256/512/1024 | 3/3/27/3 | 88.6M |
| ConvNeXt-Large | 192/384/768/1536 | 3/3/27/3 | 197.8M |

---

## 5. InceptionResNetV1 (FaceNet)

### Core Idea — Inception + Residual + Triplet Loss

**FaceNet** (Google, 2015) သည် face recognition အတွက် design လုပ်ထားတဲ့ system ။ Backbone architecture ဖြစ်တဲ့ **InceptionResNetV1/V2** သည် Inception modules (parallel multi-scale convolutions) ကို Residual connections နဲ့ ပေါင်းထားတာ။

---

### Inception Module

```
Input
  ├── 1×1 conv
  ├── 1×1 conv → 3×3 conv
  ├── 1×1 conv → 5×5 conv
  └── 3×3 maxpool → 1×1 conv
        ↓
    Concatenate all outputs
```

Multi-scale features တစ်ပြိုင်နက် extract လုပ်တာကြောင့် network က မည်သည့် scale ကအကောင်းဆုံးဆိုတာ data ကြည့်ပြီး သင်ယူနိုင်တယ်။

---

### FaceNet Architecture — InceptionResNetV1

| Property | Value |
|----------|-------|
| Parameters | ~29.6M |
| Input Size | 160 × 160 |
| Output | 512-dim embedding vector |
| Pretrained Data | VGGFace2 (or CASIA-WebFace) |

**Architecture Flow:**
```
Input (3×160×160)
  ↓
Stem (conv2d_1a → conv2d_4b)      ← initial convolutions
  ↓
Mixed_5b                           ← first Inception block
  ↓
Repeat_1  (5×  Inception-A+Residual)
  ↓
Mixed_6a  (Reduction-A)
  ↓
Repeat_2  (10× Inception-B+Residual)
  ↓
Mixed_7a  (Reduction-B)
  ↓
Repeat_3  (5×  Inception-C+Residual)
  ↓
Block8                             ← final Inception-Residual
  ↓
AvgPool_1a
  ↓
last_linear → Dropout → last_bn → last_linear
  ↓
512-dim L2-normalized face embedding
```

---

### Triplet Loss

FaceNet ၏ training loss ဖြစ်တဲ့ triplet loss:

$$L = \sum_i \left[ \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha \right]_+$$

- $f(x)$ = 512-dim embedding output
- $x^a$ = Anchor (reference face)
- $x^p$ = Positive (same person, different image)
- $x^n$ = Negative (different person)
- $\alpha$ = margin (typically 0.2)

**အလုပ်လုပ်ပုံ:**
- Same person faces ရဲ့ embedding တွေ အနီးကပ်ဖြစ်အောင်
- Different person faces ရဲ့ embedding တွေ အဝေးဆွဲရောက်အောင်

---

### Pretrained Weights

```python
from facenet_pytorch import InceptionResnetV1

# VGGFace2 — general face recognition (recommended)
model = InceptionResnetV1(pretrained='vggface2')

# CASIA-WebFace — alternative training set
model = InceptionResnetV1(pretrained='casia-webface')
```

---

## 6. All Models Comparison

### Parameter Count Summary

| Model | Params (M) | Conv+FC Layers | Designed For |
|-------|-----------|----------------|-------------|
| MobileNetV2 | 3.4M | ~52 | Mobile / Edge |
| EfficientNet-B0 | 5.3M | ~82 | Efficient general |
| MobileNetV3-Large | 5.5M | ~67 | Mobile / Edge |
| ResNet-18 | 11.7M | 20 | General baseline |
| ResNet-50 | 25.6M | 53 | General (standard) |
| ConvNeXt-Tiny | 28.6M | ~37 | Modern general |
| FaceNet (IncResV1) | 29.6M | ~213 | Face recognition |
| ResNet-152 | 60.2M | 155 | High accuracy |
| EfficientNet-B7 | 66M | ~332 | High accuracy |
| ConvNeXt-Base | 88.6M | ~101 | High accuracy |

---

### Family-wise Key Innovations

```
ResNet     → Skip Connections (2015)
             → Solved vanishing gradient with identity shortcuts

MobileNet  → Depthwise Separable Conv (V1, 2017)
             → Inverted Residuals (V2, 2018)
             → SE + h-swish + NAS (V3, 2019)

EfficientNet → Compound Scaling (2019)
               → Balanced width/depth/resolution scaling

ConvNeXt   → ViT-inspired ConvNet (2022)
             → 7×7 depthwise, LayerNorm, GELU, patchify stem

FaceNet    → Triplet Loss + 512-dim embedding (2015)
             → InceptionResNetV1/V2 backbone
```

---

### Accuracy vs Efficiency Tradeoffs

```
High Accuracy, More Params:
  ConvNeXt-Base (88.6M) ~84.1%
  EfficientNet-B7 (66M) ~84.1%
  ResNet-152 (60.2M)    ~78.3%

Balanced:
  ConvNeXt-Tiny (28.6M) ~82.1%
  EfficientNet-B0 (5.3M) ~77.7%
  ResNet-50 (25.6M)      ~76.1%

Lightweight / Mobile:
  MobileNetV3-Large (5.5M) ~75.3%
  MobileNetV2 (3.4M)       ~71.9%
```

---

### PyTorch Weight Versions

Notebook ထဲမှာ `.DEFAULT` ကိုသုံးထားတယ် — PyTorch team ရဲ့ recommended best weights:

```python
# V1 = original pretrained weights
# V2 = improved training recipe (same architecture, better accuracy)
# DEFAULT = latest/best recommendation

# Example:
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# DEFAULT == IMAGENET1K_V2 (better training recipe)
```

**V2 weights ဘာကြောင့် ကောင်းတာ:**
- Longer training (more epochs)
- Better data augmentation (MixUp, CutMix, etc.)
- Better optimizer (SGD with cosine LR)
- Same architecture, higher accuracy

---

## Quick Reference — Usage

```python
import torchvision.models as models

# ResNet
resnet18  = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet50  = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

# EfficientNet
effnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
effnet_b7 = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)

# MobileNet
mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet_v3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# ConvNeXt
convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
convnext_base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)

# FaceNet
from facenet_pytorch import InceptionResnetV1
facenet = InceptionResnetV1(pretrained='vggface2')
```
