# Scaling Laws for CNN Classification Models

## Scaling Laws ဆိုတာဘာလဲ?

Scaling law ဆိုတာ model performance (loss) ကို **model size (N)**, **dataset size (D)**, နဲ့ **compute budget (C)** တို့နဲ့ ဆက်စပ်ပြီး ခန့်မှန်းနိုင်တဲ့ power-law relationship ဖြစ်ပါတယ်။

$$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$$

- $N$ = model parameters အရေအတွက်
- $D$ = training dataset size (samples)
- $L_\infty$ = irreducible loss (data ရဲ့ inherent noise)
- $\alpha_N, \alpha_D$ = scaling exponents
- $N_c, D_c$ = characteristic constants

---

## CNN Classification Models မှာ Scaling ဘယ်လိုသက်ရောက်သလဲ

CNN classification (e.g. ResNet, EfficientNet, ConvNeXt) မှာ performance ကို တိုးတက်စေတဲ့ axes ၃ ခုရှိပါတယ်:

| Axis | ဘာကိုပြောင်းသလဲ | ဥပမာ |
|---|---|---|
| **Depth** (layers) | network ရဲ့ အတိမ်အနက် | ResNet-18 → ResNet-152 |
| **Width** (channels) | layer တစ်ခုချင်းစီရဲ့ filter အရေအတွက် | 64 → 256 channels |
| **Resolution** (input size) | input image ရဲ့ pixel resolution | 224×224 → 380×380 |

Parameters $N$ ဟာ depth နဲ့ width နှစ်ခုလုံးနဲ့ တိုက်ရိုက်ဆက်နွယ်ပါတယ်:

$$N \propto \text{depth} \times \text{width}^2$$

---

## Dataset နဲ့ Model Parameters ကို Balance ထိန်းနည်း

### 1. Chinchilla-Style Optimal Ratio

Hoffmann et al. (2022) က compute-optimal training အတွက် data နဲ့ parameters ကို roughly **linear ratio** နဲ့ scale လုပ်ရမယ်လို့ ပြထားပါတယ်:

$$D_{\text{opt}} \approx 20 \times N$$

CNN classification မှာလည်း ဒီ principle ကို apply လုပ်နိုင်ပါတယ်:

| Model Parameters ($N$) | Recommended Dataset Size ($D$) | ဥပမာ Model |
|---|---|---|
| ~1M | ~20M samples | Small MobileNet |
| ~11M | ~220M samples | ResNet-18 |
| ~25M | ~500M samples | ResNet-50 |
| ~60M | ~1.2B samples | ResNet-152 / EfficientNet-B7 |

> **Note:** Vision models တွေမှာ actual ratio ဟာ 20:1 ထက်နည်းနိုင်ပါတယ် (data augmentation နဲ့ transfer learning ကြောင့်)။ Empirically CNN models တွေမှာ $D \approx 5N$ to $10N$ ပတ်ဝန်းကျင်မှာ diminishing returns စတွေ့ရပါတယ်။

### 2. Under-fitting vs Over-fitting Diagnosis

Dataset-model balance ကို diagnose လုပ်ဖို့:

```
If train_loss ≫ target   → Model too small (under-parameterized)
                           → Increase N (deeper/wider model)

If train_loss ≪ val_loss → Model too large for dataset (over-parameterized)
                           → Increase D (more data) or reduce N
                           → Apply regularization (dropout, weight decay, augmentation)

If train_loss ≈ val_loss ≈ target → Well balanced ✓
```

### 3. Practical Scaling Strategy

#### Step A: Fixed Compute Budget ရှိရင်

Compute budget $C$ (FLOPs) ကို model size နဲ့ training tokens/samples ကြားမှာ ခွဲဝေပါ:

$$C \approx 6 \times N \times D$$

ဒါကြောင့် $C$ fixed ဖြစ်ရင်:
- $N$ ကိုတိုးရင် $D$ ကိုလျှော့ရမယ်
- $D$ ကိုတိုးရင် $N$ ကိုလျှော့ရမယ်

Optimal allocation: $N$ နဲ့ $D$ နှစ်ခုလုံးကို $\sqrt{C}$ နဲ့ proportional ဖြစ်အောင် scale လုပ်ပါ။

#### Step B: EfficientNet-Style Compound Scaling

EfficientNet (Tan & Le, 2019) က CNN dimensions တွေကို uniformly scale လုပ်ဖို့ compound coefficient $\phi$ ကိုသုံးပါတယ်:

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

Constraint: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (FLOPs roughly doubles per $\phi$ step)

EfficientNet ရဲ့ discovered values:
- $\alpha = 1.2, \quad \beta = 1.1, \quad \gamma = 1.15$

ဒီ method ဟာ depth/width/resolution ကို independently scale မလုပ်ဘဲ **balanced** ဖြစ်အောင် ထိန်းပေးပါတယ်။

#### Step C: Dataset Scaling Strategy

Model ကို scale up လုပ်တဲ့အခါ dataset ကိုလည်း match ဖြစ်အောင်:

1. **Real data collection** — labeled data ပိုရှာပါ
2. **Data augmentation** — effective dataset size ကို $k$ ဆ တိုးပါ (e.g., RandAugment, CutMix, MixUp)
3. **Pre-training on larger data** — ImageNet-21k, JFT-300M, LAION စတဲ့ large-scale datasets ပေါ်မှာ pre-train ပြီး downstream task ပေါ် fine-tune ပါ
4. **Synthetic data / self-supervised** — unlabeled data ကို leverage လုပ်ပါ

---

## Scaling Behavior: Empirical Observations (CNN Classification)

```
Performance
(Accuracy)
   ↑
   │          ╭────────────────── Data-limited regime
   │        ╱                     (model ကြီးပေမယ့် data မလုံလောက်)
   │      ╱
   │    ╱  ← Optimal frontier
   │  ╱       (N နဲ့ D balanced)
   │╱
   │╱  ← Model-limited regime
   │     (data ရှိပေမယ့် model ငယ်)
   └────────────────────────→ Compute (FLOPs)
```

Key observations:
- **Larger models are more sample-efficient** — parameter ပိုများတဲ့ model ဟာ sample တစ်ခုချင်းစီကနေ information ပိုထုတ်ယူနိုင်တယ်
- **Diminishing returns** — parameters ဒါမှမဟုတ် data တစ်ခုတည်းကို ထပ်တိုးရင် log-linear ပဲတက်တယ်
- **Balanced scaling wins** — $N$ နဲ့ $D$ နှစ်ခုလုံးကို တပြိုင်နက်တိုးမှ compute-efficient ဖြစ်တယ်

---

## Quick Reference: Scaling Decision Table

| Situation | Action |
|---|---|
| Dataset ငယ်တယ်, model ကြီးတယ် | Regularize (augmentation, dropout, weight decay) ဒါမှမဟုတ် model ချုံ့ |
| Dataset ကြီးတယ်, model ငယ်တယ် | Model scale up (deeper + wider) |
| နှစ်ခုလုံး ချိန်ညှိချင်တယ် | Compound scaling ($\phi$) သုံး၊ $D \approx 5\text{-}20 \times N$ ကို target လုပ် |
| Compute limited | $C \approx 6ND$ ကနေ optimal $N, D$ allocation ရှာ |
| Pre-trained model ကိုသုံးမယ် | Fine-tuning ဆိုရင် dataset ငယ်လည်းရတယ် (effective $D$ ကြီးပြီးသား) |

---

## References

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020)
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022)
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs* (ICML 2019)
- Zhai et al., *Scaling Vision Transformers* (2022) — CNN/ViT scaling comparisons
- Rosenfeld et al., *A Constructive Prediction of the Generalization Error Across Scales* (ICLR 2020)
