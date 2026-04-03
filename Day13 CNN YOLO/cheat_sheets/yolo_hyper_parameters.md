# YOLO User-Tunable Hyperparameters

ဒီ note က YOLO ကို version မခွဲဘဲ, user က tuning လုပ်လို့ရတဲ့ hyperparameters တွေကို task အမျိုးအစားအလိုက် ခွဲရေးထားတာပါ။

ဒီမှာ "model အမျိုးအစား" ဆိုတာကို task type အလိုက် ခွဲထားပါတယ်:

- Detect models
- Segment models
- Classify models
- Pose models
- OBB models

Examples:

- Detect: `yolo11n.pt`, `yolov8n.pt`, `best.pt`
- Segment: `yolo11n-seg.pt`, `yolov8n-seg.pt`, `best-seg.pt`
- Classify: `yolo11n-cls.pt`, `yolov8n-cls.pt`, `best-cls.pt`
- Pose: `yolo11n-pose.pt`, `yolov8n-pose.pt`, `best-pose.pt`
- OBB: `yolo11n-obb.pt`, `yolov8n-obb.pt`, `best-obb.pt`

## 1. Important Note

YOLO hyperparameters တချို့က version အလိုက် အနည်းငယ်ကွာနိုင်ပါတယ်။ ဒါပေမယ့် user tuning လုပ်တဲ့အခါ အမှန်တကယ် အများဆုံးထိရောက်တာက:

- common training settings
- augmentation settings
- task-specific loss weights
- predict-time thresholds

ဒီ note ထဲက content ကို Ultralytics-style YOLO workflow အတွက် version-agnostic reference အနေနဲ့ သုံးလို့ရပါတယ်။

## 2. Common Hyperparameters for All YOLO Models

အောက်က parameters တွေက detect / segment / classify / pose / obb models အများစုမှာ user က အဓိက tuning လုပ်လေ့ရှိတဲ့ shared hyperparameters တွေပါ။

| Hyperparameter | Meaning | Typical Use |
| --- | --- | --- |
| `epochs` | training epoch count | underfit ဖြစ်ရင်တိုး, overfit ဖြစ်ရင်လျှော့ |
| `batch` | batch size | GPU memory အလိုက် tune |
| `imgsz` | input image size | small objects / fine details များရင်တိုး |
| `optimizer` | optimizer type | `auto`, `SGD`, `AdamW`, `MuSGD` စမ်းနိုင် |
| `lr0` | initial learning rate | unstable ဖြစ်ရင်လျှော့ |
| `lrf` | final LR fraction | long training မှာ scheduler shape ထိန်း |
| `momentum` | optimizer momentum | SGD-family stability tuning |
| `weight_decay` | L2 regularization | overfit လျှော့ရန် |
| `warmup_epochs` | warmup period | training အစပိုင်း stabilize လုပ်ရန် |
| `warmup_momentum` | warmup momentum | warmup shape tune |
| `warmup_bias_lr` | warmup bias LR | early epochs stabilize |
| `patience` | early stopping patience | validation plateau ကိုစောင့်ရန် |
| `cos_lr` | cosine LR schedule | smooth LR decay လိုရင် |
| `close_mosaic` | last N epochs mosaic ပိတ် | final epochs stabilize |
| `amp` | mixed precision training | memory/save speed |
| `cache` | RAM/disk cache | training speed မြှင့် |
| `workers` | dataloader workers | input pipeline speed tune |
| `device` | CPU/GPU selection | runtime control |
| `freeze` | freeze first N layers | transfer learning / small dataset |
| `fraction` | dataset fraction | quick experiments |
| `seed` | reproducibility | repeatable experiment |
| `deterministic` | deterministic ops | reproducibility priority |
| `project`, `name` | run naming | experiment management |

### Common CLI example

```powershell
yolo detect train model=your_detect_model.pt data=data.yaml epochs=100 batch=16 imgsz=640 optimizer=auto lr0=0.01 weight_decay=0.0005 device=0
```

## 3. Common Augmentation Hyperparameters

ဒီ knobs တွေက model quality ကို အရမ်းသက်ရောက်တတ်ပါတယ်။ Dataset quality နိမ့်တဲ့အခါ augmentation tuning က learning rate tuning လောက်တောင် အရေးကြီးတတ်ပါတယ်။

| Hyperparameter | Tasks | What it changes |
| --- | --- | --- |
| `hsv_h`, `hsv_s`, `hsv_v` | all tasks | color jitter |
| `degrees` | detect, segment, pose, obb | random rotation |
| `translate` | detect, segment, pose, obb | spatial shift |
| `scale` | all tasks | zoom in/out |
| `shear` | detect, segment, pose, obb | geometric shear |
| `perspective` | detect, segment, pose, obb | perspective warp |
| `flipud`, `fliplr` | all tasks | vertical / horizontal flip |
| `mosaic` | detect, segment, pose, obb | 4-image mosaic |
| `mixup` | detect, segment, pose, obb | blended images |
| `cutmix` | detect, segment, pose, obb | partial mix |
| `copy_paste` | segment | paste objects across images |
| `copy_paste_mode` | segment | `flip`, `mixup` |
| `auto_augment` | classify | `randaugment`, `autoaugment`, `augmix` |
| `erasing` | classify | random erasing |

### Practical tuning hints

- small dataset ဆိုရင် augmentation ကို တိုးစမ်းပါ
- already noisy dataset ဆိုရင် `mosaic`, `mixup`, `cutmix` ကိုလျှော့စမ်းပါ
- aerial / rotated objects dataset တွေမှာ `degrees` နဲ့ `scale` က အထူးအရေးကြီးတယ်
- classification မှာ `auto_augment` နဲ့ `erasing` က effect ကြီးတတ်တယ်

## 4. Detect Models

ဒီ section က object detection YOLO models အတွက်ဖြစ်ပါတယ်။

### Detect အတွက်အဓိက tuning လုပ်ရမယ့် hyperparameters

| Hyperparameter | Why it matters |
| --- | --- |
| `imgsz` | small objects များရင် accuracy တိုးနိုင် |
| `batch` | gradient stability vs VRAM |
| `lr0` | convergence speed |
| `optimizer` | training stability and speed |
| `box` | localization emphasis |
| `cls` | classification emphasis |
| `dfl` | box regression detail in versions that use DFL |
| `mosaic` | crowded scenes/generalization |
| `mixup`, `cutmix` | regularization |
| `close_mosaic` | final epoch stabilization |
| `multi_scale` | variable-size robustness |
| `single_cls` | all objects as one class |
| `classes` | selected classes only |

### Detect-specific predict tuning

| Hyperparameter | Use |
| --- | --- |
| `conf` | false positive များရင်တိုး |
| `iou` | duplicate suppression tuning |
| `max_det` | dense scenes |
| `agnostic_nms` | class-agnostic duplicate removal |
| `end2end` | end-to-end/NMS-free models only |

### Detect CLI template

```powershell
yolo detect train model=your_detect_model.pt data=data.yaml epochs=150 imgsz=640 batch=16 lr0=0.005 optimizer=auto box=7.5 cls=0.5 mosaic=1.0 close_mosaic=10 device=0
```

### Detect tuning advice

- small objects များရင် `imgsz=800` သို့မဟုတ် `imgsz=960` စမ်းပါ
- false negatives များရင် `imgsz` တိုး, `mosaic` ထိန်း, dataset labels စစ်
- unstable training ဖြစ်ရင် `lr0` လျှော့, `warmup_epochs` တိုး

## 5. Segment Models

ဒီ section က instance segmentation YOLO models အတွက်ဖြစ်ပါတယ်။

### Segment အတွက်အဓိက hyperparameters

| Hyperparameter | Why it matters |
| --- | --- |
| `imgsz` | mask detail quality |
| `batch` | memory demand high ဖြစ်တတ် |
| `lr0`, `optimizer` | convergence |
| `box`, `cls` | detection side quality |
| `overlap_mask` | overlap masks training behavior |
| `mask_ratio` | mask training resolution |
| `copy_paste` | object count/diversity တိုး |
| `copy_paste_mode` | paste behavior |
| `mosaic`, `mixup` | generalization |

### Segment predict tuning

| Hyperparameter | Use |
| --- | --- |
| `conf` | weak masks များလျှင်တိုး |
| `retina_masks` | original-size masks လိုရင် `True` |
| `show_boxes` | masks only ကြည့်ချင်ရင် `False` |

### Segment CLI template

```powershell
yolo segment train model=your_segment_model.pt data=data.yaml epochs=150 imgsz=640 batch=8 lr0=0.005 optimizer=auto mask_ratio=4 overlap_mask=True copy_paste=0.2 mosaic=1.0 device=0
```

### Segment tuning advice

- mask edges မကောင်းရင် `imgsz` တိုးစမ်းပါ
- thin objects များရင် `mask_ratio` ကိုပိုသတိထားပြီး tune လုပ်ပါ
- overlap များတဲ့ scene တွေမှာ `overlap_mask` behavior ကို compare run လုပ်ပါ

## 6. Classify Models

ဒီ section က image classification YOLO models အတွက်ဖြစ်ပါတယ်။

### Classify အတွက်အဓိက hyperparameters

| Hyperparameter | Why it matters |
| --- | --- |
| `imgsz` | input crop size |
| `batch` | throughput and stability |
| `lr0`, `weight_decay` | generalization |
| `dropout` | overfitting control |
| `auto_augment` | visual diversity |
| `erasing` | robustness to missing regions |
| `fliplr`, `flipud` | symmetry assumptions |
| `hsv_h`, `hsv_s`, `hsv_v` | color robustness |

### Classification-specific notes

- class imbalance ရှိရင် data balancing ကို hyperparameter tuning ထက်အရင်စဉ်းစားပါ
- extreme aspect ratio images တွေမှာ default crop pipeline က informative region ကိုဖြတ်ပစ်နိုင်လို့ custom dataset/resize pipeline စဉ်းစားရမယ်
- overfit မြန်ရင် `dropout`, `weight_decay`, `erasing` တိုးစမ်းပါ

### Classify CLI template

```powershell
yolo classify train model=your_classify_model.pt data=dataset_cls epochs=100 imgsz=224 batch=64 lr0=0.001 optimizer=AdamW dropout=0.2 auto_augment=randaugment erasing=0.25 device=0
```

### Classify tuning advice

- small dataset ဆို `dropout=0.1` to `0.3` စမ်းနိုင်
- image detail အများကြီးလိုရင် `imgsz=256` or `320` စမ်းနိုင်
- training slow overfit မဖြစ်ဘဲ underfit ဖြစ်ရင် `dropout` လျှော့ပါ

## 7. Pose Models

ဒီ section က pose estimation YOLO models အတွက်ဖြစ်ပါတယ်။

### Pose အတွက်အဓိက hyperparameters

| Hyperparameter | Why it matters |
| --- | --- |
| `imgsz` | keypoint localization precision |
| `batch` | memory and convergence |
| `lr0`, `optimizer` | convergence |
| `pose` | keypoint loss weight |
| `kobj` | keypoint objectness balance |
| `rle` | keypoint localization refinement |
| `degrees`, `scale`, `translate` | human pose variation robustness |
| `fliplr` | left-right symmetry learning |
| `mosaic` | sometimes helpful, but too strong can hurt fine joints |

### Pose CLI template

```powershell
yolo pose train model=your_pose_model.pt data=pose_data.yaml epochs=150 imgsz=640 batch=16 lr0=0.005 optimizer=auto pose=12.0 kobj=1.0 rle=1.0 fliplr=0.5 mosaic=0.5 device=0
```

### Pose tuning advice

- keypoints jitter ဖြစ်ရင် `imgsz` တိုးစမ်းပါ
- body joints fine detail မဖမ်းနိုင်ရင် augmentation ကိုလျှော့ပြီး `pose` weight ကိုစမ်းပါ
- heavily cropped human images dataset မှာ `translate` များလွန်းရင် accuracy ကျနိုင်တယ်

## 8. OBB Models

ဒီ section က oriented bounding box YOLO models အတွက်ဖြစ်ပါတယ်။

### OBB အတွက်အဓိက hyperparameters

| Hyperparameter | Why it matters |
| --- | --- |
| `imgsz` | aerial / remote sensing detail အတွက် အရေးကြီး |
| `batch` | large-resolution training memory |
| `lr0`, `optimizer` | convergence |
| `box`, `cls` | box/class balance |
| `angle` | rotated box angle precision |
| `degrees` | orientation robustness |
| `scale`, `translate`, `perspective` | viewpoint robustness |
| `mosaic` | scene diversity |
| `close_mosaic` | training stabilization |

### OBB-specific notes

- OBB datasets တွေမှာ angle quality က annotation quality နဲ့အရမ်းဆိုင်တယ်
- OBB task တွေမှာ image size ကို detect task ထက် ပိုကြီးစွာသုံးတတ်တယ်
- version တချို့မှာ internal representation က `xywhr` style rotated box format ဖြစ်တတ်တယ်

### OBB CLI template

```powershell
yolo obb train model=your_obb_model.pt data=dota_like.yaml epochs=150 imgsz=1024 batch=8 lr0=0.005 optimizer=auto angle=1.0 degrees=10 scale=0.5 perspective=0.0005 mosaic=1.0 device=0
```

### OBB tuning advice

- aerial imagery မှာ `imgsz=1024` သို့မဟုတ်ပိုမြင့်တဲ့ size က often effective ဖြစ်တယ်
- angle prediction မတိကျရင် `angle`, `degrees`, dataset annotation quality ကိုအရင်စစ်ပါ
- tiny rotated objects များရင် larger `imgsz` က အများဆုံးအကျိုးရှိတတ်တယ်

## 9. Model Scale-wise Practical Tuning

Task မတူပေမယ့် model scale `n/s/m/l/x` အလိုက် tuning strategy လည်း ကွာတတ်ပါတယ်။

| Scale | Practical strategy |
| --- | --- |
| `n` | quick experiments, large batch, faster iteration |
| `s` | balanced baseline |
| `m` | good production starting point |
| `l` | stronger accuracy, lower batch often needed |
| `x` | max accuracy, highest VRAM use, more careful LR/batch tuning |

### Suggested starting points

| Scale | Suggested `batch` | Suggested use |
| --- | --- | --- |
| `n` | 16 to 64 | fast ablation, weak GPU |
| `s` | 16 to 32 | good first serious run |
| `m` | 8 to 24 | balanced training |
| `l` | 4 to 16 | stronger GPU |
| `x` | 2 to 8 | highest-memory setup |

Note:

- exact batch size က GPU VRAM, task type, imgsz ပေါ်မူတည်ပြီး အရမ်းကွာနိုင်တယ်
- segment / pose / obb models တွေမှာ same scale even detect ထက် memory ပိုသုံးတတ်တယ်

## 10. Train / Predict / Export Hyperparameters by Stage

User tuning လုပ်လို့ရတဲ့ hyperparameters တွေကို stage အလိုက်ပြန်ခွဲရင်:

### Train stage

- `epochs`, `batch`, `imgsz`, `optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`
- `warmup_epochs`, `warmup_momentum`, `warmup_bias_lr`
- `box`, `cls`, `dfl`, `pose`, `kobj`, `rle`, `angle`
- `mosaic`, `mixup`, `cutmix`, `copy_paste`, `auto_augment`, `erasing`
- `close_mosaic`, `multi_scale`, `freeze`, `fraction`, `amp`

### Predict stage

- `conf`, `iou`, `imgsz`, `device`, `max_det`
- `agnostic_nms`, `end2end`, `retina_masks`
- `show`, `save`, `save_txt`, `save_conf`, `line_width`

### Export stage

- `format`, `imgsz`, `half`, `int8`, `dynamic`, `simplify`, `opset`, `workspace`, `batch`, `device`, `end2end`

## 11. Recommended Tuning Order

Hyperparameters အားလုံးတစ်ခါတည်းမပြင်ဘဲ ဒီ order နဲ့ tune လုပ်ရင် ပိုကောင်းပါတယ်:

1. `imgsz`, `batch`, `epochs`
2. `lr0`, `optimizer`, `weight_decay`
3. task-specific loss weights (`box`, `cls`, `dfl`, `pose`, `angle`, `dropout`, `mask_ratio`)
4. augmentation knobs (`mosaic`, `mixup`, `copy_paste`, `auto_augment`, `erasing`)
5. predict-stage thresholds (`conf`, `iou`, `max_det`)

## 12. Example Command Templates by Task

### Detect

```powershell
yolo detect train model=your_detect_model.pt data=data.yaml epochs=100 imgsz=640 batch=16 lr0=0.005 optimizer=auto box=7.5 cls=0.5 mosaic=1.0 device=0
```

### Segment

```powershell
yolo segment train model=your_segment_model.pt data=data.yaml epochs=100 imgsz=640 batch=8 lr0=0.005 mask_ratio=4 copy_paste=0.2 device=0
```

### Classify

```powershell
yolo classify train model=your_classify_model.pt data=dataset_cls epochs=100 imgsz=224 batch=64 lr0=0.001 dropout=0.2 auto_augment=randaugment erasing=0.25 device=0
```

### Pose

```powershell
yolo pose train model=your_pose_model.pt data=pose_data.yaml epochs=100 imgsz=640 batch=16 lr0=0.005 pose=12.0 kobj=1.0 device=0
```

### OBB

```powershell
yolo obb train model=your_obb_model.pt data=dota_like.yaml epochs=100 imgsz=1024 batch=8 lr0=0.005 angle=1.0 degrees=10 device=0
```

## 13. Short Summary

အလွယ်ပြောရရင်:

- Detect: `imgsz`, `box`, `cls`, `conf`, `mosaic`
- Segment: `mask_ratio`, `overlap_mask`, `copy_paste`
- Classify: `dropout`, `auto_augment`, `erasing`, `imgsz`
- Pose: `pose`, `kobj`, `rle`, `imgsz`
- OBB: `angle`, `degrees`, `imgsz`, `perspective`

Common tuning backbone ကတော့:

- `batch`
- `lr0`
- `optimizer`
- `weight_decay`
- `epochs`
- `imgsz`

ဒီ 6 ခုက almost every YOLO experiment မှာ first priority tuning knobs ဖြစ်ပါတယ်။
