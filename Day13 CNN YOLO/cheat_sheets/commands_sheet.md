# YOLO Daily Use Commands

Ultralytics YOLO ကို command line ကနေ daily use အတွက် အများဆုံးသုံးရမယ့် commands တွေကို အောက်မှာ စုစည်းထားပါတယ်။

## 1. Install / Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install ultralytics
```

GPU ရှိမရှိ စစ်ရန်:

```powershell
yolo checks
```

## 2. Dataset Structure

ပုံမှန် custom detection dataset structure:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
		├── train/
		└── val/
```

`data.yaml` example:

```yaml
path: dataset
train: images/train
val: images/val
names:
	0: person
	1: helmet
```

## 3. Training Commands

Pretrained model နဲ့ train:

```powershell
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 device=0
```

CPU ပေါ်မှာ train:

```powershell
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=8 device=cpu
```

Custom run name / output folder နဲ့ train:

```powershell
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 project=runs/train name=my_exp device=0
```

Training resume:

```powershell
yolo detect train resume model=runs/detect/train/weights/last.pt
```

Cache သုံးပြီး train မြန်အောင်:

```powershell
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 cache=True device=0
```

## 4. Validation / Evaluation

Best weights နဲ့ validation run:

```powershell
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640 device=0
```

mAP, precision, recall စစ်ဖို့ အများဆုံးဒီ command ကိုသုံးပါတယ်။

## 5. Inference Commands

Single image inference:

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=demo.jpg conf=0.25
```

Folder inference:

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=images/
```

Video inference:

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=video.mp4
```

Webcam inference:

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=0
```

Result image/video save + label txt save:

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=images/ save=True save_txt=True
```

Confidence threshold ပြောင်းရန်:

```powershell
yolo detect predict model=runs/detect/train/weights/best.pt source=images/ conf=0.5
```

## 6. Export Commands

ONNX export:

```powershell
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

TensorRT export:

```powershell
yolo export model=runs/detect/train/weights/best.pt format=engine device=0
```

## 7. Useful Daily Commands

Model summary / info:

```powershell
yolo detect predict model=yolo11n.pt source=demo.jpg
```

Environment / package checks:

```powershell
yolo checks
```

Ultralytics package update:

```powershell
pip install -U ultralytics
```

## 8. Common Path Examples

Windows path example:

```powershell
yolo detect train data=C:/Users/RAIDER/Desktop/ai/vision.ai/yolo/data.yaml model=yolo11n.pt epochs=100 imgsz=640 device=0
```

## 9. Quick Reference

```powershell
# train
yolo detect train data=data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 device=0

# val
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640

# image inference
yolo detect predict model=runs/detect/train/weights/best.pt source=demo.jpg conf=0.25

# video inference
yolo detect predict model=runs/detect/train/weights/best.pt source=video.mp4

# webcam inference
yolo detect predict model=runs/detect/train/weights/best.pt source=0

# export
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

## 10. Notes

- `detect` ကို `segment`, `classify`, `pose` နဲ့ အစားထိုးသုံးနိုင်ပါတယ်။
- `device=0` ဆိုတာ first GPU, `device=cpu` ဆိုတာ CPU only ဖြစ်ပါတယ်။
- output တွေကို ပုံမှန်အားဖြင့် `runs/detect/...` အောက်မှာ သိမ်းပါတယ်။
