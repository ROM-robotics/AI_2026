# Test YOLO26 Pretrained Models from Command Line

ဒီ file က pretrained YOLO26 models တွေကို command line ကနေ အလွယ်တကူ စမ်းသပ်နိုင်အောင် ရေးထားတဲ့ instruction sheet ပါ။

Main tasks 4 ခုကို အဓိကထားပါတယ်:

- Detect
- Segment
- Classify
- Pose

OBB ကိုလည်း optional အဖြစ် အောက်ဆုံးမှာ ထည့်ထားပါတယ်။

## 1. Environment Ready

Virtual environment activate လုပ်ပြီး `ultralytics` install ထားရပါမယ်။

```powershell
.\.venv\Scripts\Activate.ps1
pip install -U ultralytics
yolo checks
```

Note:

- pretrained model weights တွေကို first run လုပ်တဲ့အချိန်မှာ auto-download လုပ်ပါတယ်
- URL image သုံးမယ်ဆိုရင် internet လိုပါတယ်
- output files တွေကို ပုံမှန်အားဖြင့် `runs/` folder အောက်မှာ save လုပ်ပါတယ်

## 2. Quick Test Images

အောက်က sample sources တွေကို သုံးလို့ရပါတယ်:

- detect / segment / classify: `https://ultralytics.com/images/bus.jpg`
- pose: `https://ultralytics.com/images/zidane.jpg`
- obb optional: `https://ultralytics.com/images/boats.jpg`

Local image နဲ့လည်း စမ်းလို့ရပါတယ်:

```powershell
yolo detect predict model=yolo26n.pt source=path/to/your/image.jpg
```

## 3. Detect

Object detection က image ထဲက object တွေကို bounding box နဲ့ detect လုပ်ပါတယ်။

### Basic test

```powershell
yolo detect predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg
```

### Save labels and confidence

```powershell
yolo detect predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg save=True save_txt=True save_conf=True
```

### Local image test

```powershell
yolo detect predict model=yolo26n.pt source=.\test_images\bus.jpg
```

Expected result:

- image ပေါ်မှာ boxes တွေ draw လုပ်ထားတဲ့ output image တစ်ခု save ဖြစ်မယ်
- `save_txt=True` သုံးရင် txt labels လည်းထွက်မယ်

## 4. Segment

Instance segmentation က object ကို box တင်မက mask shape နဲ့ပါ ထုတ်ပေးပါတယ်။

### Basic test

```powershell
yolo segment predict model=yolo26n-seg.pt source=https://ultralytics.com/images/bus.jpg
```

### Save mask result

```powershell
yolo segment predict model=yolo26n-seg.pt source=https://ultralytics.com/images/bus.jpg save=True save_txt=True
```

### Local image test

```powershell
yolo segment predict model=yolo26n-seg.pt source=.\test_images\bus.jpg
```

Expected result:

- object တစ်ခုချင်းစီအတွက် colored masks တွေပါလာမယ်
- detection boxes လည်း တွဲမြင်ရနိုင်တယ်

## 5. Classify

Classification က image တစ်ပုံလုံးကို class တစ်ခု assign လုပ်ပေးပါတယ်။

### Basic test

```powershell
yolo classify predict model=yolo26n-cls.pt source=https://ultralytics.com/images/bus.jpg
```

### Local image test

```powershell
yolo classify predict model=yolo26n-cls.pt source=.\test_images\bus.jpg
```

### Folder test

```powershell
yolo classify predict model=yolo26n-cls.pt source=.\test_images\
```

Expected result:

- top predicted class name နဲ့ confidence score ထွက်မယ်
- classification task မှာ bounding box မထွက်ဘူး

## 6. Pose

Pose estimation က human body keypoints လို landmarks တွေကို detect လုပ်ပေးပါတယ်။

### Basic test

```powershell
yolo pose predict model=yolo26n-pose.pt source=https://ultralytics.com/images/zidane.jpg
```

### Save keypoints result

```powershell
yolo pose predict model=yolo26n-pose.pt source=https://ultralytics.com/images/zidane.jpg save=True
```

### Local image test

```powershell
yolo pose predict model=yolo26n-pose.pt source=.\test_images\person.jpg
```

Expected result:

- လူပုံရှိရင် body joints / skeleton lines တွေ draw လုပ်ထားတာ မြင်ရမယ်
- လူမပါတဲ့ပုံနဲ့ စမ်းရင် keypoints မထွက်နိုင်ဘူး

## 7. Optional: OBB

Oriented Bounding Box detection က rotated boxes ထုတ်ပေးပါတယ်။ Drone image, aerial image, ships, solar panels လို tasks တွေအတွက် အသုံးဝင်ပါတယ်။

```powershell
yolo obb predict model=yolo26n-obb.pt source=https://ultralytics.com/images/boats.jpg
```

Local image:

```powershell
yolo obb predict model=yolo26n-obb.pt source=.\test_images\boats.jpg
```

Expected result:

- သာမန် rectangle မဟုတ်ဘဲ angled boxes တွေ ထွက်မယ်

## 8. Webcam Test

YOLO26 pretrained models တွေကို webcam နဲ့ live test လုပ်လို့ရပါတယ်။ Main 4 tasks အတွက် command တွေက အောက်ပါအတိုင်းဖြစ်ပါတယ်:

### Detect webcam

```powershell
yolo detect predict model=yolo26n.pt source=0 show=True
```

### Segment webcam

```powershell
yolo segment predict model=yolo26n-seg.pt source=0 show=True
```

### Classify webcam

```powershell
yolo classify predict model=yolo26n-cls.pt source=0 show=True
```

### Pose webcam

```powershell
yolo pose predict model=yolo26n-pose.pt source=0 show=True
```

### If you want to save webcam results

```powershell
yolo detect predict model=yolo26n.pt source=0 show=True save=True
```

### If webcam index 0 does not work

```powershell
yolo detect predict model=yolo26n.pt source=1 show=True
```

Note:

- `source=0` က default webcam ကိုဆိုလိုပါတယ်
- တချို့ laptop/USB camera တွေမှာ `source=1` သို့မဟုတ် `source=2` ဖြစ်နိုင်ပါတယ်
- `show=True` က live preview window ဖွင့်ပေးတာပါ
- `save=True` သုံးရင် recorded prediction result ကို `runs/` အောက်မှာ save လုပ်ပါတယ်
- webcam window ကို ပိတ်ချင်ရင် `q` နှိပ်ပါ သို့မဟုတ် window ကို close လုပ်ပါ
- classification webcam test မှာ box မထွက်ဘဲ top class prediction ပဲပြမယ်

## 9. Useful Options

Confidence threshold ပြောင်းရန်:

```powershell
conf=0.25
```

Image size ပြောင်းရန်:

```powershell
imgsz=640
```

CPU ပေါ်မှာ run ရန်:

```powershell
device=cpu
```

GPU 0 သုံးရန်:

```powershell
device=0
```

Example:

```powershell
yolo detect predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg conf=0.3 imgsz=640 device=0
```

## 10. Recommended Minimal Test Set

အချိန်နည်းနည်းနဲ့ model 4 ခုစမ်းချင်ရင် ဒီ commands 4 ခုနဲ့စပါ:

```powershell
yolo detect predict model=yolo26n.pt source=https://ultralytics.com/images/bus.jpg
yolo segment predict model=yolo26n-seg.pt source=https://ultralytics.com/images/bus.jpg
yolo classify predict model=yolo26n-cls.pt source=https://ultralytics.com/images/bus.jpg
yolo pose predict model=yolo26n-pose.pt source=https://ultralytics.com/images/zidane.jpg
```

## 11. Troubleshooting

### If `yolo` command not found

```powershell
python -m ultralytics checks
```

သို့မဟုတ် venv activate ပြန်လုပ်ပါ:

```powershell
.\.venv\Scripts\Activate.ps1
```

### If model download fails

- internet connection စစ်ပါ
- firewall / proxy စစ်ပါ
- command ကို ပြန် run ပါ

### If webcam does not open

- `source=0` ကို `source=1` နဲ့ပြောင်းစမ်းပါ
- camera permission စစ်ပါ

### If pose result is empty

- လူပါသော image ကိုသုံးပါ
- `zidane.jpg` က pose test အတွက် သင့်တော်တယ်

## 12. Model Name Summary

- Detect: `yolo26n.pt`
- Segment: `yolo26n-seg.pt`
- Classify: `yolo26n-cls.pt`
- Pose: `yolo26n-pose.pt`
- OBB optional: `yolo26n-obb.pt`

`n` model က အငယ်ဆုံး model ဖြစ်ပြီး quick test အတွက် အကောင်းဆုံးပါ။ Accuracy ပိုလိုရင် `s`, `m`, `l`, `x` variants တွေကိုအစားထိုးသုံးနိုင်ပါတယ်။
