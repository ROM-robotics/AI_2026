### Day12 — CNN (Transfer-Learning) 

ဒီ folder မှာ CNN လေ့ကျင့်ခန်းများ၊ dataset ထည့်သွင်းနည်း၊ example code တွေ ပါဝင်ပါတယ်။

#### Dataset Sources (01/02/03/04)

- `01_resnet50_facial_emotion.ipynb` — Hugging Face dataset
- `02_efficientnet-fruit-classification.ipynb` — Roboflow dataset
- `03_mobilenetv3-fruit-classification.ipynb` — Roboflow dataset
- `04_convnext_object_detection.ipynb` — PASCAL VOC 2012 dataset

#### Roboflow Dataset အသုံးပြုနည်း

Roboflow မှ dataset ယူရာတွင် `API Key` ကို Roboflow Settings ကနေယူပါ။  
`ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION` ကို dataset link နဲ့ ကိုက်ညီအောင် ပြောင်းပေးရပါမယ်။  

```python
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "Your API Key here")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "Your Workspace")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "Your Project")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "Your Version"))
EXPORT_FORMAT = "folder"
```

#### မှတ်ချက်များ

- `API Key` ကို Roboflow Settings ကနေ ယူပါ။
- `Workspace`/`Project`/`Version` ကို dataset link (URL) ထဲက slug နဲ့ ကိုက်ညီအောင် ထည့်ပါ။
- version အမှား ဖြစ်ပါက dataset download မဖြစ်နိုင်ပါ။


Slides URLs - Google Drive

https://drive.google.com/drive/folders/10Fy--rTTPQ4mTWQch1wJCD58CZ6haGgd?usp=sharing