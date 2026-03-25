### Day 12 — CNN (Transfer Learning)

ဒီ folder မှာ **CNN Transfer Learning** ဆိုင်ရာ လေ့ကျင့်ခန်းများ၊ dataset အသုံးပြုနည်းများ နှင့် example code များ ပါဝင်ပါတယ်။

---

### 📂 Included Notebooks (Datasets 01 / 02 / 03 / 04)

- `01_resnet50_facial_emotion.ipynb`  
  → [Hugging Face Dataset](https://huggingface.co/datasets) အသုံးပြုထားပါတယ်။

- `02_efficientnet-fruit-classification.ipynb`  
  → [Roboflow Dataset](https://roboflow.com/) အသုံးပြုထားပါတယ်။

- `03_mobilenetv3-fruit-classification.ipynb`  
  → [Roboflow Dataset](https://roboflow.com/) အသုံးပြုထားပါတယ်။

- `04_convnext_object_detection.ipynb`  
  → [PASCAL VOC 2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) အသုံးပြုထားပါတယ်။

---

### 📥 Roboflow Dataset အသုံးပြုနည်း

Roboflow dataset ကို download လုပ်ရန် အောက်ပါအဆင့်များကို လိုက်နာပါ —

1. **API Key ရယူခြင်း**  
   → Roboflow account ၏ *Settings* မှာ API Key ကို ရယူပါ

2. **Dataset URL မှ Information ထုတ်ယူခြင်း**  
   → Dataset URL ထဲက `ROBOFLOW_WORKSPACE`, `ROBOFLOW_PROJECT`, `ROBOFLOW_VERSION` ကို dataset link နဲ့ ကိုက်ညီအောင် ပြောင်းပေးရပါမယ်။  

---

### 🔍 Example URL Analysis

ဥပမာ Dataset URL —

https://app.roboflow.com/hmue/fruit-dcjhh-pkb5t/1

ဒီ URL ကို အပိုင်းလိုက် ခွဲကြည့်ရင် —

- `hmue` → **Workspace**
- `fruit-dcjhh-pkb5t` → **Project Name**
- `1` → **Version**

👉 ဒါကြောင့် code ထဲမှာ အောက်ပါအတိုင်း ပြောင်းရပါမယ် —

- `Your Workspace` → `hmue`  
- `Your Project` → `fruit-dcjhh-pkb5t`  
- `Your Version` → `1`  

---


### ⚙️ Code Configuration (Notebook 02 & 03)

အောက်ပါ code ကို မိမိ dataset အတိုင်း ပြင်ဆင်ပါ —

```python
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "Your API Key here")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "Your Workspace")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "Your Project")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", "Your Version"))
EXPORT_FORMAT = "folder"
```
### 💡 မှတ်ထားရန်

- URL ထဲက **Workspace / Project / Version** ကိုသာ ယူပြီး replace လုပ်ရမယ်  
- `"Your ..."` ဆိုတာတွေကို မိမိ dataset နဲ့ကိုက်ညီအောင် ပြောင်းပေးရမယ်  
- Version မှားရင် dataset download မရနိုင်ပါ ❌  
- Notebook ကို kaggle တို့တွင် GPU အထောက်အပံ့ဖြင့် run လုပ်ပါ။
- Roboflow API Key ကို publicly commit မလုပ်ပါနှင့်။


Slides URLs - Google Drive

https://drive.google.com/drive/folders/10Fy--rTTPQ4mTWQch1wJCD58CZ6haGgd?usp=sharing