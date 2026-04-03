
import os
# ဒီ line က ကျန်တဲ့ import အားလုံးရဲ့ အပေါ်ဆုံးမှာ ရှိနေရပါမယ်
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
# ... ကျန်တဲ့ import တွေ ဒီအောက်မှာ ဆက်ရေးပါ

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision
from PIL import Image
from pathlib import Path

# ၁။ Error များ ကာကွယ်ရန် Environment သတ်မှတ်ခြင်း
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ၂။ Configuration (ကိုယ်တိုင် ပြင်ဆင်ရန် အပိုင်း)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model သိမ်းထားတဲ့ လမ်းကြောင်း (မှန်မမှန် ပြန်စစ်ပါ)
MODEL_PATH = Path(r"saved_models/full_mobilenet_v3_large.pth")

# စမ်းသပ်မည့် ပန်းသီးပုံ လမ်းကြောင်း (ဒီနေရာမှာ သင့်ပုံရဲ့ path ကို အတိအကျ ပြင်ပါ)
# ဥပမာ - r"C:\Users\Naing\Desktop\apple_test.jpg"
TEST_IMG_PATH = r"C:\Users\Naing\Desktop\test_apple.jpg" 

IMG_SIZE = 224
CLASS_NAMES = ['Fresh Apple', 'Rotten Apple'] 

# PyTorch 2.6+ အတွက် MobileNetV3 ကို safe global အဖြစ် သတ်မှတ်ခြင်း
torch.serialization.add_safe_globals([torchvision.models.mobilenetv3.MobileNetV3])

# ၃။ Image Transformations (Train တုန်းကအတိုင်း ပြင်ဆင်ခြင်း)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def predict_image(image_path):
    # Model ရှိမရှိ အရင်စစ်မယ်
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # Model ကို Load လုပ်မယ်
    try:
        # weights_only=False ထည့်မှ model structure ကိုပါ load လုပ်နိုင်မှာပါ
        model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # ပုံကို ဖွင့်ပြီး tensor ပြောင်းမယ်
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Cannot open image file. Please check the path.")
        print(f"Path you provided: {image_path}")
        return

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Model ကို ခန့်မှန်းခိုင်းမယ်
    with torch.no_grad():
        outputs = model(image_tensor)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(prob, 1)

    result_label = CLASS_NAMES[predicted[0]]
    score = confidence[0].item() * 100

    print(f"\n" + "="*30)
    print(f"Prediction Result")
    print(f"="*30)
    print(f"File: {image_path}")
    print(f"Result: {result_label}")
    print(f"Confidence: {score:.2f}%")
    print(f"="*30)

if __name__ == "__main__":
    predict_image(r"C:\Users\Naing\Desktop\apple.jpg")