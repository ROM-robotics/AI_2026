import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ပြီးမှ ကျန်တဲ့ import တွေကို ဆက်ရေးပါ
import torch
import torchvision
# ...
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

# --- Configuration ---
MODEL_PATH = Path(r"saved_models/full_mobilenet_v3_large.pth")
CLASS_NAMES = ['Fresh Apple', 'Rotten Apple'] # လိုအပ်ရင် အသီးအသစ်တွေ ဒီမှာထည့်ပါ
IMG_SIZE = 224

st.title("🍎 Fruit Quality Checker")
st.write("ပန်းသီးပုံကို Upload တင်ပြီး အကောင်း/အပုပ် စစ်ဆေးနိုင်ပါတယ်။")

# Model Load လုပ်ခြင်း (Cache ထားမှ ခဏခဏ load မလုပ်မှာပါ)
@st.cache_resource
def load_my_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    return model

# ပုံကို Upload တင်ရန် အပိုင်း
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    # Prediction
    model = load_my_model()
    with torch.no_grad():
        outputs = model(img_tensor)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(prob, 1)

    result = CLASS_NAMES[predicted[0]]
    score = confidence[0].item() * 100

    # ရလဒ်ပြသခြင်း
    st.subheader(f"ရလဒ်: {result}")
    st.write(f"Confidence: {score:.2f}%")