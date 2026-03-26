import streamlit as st
import torch
import torch.nn as nn
import time
import pandas as pd
from torchvision import transforms, models
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "D:\\AI_Project_2026\\AI_2026\\Day12 CNN tf learning\\groupA\\po\\saved_models\\full_resnet50.pth"
IMG_SIZE = 128  # smaller input size
CLASS_NAMES = ['chicken_wings', 'club_sandwich', 'donuts', 'pizza', 'sushi']
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍔",
    layout="wide"
)

# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model():
    # Initialize model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Load full model checkpoint safely (trusted file)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # If checkpoint is state_dict or dict with 'state_dict'
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # checkpoint is full model object
        model = checkpoint

    model.to(DEVICE)
    model.eval()
    return model

# Load model once globally
model = load_model()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    inference_time = (time.time() - start) * 1000
    return probs, inference_time

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🍔 Food Image Classifier")

uploaded_file = st.file_uploader(
    "Upload an image of food",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display smaller image
    st.image(image, caption="Input Image", width=300)
    
    # Make prediction
    probs, inference_time = predict(image)
    top_idx = probs.argmax()

    st.success(f"Prediction: {CLASS_NAMES[top_idx]}")
    st.metric("Confidence", f"{probs[top_idx]:.2%}")
    st.metric("Inference Time", f"{inference_time:.1f} ms")

    # Display all probabilities as bar chart
    st.bar_chart(pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": probs
    }).set_index("Class"))