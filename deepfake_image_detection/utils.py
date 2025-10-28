# utils.py
import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import torch.nn as nn

IMAGE_SIZE = 224

def load_model(model_path):
    # Load ResNet50 base
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features

    # Match the same structure as in your notebook (512 hidden layer)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Load class names (default to real/fake)
    classes = ckpt.get("classes", ["real", "fake"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, classes, device


def preprocess_image(image_pil):
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return preprocess(image_pil).unsqueeze(0)  # add batch

def predict_image(model, classes, image_pil, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = preprocess_image(image_pil).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    # Get top class
    idx = int(probs.argmax())
    label = classes[idx]
    confidence = float(probs[idx])
    return label, confidence, probs
