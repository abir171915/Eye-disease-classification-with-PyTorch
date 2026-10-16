"""Model loading and prediction, shared by the FastAPI app and the Streamlit app."""
import io

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ["bact", "fung", "healthy"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same preprocessing as validation/test during training
preprocess = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(weights_path="best_model.pth"):
    """Rebuild the training architecture and load the best checkpoint."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=2048, out_features=len(CLASSES))
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_bytes):
    """Run one image through the model. Raises PIL.UnidentifiedImageError on unreadable input."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    predicted_index = probabilities.argmax().item()
    return {
        "predicted_class": CLASSES[predicted_index],
        "confidence": round(probabilities[predicted_index].item(), 4),
        "probabilities": {
            name: round(probabilities[i].item(), 4)
            for i, name in enumerate(CLASSES)
        },
    }
