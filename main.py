import io
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException

# 1. Create the FastAPI app
app = FastAPI()

# 2. Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Define the class name 
classes = ["bact", "fung", "healthy"]

# 4. Recreate the architecture that used to train the best model
model = models.resnet50(weights = None) # weigths are none here as I have saved the best weight in best_model.pth file

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features= 2048,
              out_features= len(classes))
)

model.load_state_dict(torch.load("best_model.pth", map_location= device))
model.to(device)
model.eval()


# 6. Using the same transform technique that been used for training
transform = transforms.Compose([
    transforms.Resize(size = (224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    return {"status": "ok", "message": "API is running", "device": str(device)}

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse("static/index.html")



@app.post("/predict")
async def predict(file : UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", "jpeg", "jpg")):
        raise HTTPException(status_code= 400, detail= "Invalid image format")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Error processing image.")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
        predicted_index = probabilities.argmax().item()
        predicted_class = classes[predicted_index]
        confidence = probabilities[predicted_index].item()

    return  {
    "predicted_class": predicted_class,
    "confidence": round(confidence, 4),
    "probabilities": {
        name: round(probabilities[i].item(), 4)
        for i, name in enumerate(classes)
    }
    
}