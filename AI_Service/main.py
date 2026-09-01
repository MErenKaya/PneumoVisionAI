import io
import os
import cv2
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OpenCVCLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        enhanced = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (224, 224))

def overlay_heatmap(pil_image, heatmap):
    img_np = np.array(pil_image.resize((224, 224)))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format="JPEG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

MODEL_PATH = "best_chest_model.pth"
CLASSES = ["NORMAL", "PNEUMONIA"]

inference_transforms = transforms.Compose([
    OpenCVCLAHETransform(clip_limit=2.0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

grad_cam = GradCAM(model, model.layer4[-1])

app = FastAPI(title="Pneumonia Detection AI Service with Grad-CAM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_chest_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = inference_transforms(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_batch)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_label = CLASSES[predicted_class_idx]
        confidence_score = float(probabilities[predicted_class_idx].item() * 100)

        input_batch_cam = input_batch.clone().detach().requires_grad_(True)
        heatmap = grad_cam.generate_heatmap(input_batch_cam, predicted_class_idx)
        heatmap_base64 = overlay_heatmap(pil_image, heatmap)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": round(confidence_score, 2),
            "probabilities": {
                "NORMAL": round(float(probabilities[0].item()) * 100, 2),
                "PNEUMONIA": round(float(probabilities[1].item()) * 100, 2)
            },
            "heatmap_base64": heatmap_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")