import torch
from PIL import Image
import numpy as np
import io
import os

# Locate model relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/best.torchscript")

# Load TorchScript model
model = torch.jit.load(MODEL_PATH, map_location="cpu")

# Quantize dynamically – smaller and faster for CPU
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Conv2d, torch.nn.Linear},
    dtype=torch.qint8
)
model.eval()

def predict_image(file_bytes: bytes):
    """Run inference on uploaded image bytes and return detections"""
    # Preprocess image
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((640, 640))
    img_array = np.array(image)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        results = model(img_tensor)

    # Convert results to JSON‑friendly format
    detections = []
    if isinstance(results, (list, tuple)):
        for r in results:
            detections.append(r.detach().cpu().tolist() if isinstance(r, torch.Tensor) else r)
    elif isinstance(results, torch.Tensor):
        detections = results.detach().cpu().tolist()
    else:
        detections = [results]

    return detections