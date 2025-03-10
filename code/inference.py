# inference.py

import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import io
import json
import os
import timm

# Class mapping
class_dict = {0: 'Chickenpox', 1: 'HFMD', 2: 'atopic', 3: 'shingles'}

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    model_path = os.path.join(model_dir, 'model.pth')
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model

def input_fn(request_body, content_type='application/x-image'):
    if content_type == 'application/x-image':
        # Read the image data
        image = Image.open(io.BytesIO(request_body))
        return image
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    # input_data is the image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize([0.61266785, 0.47934173, 0.43867121], [0.25417075, 0.22141552, 0.21571804])
    ])
    image = transform(input_data).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        acc = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(acc, 4)
    top_probs = top_probs.cpu().numpy().flatten().tolist()
    top_indices = top_indices.cpu().numpy().flatten().tolist()
    top_class = [class_dict[idx] for idx in top_indices]
    result = [{"class": cls, "probability": prob} for cls, prob in zip(top_class, top_probs)]
    return result

def output_fn(prediction_output, accept='application/json'):
    if accept == 'application/json':
        return json.dumps(prediction_output), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
