import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")

model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 5)

state_dict = torch.load("retina_model_best.pth", map_location=device)
print("Keys in checkpoint:", list(state_dict.keys())[:5])  # show first 5 layer names

missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

model.eval()

# Test on a real image - update this path to any actual image in your dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("../data/train_images/000c1434d8d7.png").convert("RGB")
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor)
    print("Raw output logits:", outputs)
    probs = torch.softmax(outputs, dim=1)
    print("Probabilities:", probs)