import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

device = torch.device("cpu")

# Rebuild the exact same architecture we trained, then load our trained weights
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 5)
model.load_state_dict(torch.load("retina_model_best.pth", map_location=device))
model.eval()  # inference mode - disables dropout etc.

# Same preprocessing as validation (no augmentation - we want consistent,
# unmodified predictions on real uploaded images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image: Image.Image) -> dict:
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

    return {
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "all_probabilities": {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
    }