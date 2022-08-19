from PIL import Image, ImageFont, ImageDraw
import torch
im = Image.open("drake.jpg")
from torchvision import transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print(image_transform(im))
