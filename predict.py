import torch
from torchvision import transforms
from PIL import Image
from cnn_model import ChestXRayCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = ['Covid', 'Lung_Opacity', 'Normal', 'Pneumonia', 'Tuberculosis']

def predict_image(image_path, model_path="cnn_model.pth"):
    model = ChestXRayCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]
