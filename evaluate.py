import torch
from dataset_loader import get_dataloaders
from cnn_model import ChestXRayCNN
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader, classes = get_dataloaders()

model = ChestXRayCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=classes))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
