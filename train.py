import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import get_dataloaders
from cnn_model import ChestXRayCNN



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, test_loader, classes = get_dataloaders("train", "test")
print("Classes found:", classes, flush=True)
print("Train batches:", len(train_loader), flush=True)

# Initialize model
model = ChestXRayCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5
losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}", flush=True)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}", flush=True)

# Save model
torch.save(model.state_dict(), "cnn_model.pth")
print("Model saved as cnn_model.pth", flush=True)

# Plot loss curve
