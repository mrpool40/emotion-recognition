import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import wandb
import os
from sklearn.metrics import classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# ===  W&B Login and Init ===
wandb.login()
wandb.init(project="emotion-recognition-pytorch", name="resnet18-gpu")

# ===  Settings ===
train_dir = "dataset/train"
test_dir = "dataset/test"
BATCH_SIZE = 32
NUM_EPOCHS = 50
IMG_SIZE = 224
NUM_CLASSES = len(os.listdir(train_dir))

#  Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===  Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===  Dataset Loading
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

# ===  Load ResNet18 (fixed warning)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# ===  Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ===  Training Loop
for epoch in range(NUM_EPOCHS):
    print(f"\n Starting Epoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for batch_i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

        if batch_i % 10 == 0:
            print(f"  Batch {batch_i+1}/{len(train_loader)} processed")

    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_corrects.double() / len(train_data)

    # ===  Evaluation on Test Set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ===  Metrics
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    binarized_labels = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    binarized_preds = label_binarize(all_preds, classes=list(range(NUM_CLASSES)))
    try:
        map_50 = average_precision_score(binarized_labels, binarized_preds, average="macro")
    except ValueError:
        map_50 = 0.0  # in case of only 1 class or empty preds

    print(f" Epoch {epoch+1} Results:")
    print(f"  Loss     : {epoch_loss:.4f}")
    print(f"  Accuracy : {epoch_acc:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, mAP@0.5: {map_50:.4f}")

    # ===  Log to W&B
    wandb.log({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP@0.5": map_50
    })

# ===  Save Model
torch.save(model.state_dict(), "emotion_model.pth")
print(" Model saved to emotion_model.pth")
