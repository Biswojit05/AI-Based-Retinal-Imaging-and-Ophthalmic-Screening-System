"""
Trains the ResNet50 eye-disease classifier (same pipeline as main.ipynb)
and saves the weights + label mapping to disk so a backend service can
load them for inference.

Usage: python train_model.py
"""
import json
import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MODEL_DIR = "model"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv("full_df.csv")
df["new_path"] = df["filename"].apply(lambda x: os.path.join("preprocessed_images", x))

train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["labels"], random_state=42
)

le = LabelEncoder()
train_df = train_df.copy()
test_df = test_df.copy()
train_df["encoded_label"] = le.fit_transform(train_df["labels"])
test_df["encoded_label"] = le.transform(test_df["labels"])

label_map = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
print("Label mapping:", label_map)

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EyeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "new_path"]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.df.loc[idx, "encoded_label"], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


train_loader = DataLoader(
    EyeDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    EyeDataset(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = models.resnet50(weights="DEFAULT")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 8)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_acc:.2f}%"
    )

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

checkpoint_path = os.path.join(MODEL_DIR, "eye_disease_resnet50.pth")
torch.save(model.state_dict(), checkpoint_path)
print("Saved model weights to", checkpoint_path)

idx_to_label = {v: k.strip("[]'\"") for k, v in label_map.items()}
labels_path = os.path.join(MODEL_DIR, "labels.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(idx_to_label, f, indent=2)
print("Saved label mapping to", labels_path)
print("Test accuracy:", round(test_acc, 2))
