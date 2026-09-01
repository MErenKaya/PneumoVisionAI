import os
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 1. OpenCV CLAHE Preprocessing Class (Pickle-safe for Windows Multiprocessing)
class OpenCVCLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        # Create CLAHE object dynamically inside call to prevent serialization issues
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img_pil)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        enhanced = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

def main():
    # 2. Device Configuration (NVIDIA GPU / CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Execution Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU Model: {torch.cuda.get_device_name(0)}")

    # 3. Data Transformations Pipeline
    train_transforms = transforms.Compose([
        OpenCVCLAHETransform(clip_limit=2.0),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        OpenCVCLAHETransform(clip_limit=2.0),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Dataset Path Resolution
    if os.path.exists('./chest_xray'):
        data_dir = './chest_xray'
    elif os.path.exists('../chest_xray'):
        data_dir = '../chest_xray'
    else:
        raise FileNotFoundError("Directory 'chest_xray' was not found in current or parent path!")

    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)

    # 85% Training - 15% Validation Split
    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"📊 Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")
    print(f"🏷️ Classes: {full_train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA']

    # 5. ResNet18 Transfer Learning Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)

    # 6. Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 7. Training Loop
    epochs = 5
    best_acc = 0.0
    start_time = time.time()

    print("\n🚀 Starting Model Training...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = corrects.double() / len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch {epoch+1}/{epochs} -> "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc*100:.2f}%")

        # Save Checkpoint with Highest Validation Accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_chest_model.pth')
            print(f"  --> ✨ Saved new best model checkpoint! (Val Acc: {best_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time//60:.0f}m {total_time%60:.0f}s!")
    print(f"🏆 Best Validation Accuracy: {best_acc*100:.2f}% -> Saved as 'best_chest_model.pth'")

# Windows Multiprocessing Main Guard
if __name__ == '__main__':
    main()