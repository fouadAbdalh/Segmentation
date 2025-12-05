import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.train import load_train_data
from data.test import load_test_data
from data.dataset import SegmentationDataset

from model.unet import UNet
from engine.train import train_epoch
from engine.validate import validate


train_dir = '../input/chest-xray-masks-and-defect-detection/train_images/'
test_dir = '../input/chest-xray-masks-and-defect-detection/test_images/'

X_train, y_train = load_train_data(train_dir)
X_test = load_test_data(test_dir)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

train_dataset = SegmentationDataset(X_train, y_train)
valid_dataset = SegmentationDataset(X_valid, y_valid)
test_dataset = SegmentationDataset(X_test, np.zeros((len(X_test), 256, 256)))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, valid_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
        best_val_loss = val_loss
        print("✅ Model saved!")