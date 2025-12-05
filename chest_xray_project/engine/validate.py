import torch

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == masks).sum().item()
            total += masks.numel()

    return total_loss / len(loader), correct / total