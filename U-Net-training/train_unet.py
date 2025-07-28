import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CustomDataset
from unet_model import UNet
import matplotlib.pyplot as plt
import os

def dice_coef(pred, target, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def train():
    train_ds = CustomDataset("dataset/images", "dataset/masks")
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)

    val_ds = CustomDataset("dataset/images", "dataset/masks")
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100
    train_losses = []
    val_losses = []
    val_dices = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_dl:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_dl:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_coef(outputs, masks)
                val_loss += loss.item()
                val_dice += dice.item()

        avg_train_loss = running_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        avg_val_dice = val_dice / len(val_dl)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"unet_model_epoch{epoch+1}.pth")

    # Plot and save metrics
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.plot(range(1, num_epochs+1), val_dices, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curves.png")
    print("Saved training curve to training_curves.png")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
