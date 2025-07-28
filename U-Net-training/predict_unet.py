import torch
from unet_model import UNet
from dataset_loader import CustomDataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import os
from PIL import Image

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("unet_epoch_25.pth", map_location=device))
    model.eval()

    test_ds = CustomDataset("dataset/images", "dataset/masks")
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for i, (image, _) in enumerate(test_dl):
            image = image.to(device)
            output = model(image)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()

            pred_mask = TF.to_pil_image(output.squeeze().cpu())
            pred_mask.save(f"predictions/pred_{i:04d}.png")

if __name__ == "__main__":
    predict()
