"""
train.py

Training loop for the SOD model:
- Loads dataloaders
- Trains model with BCE + (1 - IoU) loss
- Logs loss and metrics
- Saves best model by validation loss
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data_loader import create_dataloaders
from sod_model import SODNet
from evaluate import compute_iou


def sod_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Combined loss: BCE + 0.5 * (1 - IoU)
    """
    bce = nn.functional.binary_cross_entropy(preds, targets)

    with torch.no_grad():
        iou = compute_iou(preds, targets)

    # IoU is a float, convert to tensor on same device
    iou_tensor = preds.new_tensor(iou)
    loss = bce + 0.5 * (1.0 - iou_tensor)

    return loss


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)

        loss = sod_loss(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_iou += compute_iou(preds.detach(), masks.detach())

    epoch_loss = running_loss / len(loader)
    epoch_iou = running_iou / len(loader)
    return epoch_loss, epoch_iou


def validate(
    model: nn.Module,
    loader,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = sod_loss(preds, masks)

            running_loss += loss.item()
            running_iou += compute_iou(preds, masks)

    epoch_loss = running_loss / len(loader)
    epoch_iou = running_iou / len(loader)
    return epoch_loss, epoch_iou


def main():
    # TODO: change this to your real dataset path
    data_root = os.path.join("data", "DUTS")  # e.g. data/DUTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=data_root,
    img_size=128,
    batch_size=8,
)

    model = SODNet(in_channels=3, base_channels=32).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 25
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 3  # stop if val loss doesn't improve for 3 epochs

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # training step
        train_loss, train_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        # validation step
        val_loss, val_iou = validate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
        )

        # -------------------------
        #     EARLY STOPPING
        # -------------------------
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0

            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join("checkpoints", "best_model.pth"),
            )
            print("Saved new best model to checkpoints/best_model.pth")
        else:
            epochs_no_improve += 1
            print(f"No improvement in val loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break



if __name__ == "__main__":
    main()
