"""
visualize_batch.py

Run the trained SOD model on a batch of images from the DATASET
and save visualizations (input, GT mask, pred mask, overlay) for
several samples. Good for the report & slides.
"""

import os

import torch

from data_loader import create_dataloaders
from sod_model import SODNet
from evaluate import visualize_sample  # we already defined this there


def main():
    # --- config ---
    data_root = os.path.join("data", "DUTS")
    ckpt_path = os.path.join("checkpoints", "best_model.pth")
    img_size = 128
    batch_size = 8
    num_samples_to_save = 6          # how many examples to visualize
    save_dir = os.path.join("results", "batch_test")

    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataloaders (we only really need test_loader here)
    _, _, test_loader = create_dataloaders(
        root_dir=data_root,
        img_size=img_size,
        batch_size=batch_size,
    )

    # Model
    model = SODNet(in_channels=3, base_channels=32).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run train.py first."
        )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded best model from {ckpt_path}")

    saved = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)  # [B, 1, H, W]

            batch_size_now = images.size(0)
            for i in range(batch_size_now):
                if saved >= num_samples_to_save:
                    break

                img = images[i]
                gt = masks[i]
                pred = preds[i]

                save_path = os.path.join(save_dir, f"sample_{saved+1}.png")
                title = f"Test sample {saved+1}"

                visualize_sample(img, gt, pred, save_path=save_path, title=title)
                print(f"Saved {save_path}")
                saved += 1

            if saved >= num_samples_to_save:
                break

    print(f"\nDone. Saved {saved} visualizations in {save_dir}")


if __name__ == "__main__":
    main()
