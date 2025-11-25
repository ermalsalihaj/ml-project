"""
visualize_single.py

Run the trained SOD model on a single image and visualize:
- Input image
- Predicted saliency mask
- Overlay
"""

import os
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from sod_model import SODNet


def preprocess_image(img_path: str, img_size: int = 128):
    """
    Load an RGB image, resize, and convert to tensor in [0,1].
    Returns:
      - original PIL image (for nicer plotting)
      - tensor of shape [1, 3, H, W]
    """
    img = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0,1]
    ])

    tensor = transform(img)  # [3, H, W]
    tensor = tensor.unsqueeze(0)  # [1, 3, H, W]
    return img, tensor


def visualize(image_pil, pred_mask: torch.Tensor, save_path: str = None, title: str = ""):
    """
    Visualization:
    - Input image
    - Predicted mask
    - Overlay (mask over image)
    """
    pred = pred_mask.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Input
    axes[0].imshow(image_pil)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title("Pred Mask")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image_pil)
    axes[2].imshow(pred, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize SOD prediction for a single image.")
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to an input RGB image."
    )
    parser.add_argument(
        "--ckpt",
        default=os.path.join("checkpoints", "best_model.pth"),
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Resize image to this size (H=W=img_size)."
    )
    parser.add_argument(
        "--save", "-s",
        default=None,
        help="Optional path to save the visualization PNG (if not set, just shows a window)."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load model
    model = SODNet(in_channels=3, base_channels=32).to(device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found at {args.ckpt}. "
            f"Please run train.py first to generate best_model.pth."
        )

    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded best model from {args.ckpt}")

    # 2) Preprocess image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found at {args.image}")

    orig_img, img_tensor = preprocess_image(args.image, img_size=args.img_size)
    img_tensor = img_tensor.to(device)

    # 3) Run model
    with torch.no_grad():
        preds = model(img_tensor)  # [1, 1, H, W]
    pred_mask = preds[0]  # [1, H, W]

    # 4) Visualize
    title = os.path.basename(args.image)
    visualize(orig_img, pred_mask, save_path=args.save, title=title)


if __name__ == "__main__":
    main()
