"""
evaluate.py

Evaluation metrics and simple visualization helpers.
"""

from typing import Dict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    preds: [B, 1, H, W] after sigmoid
    targets: [B, 1, H, W], binary {0,1}
    """
    preds_bin = (preds > threshold).float()

    intersection = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection + 1e-7

    iou = (intersection / union).mean().item()
    return iou


def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Pixel-wise Precision, Recall, F1.
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()

    tp = (preds_bin * targets_bin).sum().item()
    fp = (preds_bin * (1 - targets_bin)).sum().item()
    fn = ((1 - preds_bin) * targets_bin).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def visualize_sample(image, gt_mask, pred_mask, save_path: str = None, title: str = ""):
    """
    Simple visualization:
    - Input image
    - Ground truth mask
    - Predicted mask
    - Overlay (pred over image)
    """
    image = image.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
    gt = gt_mask.detach().cpu().squeeze().numpy()
    pred = pred_mask.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Pred Mask")
    axes[2].axis("off")

    axes[3].imshow(image)
    axes[3].imshow(pred, cmap="jet", alpha=0.4)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print("This file defines IoU, precision/recall/F1 and visualization helpers.")
