"""
evaluate.py

Evaluation metrics, full test evaluation, and visualization helpers.
"""

import os
from typing import Dict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_loader import create_dataloaders
from sod_model import SODNet


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


def evaluate_on_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Run full evaluation on a DataLoader (e.g. test_loader) and return
    average IoU, precision, recall, F1, and MAE.
    """
    model.eval()

    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)  # [B, 1, H, W], sigmoid output

            # IoU
            iou = compute_iou(preds, masks, threshold=threshold)

            # Precision / Recall / F1
            cls_metrics = compute_classification_metrics(preds, masks, threshold=threshold)

            # Mean Absolute Error
            mae = F.l1_loss(preds, masks).item()

            total_iou += iou
            total_precision += cls_metrics["precision"]
            total_recall += cls_metrics["recall"]
            total_f1 += cls_metrics["f1"]
            total_mae += mae
            num_batches += 1

    if num_batches == 0:
        return {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mae": 0.0,
        }

    return {
        "iou": total_iou / num_batches,
        "precision": total_precision / num_batches,
        "recall": total_recall / num_batches,
        "f1": total_f1 / num_batches,
        "mae": total_mae / num_batches,
    }


if __name__ == "__main__":
    # Full test-set evaluation
    data_root = os.path.join("data", "DUTS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for evaluation:", device)

    # Load dataloaders (only need test_loader, but function returns all three)
    _, _, test_loader = create_dataloaders(
        root_dir=data_root,
        img_size=128,
        batch_size=8,
    )

    # Build model and load best weights
    model = SODNet(in_channels=3, base_channels=32).to(device)

    ckpt_path = os.path.join("checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Please run train.py first to generate best_model.pth."
        )

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded best model from {ckpt_path}")

    # Run full evaluation on test set
    metrics = evaluate_on_loader(model, test_loader, device, threshold=0.5)

    print("\n=== Test Set Metrics ===")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"MAE:       {metrics['mae']:.4f}")
