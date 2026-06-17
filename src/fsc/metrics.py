### Original Code:
### Copyright (c) 2024 Presto Authors
### Licensed under the MIT License.
### A copy of the MIT License is available in the LICENSE file in the root directory of this project.

### Modifications by marlens123:
### - Extended to include regression and classification metrics.

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


def compute_regression_metrics(preds: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(target, preds)),
        "r2": float(r2_score(target, preds)),
        "mean_absolute_error": float(mean_absolute_error(target, preds)),
        "median_absolute_error": float(median_absolute_error(target, preds)),
    }


def compute_classification_metrics(preds: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "overall_accuracy": float(accuracy_score(target, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(target, preds)),
        "recall": float(recall_score(target, preds, average="weighted")),
        "precision": float(precision_score(target, preds, average="weighted")),
        "f1": float(f1_score(target, preds, average="weighted")),
    }


def compute_segmentation_metrics(preds: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "miou": float(mean_iou(preds, target, num_classes=10)),
    }


def class_wise_f1(y_pred, y_true, num_classes):
    return [f1_score(np.array(y_true) == i, np.array(y_pred) == i) for i in range(num_classes)]


def mean_iou(
    predictions: np.ndarray, labels: np.ndarray, num_classes: int, ignore_label: int = -1
):
    """
    Calculate mean IoU given prediction and label tensors, ignoring pixels with a specific label.

    Args:
    predictions (torch.Tensor): Predicted segmentation masks of shape (N, H, W)
    labels (torch.Tensor): Ground truth segmentation masks of shape (N, H, W)
    num_classes (int): Number of classes in the segmentation task
    ignore_label (int): Label value to ignore in IoU calculation (default: -1)

    Returns:
    float: Mean IoU across all classes
    """

    # Initialize tensors to store intersection and union for each class
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    # Create a mask for valid pixels (i.e., not ignore_label)
    valid_mask = labels != ignore_label

    # Iterate through each class
    for class_id in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (predictions == class_id) & valid_mask
        label_mask = (labels == class_id) & valid_mask

        # Calculate intersection and union
        intersection[class_id] = (pred_mask & label_mask).sum()
        union[class_id] = (pred_mask | label_mask).sum()

    # Calculate IoU for each class
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero

    # Calculate mean IoU (excluding classes with zero union)
    valid_classes = union > 0
    mean_iou = iou[valid_classes].mean()

    return mean_iou
