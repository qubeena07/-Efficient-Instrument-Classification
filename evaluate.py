# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
Loads the saved checkpoint (outputs/best_model.pt) and evaluates on the
test split produced by train.py.  Prints a classification report, confusion
matrix, and saves outputs/model_comparison.csv with accuracy / F1 columns.

Usage:
    python evaluate.py
    python evaluate.py --model resnet18   # to evaluate the other architecture
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    MODEL_COMPARE_CSV,
    NUM_CLASSES,
    NUM_CPU_WORKERS,
    SEED,
    build_model,
    device,
    eval_transform,
    safe_open_image,
    set_seed,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="mobilenet_v3_small",
    choices=["mobilenet_v3_small", "resnet18"],
    help="Architecture whose checkpoint to evaluate (default: mobilenet_v3_small)",
)
parser.add_argument(
    "--checkpoint",
    default=str(CHECKPOINT_PATH),
    help="Path to .pt checkpoint file",
)
args = parser.parse_args()

set_seed(SEED)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = safe_open_image(self.paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, self.paths[idx]

# ---------------------------------------------------------------------------
# Load split
# ---------------------------------------------------------------------------

with open("outputs/data_split.json") as f:
    split = json.load(f)

test_paths  = split["test_paths"]
test_labels = split["test_labels"]

test_dataset = FrameDataset(test_paths, test_labels, eval_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test samples: {len(test_paths)}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model = build_model(args.model, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.eval()
print(f"Loaded {args.model} checkpoint from: {args.checkpoint}")
print(f"Running on: {device}")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader):
    all_targets, all_preds, all_probs, all_paths = [], [], [], []

    with torch.no_grad():
        for images, labels, paths in loader:
            images  = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = outputs.argmax(dim=1)

            all_targets.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    return {
        "accuracy":  accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "macro_f1":  f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "y_true":    all_targets,
        "y_pred":    all_preds,
        "probs":     np.array(all_probs),
        "paths":     all_paths,
    }


results = evaluate_model(model, test_loader)

print("\n=== Classification Report ===")
print(classification_report(
    results["y_true"],
    results["y_pred"],
    target_names=CLASS_NAMES,
    digits=4,
))

report_df = pd.DataFrame(classification_report(
    results["y_true"],
    results["y_pred"],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)).T
print(report_df)

# ---------------------------------------------------------------------------
# Update model_comparison.csv with accuracy / F1 columns
# ---------------------------------------------------------------------------

try:
    cmp_df = pd.read_csv(MODEL_COMPARE_CSV)
    mask   = cmp_df["model"] == args.model
    cmp_df.loc[mask, "accuracy"] = results["accuracy"]
    cmp_df.loc[mask, "macro_f1"] = results["macro_f1"]
    cmp_df.to_csv(MODEL_COMPARE_CSV, index=False)
    print(f"\nUpdated {MODEL_COMPARE_CSV} with evaluation metrics.")
except FileNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

cm = confusion_matrix(results["y_true"], results["y_pred"])

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix — {args.model}")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Prediction examples
# ---------------------------------------------------------------------------

def show_prediction_examples(results, correct=True, n=6):
    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])
    probs  = results["probs"]
    paths  = results["paths"]

    mask    = (y_true == y_pred) if correct else (y_true != y_pred)
    indices = np.where(mask)[0][:n]

    label = "Correct" if correct else "Incorrect"
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        image      = safe_open_image(paths[idx])
        confidence = probs[idx].max()

        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(
            f"True: {IDX_TO_CLASS[y_true[idx]]}\n"
            f"Pred: {IDX_TO_CLASS[y_pred[idx]]}\n"
            f"Conf: {confidence:.2f}"
        )
        plt.axis("off")

    plt.suptitle(f"{label} Predictions", fontsize=13)
    plt.tight_layout()
    plt.show()


show_prediction_examples(results, correct=True,  n=6)
show_prediction_examples(results, correct=False, n=6)

print(f"\nAccuracy : {results['accuracy']:.4f}")
print(f"Macro F1 : {results['macro_f1']:.4f}")