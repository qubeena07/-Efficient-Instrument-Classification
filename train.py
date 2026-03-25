# -*- coding: utf-8 -*-
"""
train.py
--------
Loads data, trains MobileNetV3-Small and ResNet18, saves:
  - outputs/best_model.pt          (MobileNetV3 checkpoint)
  - outputs/model_comparison.csv   (accuracy / F1 / speed per model)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DATA_DIR,
    EPOCHS,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_COMPARE_CSV,
    NUM_CLASSES,
    NUM_CPU_WORKERS,
    SEED,
    build_model,
    collect_samples,
    count_parameters,
    device,
    eval_transform,
    safe_open_image,
    set_seed,
    train_transform,
)

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

set_seed(SEED)
print("Data dir:", DATA_DIR)
print("Device  :", device)

# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
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
# Data split
# ---------------------------------------------------------------------------

samples = collect_samples(DATA_DIR, CLASS_NAMES)
print("Total images:", len(samples))

paths  = [p for p, _ in samples]
labels = [y for _, y in samples]

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths, labels, test_size=0.30, random_state=SEED, stratify=labels
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=1 / 3, random_state=SEED, stratify=temp_labels
)

print("Train      :", len(train_paths))
print("Validation :", len(val_paths))
print("Test       :", len(test_paths))

# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

count_df = pd.DataFrame({
    "class_name": CLASS_NAMES,
    "count": [Counter(labels)[i] for i in range(NUM_CLASSES)],
})
print(count_df)

plt.figure(figsize=(8, 4))
plt.bar(count_df["class_name"], count_df["count"])
plt.xticks(rotation=30, ha="right")
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Sample visualisation
# ---------------------------------------------------------------------------

def show_samples(paths, labels, n=6):
    plt.figure(figsize=(12, 6))
    for i in range(min(n, len(paths))):
        plt.subplot(2, 3, i + 1)
        image = safe_open_image(paths[i])
        plt.imshow(image)
        plt.title(IDX_TO_CLASS[labels[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(train_paths, train_labels, n=6)

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

train_dataset = FrameDataset(train_paths, train_labels, train_transform)
val_dataset   = FrameDataset(val_paths,   val_labels,   eval_transform)
test_dataset  = FrameDataset(test_paths,  test_labels,  eval_transform)

train_counts   = Counter(train_labels)
sample_weights = [1.0 / train_counts[lbl] for lbl in train_labels]
sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, all_targets, all_preds = 0.0, [], []

    for images, batch_labels, _ in loader:
        images       = images.to(device)
        batch_labels = batch_labels.to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss    = criterion(outputs, batch_labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        all_targets.extend(batch_labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc


def train_model(model_name):
    model     = build_model(model_name, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.5)

    history        = []
    best_val_loss  = float("inf")
    best_state     = None
    early_stopping_patience = 50
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion)
        scheduler.step(val_loss)

        history.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_acc,
            "val_acc":    val_acc,
        })

        print(
            f"{model_name} | Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement.")
                break 

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

# ---------------------------------------------------------------------------
# Train both models
# ---------------------------------------------------------------------------
import time
from sklearn.metrics import accuracy_score, f1_score

trained_models  = {}
history_tables  = {}

for model_name in ["mobilenet_v3_small", "resnet18"]:
    mdl, hist = train_model(model_name)
    trained_models[model_name] = mdl
    history_tables[model_name] = hist

# Save MobileNet checkpoint (default best model)
torch.save(trained_models["mobilenet_v3_small"].state_dict(), CHECKPOINT_PATH)
print("Saved checkpoint to:", CHECKPOINT_PATH)

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

for model_name, history_df in history_tables.items():
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"],   label="Val Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Acc")
    plt.plot(history_df["epoch"], history_df["val_acc"],   label="Val Acc")
    plt.title(f"{model_name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Quick inference-speed comparison and save model_comparison.csv
# ---------------------------------------------------------------------------

def measure_inference_time(model, image_size=224, runs=30):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


comparison_rows = []
for model_name, mdl in trained_models.items():
    comparison_rows.append({
        "model":              model_name,
        "parameters":         count_parameters(mdl),
        "avg_inference_ms":   measure_inference_time(mdl, IMAGE_SIZE),
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(MODEL_COMPARE_CSV, index=False)
print("Saved model comparison to:", MODEL_COMPARE_CSV)
print(comparison_df)

# ---------------------------------------------------------------------------
# Persist split paths so other scripts can reload them
# ---------------------------------------------------------------------------

import json

split_path = "outputs/data_split.json"
with open(split_path, "w") as f:
    json.dump({
        "train_paths":  train_paths,
        "val_paths":    val_paths,
        "test_paths":   test_paths,
        "train_labels": train_labels,
        "val_labels":   val_labels,
        "test_labels":  test_labels,
    }, f)

print("Saved data split to:", split_path)
print("Training complete.")