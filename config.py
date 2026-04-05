# -*- coding: utf-8 -*-
"""
config.py
---------
This file stores all shared settings and helper functions used by the
other scripts in the project.

Why this file is important:
- It keeps the project organized.
- It avoids repeating the same code in train.py, evaluate.py, benchmark.py, and visualize.py.
- If we want to change a class name, image size, path, or hyperparameter,
  we can change it here once and the whole project updates consistently.
"""
# Standard library imports
import os
import subprocess
from pathlib import Path

# Third-party libraries used across the pipeline
import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
# These are the seven target categories that the model will learn to classify.
CLASS_NAMES = [
    "Chain",
    "Clip",
    "Doubt",
    "Hold",
    "Hook",
    "No Instrument",
    "White Tube",
]

# Convert class name -> numeric label.
# Example: "Chain" -> 0
# ML models work with numbers, not text labels.
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Convert numeric label -> class name.
# Example: 0 -> "Chain"
# This is useful when we want to display model predictions in readable form.
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

# Only these file extensions will be treated as valid input images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Total number of classes in the classification task.
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# We check whether the code is running in Google Colab.
# This allows the same project to work both locally and in Colab.
try:
    from google.colab import drive  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# Choose the dataset location automatically.
# If we are in Colab and the Drive dataset exists, use that.
# Otherwise use a local folder called "Data".
if IN_COLAB and Path("/content/drive/MyDrive/Data").exists():
    DATA_DIR = Path("/content/drive/MyDrive/Data")
else:
    DATA_DIR = Path("Data")

# Create an output folder where all results, checkpoints, and CSV files will be stored.
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# File path where the best trained model checkpoint will be saved.
CHECKPOINT_PATH      = OUTPUT_DIR / "best_model.pt"

# CSV/JSON files used by the later stages of the project.
# These save benchmark results and summary tables for analysis.
BENCHMARK_CSV        = OUTPUT_DIR / "benchmark_results.csv"
MODEL_COMPARE_CSV    = OUTPUT_DIR / "model_comparison.csv"
FINAL_SUMMARY_CSV    = OUTPUT_DIR / "final_summary.csv"
QUEUE_COMPARE_CSV    = OUTPUT_DIR / "queue_comparison.csv"
STREAM_RATE_CSV      = OUTPUT_DIR / "stream_rate_comparison.csv"
CPU_GPU_CSV          = OUTPUT_DIR / "cpu_gpu_comparison.csv"
BENCHMARK_JSON       = OUTPUT_DIR / "benchmark_results.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEED             = 42   # Fixed random seed for reproducibility.
IMAGE_SIZE       = 256  # Input image size for the model (increased from 224 for better feature capture).
BATCH_SIZE       = 16   # Number of images processed in each training batch (reduced for better gradient stability).
EPOCHS           = 150  # Number of times the entire dataset is passed through the model during training (increased for convergence).
LEARNING_RATE    = 5e-4 # Learning rate for the optimizer (reduced for finer tuning).

# Settings used during benchmark experiments.
STREAM_FPS       = 15   # Frames per second for the video stream (increased from 10).
STREAM_FRAMES    = 100  # Number of frames to process in each stream iteration.

# Example queue sizes used to study buffering behavior.
SMALL_QUEUE      = 4
LARGE_QUEUE      = 16

# Candidate queue sizes for experiment 2 in benchmark.py.
QUEUE_SIZE_OPTIONS  = [4, 8, 16, 32]
# Candidate stream rates for experiment 3 in benchmark.py.
STREAM_RATE_OPTIONS = [15, 25, 35]

# Delay parameters for testing system robustness
PROCESSING_DELAY_MS = 0    # Milliseconds delay to introduce between frame processing (0 = no delay)
MODEL_INFERENCE_DELAY_MS = 0  # Additional delay for inference simulation (0 = no artificial delay)

# Number of CPU worker processes that could be used for loading data.
# This keeps the value small so the project stays lightweight and stable.
NUM_CPU_WORKERS = max(0, min(2, (os.cpu_count() or 2) - 1)) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# Transform pipeline used during training.
train_transform = transforms.Compose([ # Data augmentation and preprocessing steps for training images.
    transforms.Resize((320, 320)), #Resize the image to a larger size before cropping.
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)), # Randomly crop a region of the image and resize it to the target size.
    transforms.RandomHorizontalFlip(p=0.5), # Random left-right flip for data augmentation 
    transforms.RandomRotation(20), # Random rotation for robustness (increased to 20 degrees)
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # Enhanced color augmentation
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)), # Random translation (increased)
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # Blur augmentation for robustness
    transforms.ToTensor(), # convert the PIL image to a PyTorch tensor and scale pixel values to [0, 1].
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize the image using mean and std values from ImageNet dataset (common for pretrained models).
                         [0.229, 0.224, 0.225]),
])

# Transform pipeline used during validation, testing, and inference
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize the image to the target size without cropping or augmentation.
    transforms.ToTensor(), # Convert the PIL image to a PyTorch tensor and scale pixel values to [0, 1].
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize the image using mean and std values from ImageNet dataset (common for pretrained models).
                         [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None: # Set random seeds for reproducibility across numpy and PyTorch.
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_open_image(path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB") # Open the image and convert it to RGB format (3 color channels). This ensures consistency even if some images are grayscale or have an alpha channel.
    except Exception:
        return Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)) # Return a blank image if the file cannot be opened, to avoid crashing the pipeline.


def collect_samples(data_dir: Path, class_names: list) -> list:
    """Return list of (file_path_str, class_idx) tuples."""
    samples = []
    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: missing class folder -> {class_dir}")
            continue
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(file_path), CLASS_TO_IDX[class_name]))
    return samples


def count_parameters(model: nn.Module) -> int: # Count the total number of trainable parameters in the model. This is a common metric to understand model complexity.
    return sum(p.numel() for p in model.parameters())

def get_gpu_stats():
    """
    Return GPU utilization and GPU memory usage if CUDA is available.
    Output:
    - gpu_utilization_percent
    - gpu_memory_mb
    """
    if not torch.cuda.is_available():
        return None, 0.0
    gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        gpu_util, gpu_mem_used = [float(x.strip()) for x in output.split(",")]
        return gpu_util, gpu_mem_used
    except Exception:
        return None, gpu_mem_mb


def get_resource_usage(process=None):
    """
    Measure current CPU, RAM, and GPU usage.

    Returns:
    - cpu_percent
    - ram_mb
    - gpu_mem_mb
    - gpu_util_percent
    """
    process = process or psutil.Process(os.getpid()) # Get the current process to measure its resource usage.
    cpu_count = max(psutil.cpu_count(logical=True) or 1, 1) #Logical CPU count is used to normalize CPU usage.
    cpu_percent = process.cpu_percent(interval=None) / cpu_count # Divide by cpu_count so usage is easier to interpret on multicore systems.
    ram_mb = process.memory_info().rss / (1024 * 1024)  # Resident memory used by the current Python process.
    gpu_util_percent, gpu_mem_mb = get_gpu_stats()  # GPU stats come from the helper above.
    return cpu_percent, ram_mb, gpu_mem_mb, gpu_util_percent

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Create a pretrained model and replace its final classification layer."""
    try:
        if model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT) # load the pretrained MobileNetv3 small model with default weights.
            in_features = model.classifier[-1].in_features # Get the number of input features to the final classification layer (this is needed to replace it correctly).
            model.classifier[-1] = nn.Linear(in_features, num_classes) # Replace the final classification layer with a new one that has the correct number of output classes for our task.
        elif model_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # load the pretrained ResNet18 model with default weights.
            in_features = model.fc.in_features   # Get the number of input features to the final classification layer (this is needed to replace it correctly).
            model.fc = nn.Linear(in_features, num_classes) # Replace the final classification layer with a new one that has the correct number of output classes for our task.
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception:
        # Fallback: no pretrained weights
        if model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    return model

# ---------------------------------------------------------------------------
# Frame inference helpers (used by benchmark.py)
# ---------------------------------------------------------------------------

def preprocess_frame(frame_bgr, transform=None, target_device=None):
    transform = transform or eval_transform
    target_device = target_device or device
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # convert the frame from BGR format (used by OpenCV) to RGB format (used by PIL and PyTorch).
    pil_image = Image.fromarray(frame_rgb) # Convert NumPy image to PIL image so torchvision transforms can be applied.
    tensor = transform(pil_image).unsqueeze(0).to(target_device) # Apply preprocessing, add batch dimension, and move to CPU/GPU.
    return tensor

def predict_frame(model, frame_bgr, target_device=None):
    import time
    target_device = target_device or device

# Measure preprocessing time separately.
    preprocess_start = time.perf_counter()
    tensor = preprocess_frame(frame_bgr, target_device=target_device)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

# Synchronize before timing GPU inference so the measurement is accurate.
    if target_device.type == "cuda":
        torch.cuda.synchronize()

# Measure model forward-pass time.
    inference_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

# Synchronize again so GPU timing finishes before we stop the timer.       
    if target_device.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - inference_start) * 1000

# Convert numeric prediction back to a readable class name.
    return IDX_TO_CLASS[pred.item()], float(confidence.item()), preprocess_ms, inference_ms