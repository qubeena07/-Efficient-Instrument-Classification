# Real-Time Efficient Instrument Classification Project

A comprehensive deep learning project for classifying seven types of instruments in video streams using PyTorch. This project benchmarks model performance across various real-time scenarios including single/multi-threaded processing, queue buffering analysis, and CPU/GPU comparisons.

## 📋 Project Overview

This project trains and evaluates deep learning models (MobileNetV3-Small and ResNet18) to classify instruments from video frames. It includes extensive benchmarking to understand how these models perform as real-time streaming systems under different conditions.

### Classification Classes
- Chain
- Clip
- Doubt
- Hold
- Hook
- No Instrument
- White Tube

### Key Metrics (MobileNetV3-Small)
- **Test Accuracy:** 86.36%
- **Macro F1 Score:** 0.44
- **Model Size:** 5.9 MB (1.52M parameters)
- **Inference Time:** 4.55 ms (avg)
- **Real-time Throughput:** 10 FPS

## � Dataset

**Surgical Instrument Classification Dataset:** https://www.kaggle.com/datasets/debeshjha1/surgical-instrument-classification

This project uses a comprehensive dataset of surgical instrument images for training and evaluation of the classification models.

## �🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU support)
- See `requirements.txt` for all dependencies

### Installation

```bash
# Clone or navigate to the project directory
cd /path/to/project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torch-vision
pip install opencv-python pillow scikit-learn pandas matplotlib psutil
```

### Directory Structure

```
.
├── train.py                      # Train models (MobileNetV3-Small, ResNet18)
├── evaluate.py                   # Evaluate trained models
├── benchmark.py                  # Run 4 real-time performance experiments
├── visualize.py                  # Create benchmark visualizations
├── view_results.py               # Display results summary
├── config.py                     # Centralized configuration & utilities
├── RESULTS_SUMMARY.md            # Performance results & findings
│
├── Data/                         # Training data (7 instrument classes)
│   ├── Chain/
│   ├── Clip/
│   ├── Doubt/
│   ├── Hold/
│   ├── Hook/
│   ├── No Instrument/
│   └── White Tube/
│
└── outputs/                      # Generated outputs
    ├── best_model.pt            # Best trained model checkpoint
    ├── model_comparison.csv      # Model architecture comparison
    ├── final_summary.csv         # Single vs Multi-thread results
    ├── queue_comparison.csv      # Queue size impact analysis
    ├── stream_rate_comparison.csv # Stream rate stress test
    ├── cpu_gpu_comparison.csv    # CPU vs GPU performance
    ├── data_split.json           # Train/val/test split info
    └── benchmark_results.json    # All benchmark results
```

## 📚 Usage Guide

### 1. Train Models

```bash
python train.py
```

**What it does:**
- Loads images from the `Data/` directory organized by class
- Splits data into train/validation/test sets
- Trains MobileNetV3-Small (primary) and ResNet18 (comparison)
- Uses data augmentation and class-weighted sampling
- Saves best model to `outputs/best_model.pt`
- Generates `outputs/model_comparison.csv`

**Configuration in `config.py`:**
- `BATCH_SIZE`: 32
- `EPOCHS`: 50
- `LEARNING_RATE`: 0.001
- `IMAGE_SIZE`: 224×224

### 2. Evaluate Model

```bash
python evaluate.py
```

**What it does:**
- Loads the trained checkpoint (`outputs/best_model.pt`)
- Evaluates on test set
- Generates classification report and confusion matrix
- Outputs detailed metrics (accuracy, precision, recall, F1)

**Optional flags:**
```bash
python evaluate.py --model resnet18  # Evaluate ResNet18 instead
```

### 3. Run Benchmarks

```bash
python benchmark.py
```

**Runs 4 experiments:**

1. **Single-thread vs Multi-thread**
   - Tests producer-consumer pattern with threading
   - Measures throughput, latency, and inference time
   - Output: `outputs/final_summary.csv`
   - **Finding:** 6.1% latency improvement with multi-threading

2. **Queue Size Impact**
   - Tests buffer sizes: 2, 8, 16
   - Analyzes buffer-induced latency vs frame loss
   - Output: `outputs/queue_comparison.csv`
   - **Finding:** Queue size 16 achieves optimal 18.00 ms latency with 0% drop

3. **Stream Rate Performance**
   - Stress tests with varying FPS rates
   - Identifies system bottlenecks
   - Output: `outputs/stream_rate_comparison.csv`

4. **CPU vs GPU Comparison**
   - Measures inference speed on CPU and GPU
   - Output: `outputs/cpu_gpu_comparison.csv`

### 4. View Results

```bash
python view_results.py
```

Displays all results in formatted tables:
- Model comparison metrics
- Single vs multi-thread performance
- Queue buffering impact
- Stream rate analysis
- CPU vs GPU speed comparison

### 5. Visualize Benchmarks

```bash
python visualize.py
```

Generates plots and visualizations of:
- Model performance metrics
- Threading comparison charts
- Queue size impact graphs
- Stream rate performance curves
- CPU vs GPU latency comparison

Outputs saved as image files for detailed analysis.

## 📊 Results Summary

### Model Performance

| Model | Accuracy | F1 Score | Params | Inference |
|-------|----------|----------|--------|-----------|
| MobileNetV3-Small | 86.36% | 0.44 | 1.52M | 4.55 ms |
| ResNet18 | - | - | 11.18M | 2.48 ms |

### Real-Time Performance

**Multi-threaded mode (optimal):**
- Throughput: 10.09 FPS
- Latency: 18.23 ms
- Frame Loss: 0%

**Queue Size 16 (recommended):**
- Stable throughput across queue sizes
- Optimal latency: 18.00 ms
- Zero frame dropping

See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for detailed results.

## 🔧 Configuration

All settings are centralized in `config.py`:

```python
# Data settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Training
EPOCHS = 50
LEARNING_RATE = 0.001

# Paths
DATA_DIR = "Data"
CHECKPOINT_PATH = "outputs/best_model.pt"
```

Modify as needed for your hardware and requirements.

## 🎯 Key Features

- **Multi-Model Comparison:** Compare MobileNetV3-Small vs ResNet18
- **Real-Time Streaming:** Full benchmarking suite for real-time performance
- **Threading Support:** Producer-consumer pattern with queue-based buffering
- **Hardware Flexibility:** Works on CPU and GPU with automatic detection
- **Comprehensive Metrics:** Accuracy, F1, precision, recall, inference time, throughput
- **Data Augmentation:** Random crops, flips, and color jittering for robust training
- **Class Balancing:** Weighted sampling to handle imbalanced datasets

## 📈 Understanding the Results

### Throughput
Frames per second (FPS) the system can process steadily.

### Latency
Time from frame acquisition to prediction output (includes model inference + queue overhead).

### Inference Time
Pure model forward pass time (excludes queue/threading overhead).

### Drop Rate
Percentage of frames lost when queue buffer fills up.

## 🤝 Contributing

To modify or extend this project:

1. Update hyperparameters in `config.py`
2. Modify model architecture in `config.py` `build_model()` function
3. Add custom benchmarks to `benchmark.py`
4. Add new classes by creating subdirectories in `Data/`

## � Project Link

**EICC Challenge:** https://perceptionintelligencelab.github.io/EICC-Challenge/#home

This project is part of the EICC ( Efficient Instrument Classification Challenge) initiative.

## �📝 License

MIT License

Copyright (c) 2026 Dipika Ranabhat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 📧 Contact

For questions or collaborations, reach out at: **dipika.ranabhat2001@gmail.com**

---

**Last Updated:** March 2026
**Best Model:** MobileNetV3-Small (86.36% accuracy, 5.9 MB)
