# -*- coding: utf-8 -*-
"""
benchmark.py
------------
Runs four experiments to understand how a video-frame classifier behaves
as a real-time streaming system.

Experiment 1 — Single-thread vs Multi-thread
    Does processing frames in a background thread improve throughput?

Experiment 2 — Queue Size
    How does the size of the buffer between producer and consumer
    affect latency and how many frames get dropped?

Experiment 3 — Stream Rate
    What happens when the camera sends frames faster than the model
    can process them?

Experiment 4 — CPU vs GPU
    How much faster is inference when a GPU is available?

Run order:
    python train.py        # must run first to create checkpoint + data split
    python benchmark.py

Outputs (saved to outputs/):
    final_summary.csv           — Experiment 1 results
    queue_comparison.csv        — Experiment 2 results
    stream_rate_comparison.csv  — Experiment 3 results
    cpu_gpu_comparison.csv      — Experiment 4 results
    benchmark_results.json      — all four results in one file
"""

import json
import queue
import threading
import time

import cv2
import pandas as pd
import torch

from config import (
    BENCHMARK_JSON,
    CHECKPOINT_PATH,
    CPU_GPU_CSV,
    FINAL_SUMMARY_CSV,
    LARGE_QUEUE,
    MODEL_INFERENCE_DELAY_MS,
    NUM_CLASSES,
    PROCESSING_DELAY_MS,
    QUEUE_COMPARE_CSV,
    QUEUE_SIZE_OPTIONS,
    SEED,
    STREAM_FPS,
    STREAM_FRAMES,
    STREAM_RATE_CSV,
    STREAM_RATE_OPTIONS,
    build_model,
    device,
    predict_frame,
    set_seed,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
set_seed(SEED)

# ── Load model ───────────────────────────────────────────────────────────────
# We always benchmark the MobileNetV3-Small checkpoint saved by train.py.
MODEL_NAME = "mobilenet_v3_small"

model = build_model(MODEL_NAME, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

print(f"Model     : {MODEL_NAME}")
print(f"Device    : {device}")
print(f"Checkpoint: {CHECKPOINT_PATH}")

# ── Load test image paths ─────────────────────────────────────────────────────
# train.py saves the data split so we use the exact same test images here.
with open("outputs/data_split.json") as f:
    split = json.load(f)

stream_paths = split["test_paths"][:STREAM_FRAMES]
print(f"Frames to stream: {len(stream_paths)}\n")


# =============================================================================
# Helper: summarise a results DataFrame into one readable row
# =============================================================================
# Metrics we keep (all others are dropped to avoid clutter):
#
#   throughput_fps    — frames fully processed per second (higher = better)
#   latency_mean_ms   — avg time from "frame arrived" to "prediction ready"
#   inference_mean_ms — avg time the neural network alone took (forward pass)
#   drop_rate         — fraction of frames discarded (0.0 = nothing dropped)
#   dropped_frames    — raw count of discarded frames

def summarize(df: pd.DataFrame, extra: dict = None) -> dict:
    if df.empty:
        return {}

    runtime_s        = float(df.attrs.get("runtime_s", 1))
    frames_processed = len(df)
    frames_input     = int(df.attrs.get("frames_input", frames_processed))
    dropped          = int(df.attrs.get("dropped_frames", 0))

    row = {
        "mode":              df["mode"].iloc[0],
        "throughput_fps":    round(frames_processed / runtime_s, 2),
        "latency_mean_ms":   round(df["latency_ms"].mean(), 2),
        "inference_mean_ms": round(df["inference_ms"].mean(), 2),
        "drop_rate":         round(dropped / max(frames_input, 1), 3),
        "dropped_frames":    dropped,
    }

    if extra:
        row.update(extra)

    return row


# =============================================================================
# run_single_thread
# =============================================================================

def run_single_thread(image_paths, mdl, fps=10) -> pd.DataFrame:
    """
    The simplest pipeline: read a frame → preprocess → inference → repeat.
    Everything happens sequentially in one thread — no parallelism at all.

    fps controls how fast the simulated camera sends frames. If the model
    is too slow to keep up, latency climbs but no frames are dropped
    (the loop just falls behind).
    """
    rows            = []
    frame_interval  = 1.0 / fps      # e.g. 0.1 s between frames at 10 FPS
    benchmark_start = time.perf_counter()

    for i, img_path in enumerate(image_paths):
        # Simulate the camera: wait until this frame's scheduled arrival time
        scheduled_time = benchmark_start + i * frame_interval
        wait = scheduled_time - time.perf_counter()
        if wait > 0:
            time.sleep(wait)

        # Read image from disk
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # Preprocess + inference
        pred_label, confidence, _, inference_ms = predict_frame(mdl, frame)

        latency_ms = (time.perf_counter() - scheduled_time) * 1000

        rows.append({
            "mode":            "single_thread",
            "frame_idx":       i,
            "predicted_class": pred_label,
            "confidence":      round(confidence, 3),
            "inference_ms":    round(inference_ms, 2),
            "latency_ms":      round(latency_ms, 2),
        })

    runtime_s = time.perf_counter() - benchmark_start
    df = pd.DataFrame(rows)
    df.attrs.update({"runtime_s": runtime_s,
                     "frames_input": len(image_paths),
                     "dropped_frames": 0,
                     "fps_requested": fps})
    return df


# =============================================================================
# run_multi_thread
# =============================================================================

def run_multi_thread(image_paths, mdl, fps=10,
                     max_queue_size=8, drop_policy="drop_newest") -> pd.DataFrame:
    """
    A producer-consumer pipeline using two threads and a shared queue.

    Producer thread  — reads frames from disk and puts them in the queue.
    Consumer thread  — pulls frames from the queue and runs inference.

    This mirrors a real camera setup: the camera keeps producing frames
    regardless of whether the model has finished the previous one.

    When the queue is full:
      'drop_newest'  — discard the incoming frame (default, low latency)
      'drop_oldest'  — discard the oldest queued frame to make room

    max_queue_size controls how many unprocessed frames can accumulate.
    A larger queue reduces drops but increases latency (frames wait longer).
    """
    frame_queue     = queue.Queue(maxsize=max_queue_size)
    rows            = []
    dropped_frames  = 0
    STOP            = object()    # sentinel: tells consumer "no more frames"
    frame_interval  = 1.0 / fps
    benchmark_start = time.perf_counter()

    # ── Producer ──────────────────────────────────────────────────────────────
    def producer():
        nonlocal dropped_frames
        for i, img_path in enumerate(image_paths):
            scheduled_time = benchmark_start + i * frame_interval
            wait = scheduled_time - time.perf_counter()
            if wait > 0:
                time.sleep(wait)

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            item = {
                "frame_idx":      i,
                "frame":          frame,
                "scheduled_time": scheduled_time,
                "enqueue_time":   time.perf_counter(),
            }

            if frame_queue.full():
                if drop_policy == "drop_oldest":
                    try:
                        frame_queue.get_nowait()   # evict oldest frame
                    except queue.Empty:
                        pass
                    dropped_frames += 1
                    frame_queue.put(item)
                else:                              # drop_newest (default)
                    dropped_frames += 1
            else:
                frame_queue.put(item)

        frame_queue.put(STOP)

    # ── Consumer ──────────────────────────────────────────────────────────────
    def consumer():
        while True:
            item = frame_queue.get()
            if item is STOP:
                break

            dequeue_time = time.perf_counter()
            pred_label, confidence, _, inference_ms = predict_frame(mdl, item["frame"])
            
            # Apply artificial delay if configured
            if MODEL_INFERENCE_DELAY_MS > 0:
                time.sleep(MODEL_INFERENCE_DELAY_MS / 1000.0)
            
            finish_time = time.perf_counter()

            # queue_wait_ms: time the frame spent sitting in the queue
            queue_wait_ms = (dequeue_time - item["enqueue_time"]) * 1000
            # latency_ms: total time from scheduled arrival to prediction done
            latency_ms    = (finish_time  - item["scheduled_time"]) * 1000
            
            # Apply processing delay between frames if configured
            if PROCESSING_DELAY_MS > 0:
                time.sleep(PROCESSING_DELAY_MS / 1000.0)

            rows.append({
                "mode":            "multi_thread",
                "frame_idx":       item["frame_idx"],
                "predicted_class": pred_label,
                "confidence":      round(confidence, 3),
                "inference_ms":    round(inference_ms, 2),
                "queue_wait_ms":   round(queue_wait_ms, 2),
                "latency_ms":      round(latency_ms, 2),
            })

    # ── Run both threads ──────────────────────────────────────────────────────
    p = threading.Thread(target=producer, daemon=True)
    c = threading.Thread(target=consumer, daemon=True)
    p.start(); c.start()
    p.join();  c.join()

    runtime_s = time.perf_counter() - benchmark_start
    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    df.attrs.update({"runtime_s": runtime_s,
                     "frames_input": len(image_paths),
                     "dropped_frames": dropped_frames,
                     "fps_requested": fps})
    return df


# =============================================================================
# Experiment 1 — Single-thread vs Multi-thread
# =============================================================================
print("=" * 60)
print("Experiment 1: Single-thread vs Multi-thread")
print("=" * 60)

single_df = run_single_thread(stream_paths, model, fps=STREAM_FPS)
print(f"  Single-thread: {len(single_df)} frames processed")

multi_df = run_multi_thread(stream_paths, model, fps=STREAM_FPS,
                             max_queue_size=LARGE_QUEUE, drop_policy="drop_newest")
print(f"  Multi-thread : {len(multi_df)} frames processed")

summary_df = pd.DataFrame([summarize(single_df), summarize(multi_df)])
summary_df.to_csv(FINAL_SUMMARY_CSV, index=False)
print(f"\nSaved → {FINAL_SUMMARY_CSV}")
print(summary_df.to_string(index=False))


# =============================================================================
# Experiment 2 — Queue Size
# =============================================================================
print("\n" + "=" * 60)
print("Experiment 2: Effect of Queue Size")
print("  Larger queue = more buffering → lower drop rate but higher latency")
print("=" * 60)

queue_rows = []
for qsize in QUEUE_SIZE_OPTIONS:
    df = run_multi_thread(stream_paths, model, fps=STREAM_FPS,
                          max_queue_size=qsize, drop_policy="drop_newest")
    queue_rows.append(summarize(df, extra={"queue_size": qsize}))
    print(f"  Queue size {qsize:2d}: dropped = {queue_rows[-1]['dropped_frames']}")

queue_compare_df = pd.DataFrame(queue_rows)
queue_compare_df.to_csv(QUEUE_COMPARE_CSV, index=False)
print(f"\nSaved → {QUEUE_COMPARE_CSV}")
print(queue_compare_df.to_string(index=False))


# =============================================================================
# Experiment 3 — Stream Rate
# =============================================================================
print("\n" + "=" * 60)
print("Experiment 3: Effect of Input Stream Rate (FPS)")
print("  Higher FPS = more frames per second; model may not keep up")
print("=" * 60)

stream_rows = []
for fps in STREAM_RATE_OPTIONS:
    df = run_multi_thread(stream_paths, model, fps=fps,
                          max_queue_size=LARGE_QUEUE, drop_policy="drop_newest")
    stream_rows.append(summarize(df, extra={"stream_fps": fps}))
    print(f"  {fps} FPS: dropped = {stream_rows[-1]['dropped_frames']}")

stream_rate_df = pd.DataFrame(stream_rows)
stream_rate_df.to_csv(STREAM_RATE_CSV, index=False)
print(f"\nSaved → {STREAM_RATE_CSV}")
print(stream_rate_df.to_string(index=False))


# =============================================================================
# Experiment 4 — CPU vs GPU
# =============================================================================
print("\n" + "=" * 60)
print("Experiment 4: CPU vs GPU Inference Speed")
print("  Using first 50 frames to keep this quick")
print("=" * 60)

SMALL_SAMPLE = stream_paths[:50]
cpu_gpu_rows = []

# -- CPU run ------------------------------------------------------------------
import config as _cfg

cpu_model = build_model(MODEL_NAME, NUM_CLASSES).cpu()
cpu_model.load_state_dict(model.state_dict())
cpu_model.eval()

_original_device = _cfg.device      # save so we can restore after
_cfg.device = torch.device("cpu")   # tell predict_frame to use CPU

cpu_df = run_single_thread(SMALL_SAMPLE, cpu_model, fps=STREAM_FPS)
cpu_gpu_rows.append(summarize(cpu_df, extra={"device": "cpu"}))
print("  CPU run complete")

# -- GPU run (skipped if no CUDA device is present) ---------------------------
if torch.cuda.is_available():
    _cfg.device = torch.device("cuda")

    gpu_model = build_model(MODEL_NAME, NUM_CLASSES).cuda()
    gpu_model.load_state_dict(model.state_dict())
    gpu_model.eval()

    gpu_df = run_single_thread(SMALL_SAMPLE, gpu_model, fps=STREAM_FPS)
    cpu_gpu_rows.append(summarize(gpu_df, extra={"device": "cuda"}))
    print("  GPU run complete")
else:
    print("  No CUDA GPU found — skipping GPU run")

_cfg.device = _original_device      # restore

cpu_gpu_df = pd.DataFrame(cpu_gpu_rows)
cpu_gpu_df.to_csv(CPU_GPU_CSV, index=False)
print(f"\nSaved → {CPU_GPU_CSV}")
print(cpu_gpu_df.to_string(index=False))


# =============================================================================
# Bundle all results into one JSON file
# =============================================================================
results_bundle = {
    "experiment_1_single_vs_multi":  summary_df.to_dict(orient="records"),
    "experiment_2_queue_size":       queue_compare_df.to_dict(orient="records"),
    "experiment_3_stream_rate":      stream_rate_df.to_dict(orient="records"),
    "experiment_4_cpu_vs_gpu":       cpu_gpu_df.to_dict(orient="records"),
}
with open(BENCHMARK_JSON, "w") as f:
    json.dump(results_bundle, f, indent=2)

print(f"\nAll results bundled → {BENCHMARK_JSON}")
print("\nBenchmarking complete.")