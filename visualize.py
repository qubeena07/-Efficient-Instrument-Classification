
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BENCHMARK_CSV,
    CPU_GPU_CSV,
    FINAL_SUMMARY_CSV,
    MODEL_COMPARE_CSV,
    QUEUE_COMPARE_CSV,
    STREAM_RATE_CSV,
)



def load(path, label):
    try:
        df = pd.read_csv(path)
        print(f"Loaded {label}: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"[Warning] {label} not found at {path} — skipping related plots.")
        return None



benchmark_df     = load(BENCHMARK_CSV,     "benchmark_results")
summary_df       = load(FINAL_SUMMARY_CSV, "final_summary")
model_compare_df = load(MODEL_COMPARE_CSV, "model_comparison")
queue_compare_df = load(QUEUE_COMPARE_CSV, "queue_comparison")
stream_rate_df   = load(STREAM_RATE_CSV,   "stream_rate_comparison")
cpu_gpu_df       = load(CPU_GPU_CSV,       "cpu_gpu_comparison")



if benchmark_df is not None and "mode" in benchmark_df.columns:
    single_lat = benchmark_df.loc[benchmark_df["mode"] == "single_thread", "latency_ms"]
    multi_lat  = benchmark_df.loc[benchmark_df["mode"] == "multi_thread",  "latency_ms"]

    plt.figure(figsize=(8, 4))
    plt.hist(single_lat, bins=20, alpha=0.7, label="Single Thread")
    plt.hist(multi_lat,  bins=20, alpha=0.7, label="Multi Thread")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frames")
    plt.title("Latency Histogram — Single vs Multi Thread")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/01_latency_histogram.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/01_latency_histogram.png")
    plt.close()



if summary_df is not None and "mode" in summary_df.columns:
    plt.figure(figsize=(7, 4))
    plt.bar(summary_df["mode"], summary_df["throughput_fps"])
    plt.ylabel("Throughput (FPS)")
    plt.title("Single vs Multi-Thread Throughput")
    plt.tight_layout()
    plt.savefig("outputs/02_throughput_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/02_throughput_comparison.png")
    plt.close()

    print("\n=== Single vs Multi-Thread Summary ===")
    print(summary_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Queue size tradeoff
# ---------------------------------------------------------------------------

if queue_compare_df is not None and "queue_size" in queue_compare_df.columns:
    plt.figure(figsize=(7, 4))
    plt.plot(queue_compare_df["queue_size"],
             queue_compare_df["latency_mean_ms"], marker="o", label="Mean Latency (ms)")
    plt.plot(queue_compare_df["queue_size"],
             queue_compare_df["throughput_fps"],  marker="s", label="Throughput (FPS)")
    plt.xlabel("Queue Size")
    plt.title("Queue Size Trade-off: Latency vs Throughput")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/03_queue_tradeoff.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/03_queue_tradeoff.png")
    plt.close()

    print("\n=== Queue Comparison ===")
    print(queue_compare_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Stream rate stress test
# ---------------------------------------------------------------------------

if stream_rate_df is not None and "stream_fps" in stream_rate_df.columns:
    plt.figure(figsize=(7, 4))
    plt.plot(stream_rate_df["stream_fps"], stream_rate_df["throughput_fps"],
             marker="o", label="Throughput (FPS)")
    plt.plot(stream_rate_df["stream_fps"], stream_rate_df["drop_rate"],
             marker="s", label="Drop Rate")
    plt.xlabel("Requested Input Stream FPS")
    plt.ylabel("FPS / Drop Rate")
    plt.title("Stream Rate Stress Test: Scaling & Robustness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/04_stream_rate_stress.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/04_stream_rate_stress.png")
    plt.close()

    print("\n=== Stream Rate Comparison ===")
    print(stream_rate_df.to_string(index=False))



if cpu_gpu_df is not None and "device" in cpu_gpu_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(cpu_gpu_df["device"], cpu_gpu_df["throughput_fps"])
    axes[0].set_title("Throughput (FPS)")
    axes[0].set_ylabel("FPS")

    axes[1].bar(cpu_gpu_df["device"], cpu_gpu_df["latency_mean_ms"])
    axes[1].set_title("Mean Latency (ms)")
    axes[1].set_ylabel("ms")

    plt.suptitle("CPU vs GPU Hardware Acceleration Impact")
    plt.tight_layout()
    plt.savefig("outputs/05_cpu_vs_gpu.png", dpi=150, bbox_inches="tight")
    print("Saved: outputs/05_cpu_vs_gpu.png")
    plt.close()

    print("\n=== CPU vs GPU Comparison ===")
    print(cpu_gpu_df.to_string(index=False))



if model_compare_df is not None:
    print("\n=== Model Comparison ===")
    print(model_compare_df.to_string(index=False))

    if "avg_inference_ms" in model_compare_df.columns:
        plt.figure(figsize=(7, 4))
        plt.bar(model_compare_df["model"], model_compare_df["avg_inference_ms"], color="steelblue")
        plt.ylabel("Avg Inference (ms)")
        plt.title("Model Inference Speed Comparison")
        plt.tight_layout()
        plt.savefig("outputs/06_model_inference_speed.png", dpi=150, bbox_inches="tight")
        print("Saved: outputs/06_model_inference_speed.png")
        plt.close()

    if "accuracy" in model_compare_df.columns:
        valid_acc = model_compare_df.dropna(subset=["accuracy"])
        if len(valid_acc) > 0:
            plt.figure(figsize=(7, 4))
            plt.bar(valid_acc["model"], valid_acc["accuracy"], color="green", alpha=0.7)
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)
            plt.title("Model Test Accuracy Comparison")
            plt.tight_layout()
            plt.savefig("outputs/07_model_accuracy.png", dpi=150, bbox_inches="tight")
            print("Saved: outputs/07_model_accuracy.png")
            plt.close()

    if "macro_f1" in model_compare_df.columns:
        valid_f1 = model_compare_df.dropna(subset=["macro_f1"])
        if len(valid_f1) > 0:
            plt.figure(figsize=(7, 4))
            plt.bar(valid_f1["model"], valid_f1["macro_f1"], color="orange", alpha=0.7)
            plt.ylabel("Macro F1")
            plt.ylim(0, 1)
            plt.title("Model Macro-F1 Comparison")
            plt.tight_layout()
            plt.savefig("outputs/08_model_macro_f1.png", dpi=150, bbox_inches="tight")
            print("Saved: outputs/08_model_macro_f1.png")
            plt.close()
    
    # Model parameters comparison
    if "parameters" in model_compare_df.columns:
        plt.figure(figsize=(7, 4))
        plt.bar(model_compare_df["model"], model_compare_df["parameters"] / 1e6, color="purple", alpha=0.7)
        plt.ylabel("Parameters (Millions)")
        plt.title("Model Size Comparison")
        plt.tight_layout()
        plt.savefig("outputs/09_model_size.png", dpi=150, bbox_inches="tight")
        print("Saved: outputs/09_model_size.png")
        plt.close()



print("\n=== Key OS Findings ===")
print("1. Throughput is computed using true wall-clock runtime.")
print("2. Queue wait time separates buffering delay from actual service time.")
print("3. Queue-size experiments show the throughput-latency trade-off.")
print("4. Stream-rate experiments show overload behaviour and dropped frames.")
print("5. CPU vs GPU comparison shows the effect of hardware on system performance.")