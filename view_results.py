
import pandas as pd
import json

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# Model Comparison
print_section("MODEL COMPARISON - Architecture Performance")
try:
    model_df = pd.read_csv("outputs/model_comparison.csv")
    print(model_df.to_string(index=False))
except Exception as e:
    print(f"Error: {e}")

# Final Summary
print_section("FINAL SUMMARY - Single vs Multi-Thread Performance")
try:
    summary_df = pd.read_csv("outputs/final_summary.csv")
    print(summary_df.to_string(index=False))
except Exception as e:
    print(f"Error: {e}")

# Queue Comparison
print_section("QUEUE COMPARISON - Buffering Impact on Latency & Throughput")
try:
    queue_df = pd.read_csv("outputs/queue_comparison.csv")
    print(queue_df.to_string(index=False))
except Exception as e:
    print(f"Error: {e}")

# Stream Rate Comparison
print_section("STREAM RATE COMPARISON - Real-time Performance at Different FPS")
try:
    stream_df = pd.read_csv("outputs/stream_rate_comparison.csv")
    print(stream_df.to_string(index=False))
except Exception as e:
    print(f"Error: {e}")

# CPU vs GPU
print_section("CPU vs GPU COMPARISON - Hardware Performance Impact")
try:
    cpu_gpu_df = pd.read_csv("outputs/cpu_gpu_comparison.csv")
    print(cpu_gpu_df.to_string(index=False))
except Exception as e:
    print(f"Error: {e}")

# Detailed Benchmark JSON
print_section("DETAILED BENCHMARK RESULTS")
try:
    with open("outputs/benchmark_results.json") as f:
        bench_data = json.load(f)
        for key, value in bench_data.items():
            print(f"\n{key.upper()}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {value}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
