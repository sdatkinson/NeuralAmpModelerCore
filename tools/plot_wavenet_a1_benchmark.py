#!/usr/bin/env python3
"""
Script to plot wavenet_a1 benchmark results against the number of channels
in the first layer array of each model.
"""

import json
import glob
import os
import subprocess
import sys
import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib is required for plotting")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


def extract_channels(model_path):
    """Extract the number of channels from the first layer of a model."""
    with open(model_path, 'r') as f:
        data = json.load(f)
    return data['config']['layers'][0]['channels']


def run_benchmark_and_extract_median(model_path, num_runs=10):
    """Run benchmodel tool and extract median result."""
    # Get the project root (parent of tools/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    benchmodel_exe = project_root / "build" / "tools" / "benchmodel"
    
    if not benchmodel_exe.exists():
        print(f"Error: benchmodel not found at {benchmodel_exe}")
        print("Please build the project first")
        sys.exit(1)
    
    # Run benchmark multiple times
    results = []
    for i in range(1, num_runs + 1):
        print(f"  Run {i}/{num_runs}... ", end='', flush=True)
        try:
            output = subprocess.run(
                [str(benchmodel_exe), str(model_path)],
                capture_output=True,
                text=True,
                cwd=str(project_root)
            )
            
            # Extract double precision milliseconds value
            match = re.search(r'^(\d+\.\d+)ms$', output.stdout, re.MULTILINE)
            if match:
                ms = float(match.group(1))
                results.append(ms)
                print(f"{ms}ms")
            else:
                print("Failed to extract timing")
                print("Output was:", output.stdout)
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    # Calculate median
    results.sort()
    n = len(results)
    if n == 0:
        return None
    elif n % 2 == 0:
        median = (results[n//2 - 1] + results[n//2]) / 2
    else:
        median = results[n//2]
    
    return median


def main():
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    example_models_dir = project_root / "example_models"
    
    # Find all wavenet_a1 models
    model_files = sorted(glob.glob(str(example_models_dir / "wavenet_a1*.nam")))
    
    if not model_files:
        print(f"Error: No wavenet_a1*.nam files found in {example_models_dir}")
        sys.exit(1)
    
    print("==========================================")
    print("Wavenet A1 Models Benchmark & Plot")
    print("==========================================")
    print(f"Found {len(model_files)} models")
    print()
    
    # Extract channels and run benchmarks
    channels_list = []
    median_times = []
    model_names = []
    
    for model_path in model_files:
        model_name = Path(model_path).name
        print(f"Processing: {model_name}")
        
        # Extract channels
        channels = extract_channels(model_path)
        print(f"  Channels in first layer: {channels}")
        
        # Run benchmark
        print(f"  Running benchmark...")
        median = run_benchmark_and_extract_median(model_path)
        
        if median is None:
            print(f"  Error: Failed to benchmark {model_name}")
            continue
        
        channels_list.append(channels)
        median_times.append(median)
        model_names.append(model_name)
        print(f"  Median: {median}ms")
        print()
    
    if not channels_list:
        print("Error: No valid benchmark results collected")
        sys.exit(1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(channels_list, median_times, 'o', markersize=8)
    
    # Add red dashed horizontal line at 60ms with label
    plt.axhline(y=60, color='red', linestyle='--', linewidth=2, label='limit')
    
    # Label each point
    for i, (ch, time, name) in enumerate(zip(channels_list, median_times, model_names)):
        short_name = name.replace('wavenet_a1_', '').replace('.nam', '')
        plt.annotate(short_name, (ch, time), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Number of Channels (First Layer)', fontsize=12)
    plt.ylabel('Median Time (ms)', fontsize=12)
    plt.title('Wavenet A1 Models: Benchmark vs Channels', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make it look nice
    plt.tight_layout()
    
    # Save plot
    output_file = project_root / "wavenet_a1_benchmark_plot.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")
    
    # Display plot
    try:
        plt.show()
    except:
        print("Note: Could not display plot (no display available), but saved to file")
    
    # Print summary table
    print()
    print("==========================================")
    print("Summary")
    print("==========================================")
    print(f"{'Model':<30} {'Channels':<10} {'Median (ms)':<15}")
    print("-" * 55)
    for name, ch, time in zip(model_names, channels_list, median_times):
        print(f"{name:<30} {ch:<10} {time:<15.3f}")


if __name__ == "__main__":
    main()
