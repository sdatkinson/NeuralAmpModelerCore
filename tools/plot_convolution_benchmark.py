#!/usr/bin/env python3
"""
Visualization script for convolution benchmark results.

Usage:
    python plot_convolution_benchmark.py results.csv
    python plot_convolution_benchmark.py before.csv after.csv  # Compare two runs
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: This script requires pandas and matplotlib.")
    print("Install with: pip install pandas matplotlib")
    sys.exit(1)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    df = pd.read_csv(csv_path)
    # Convert ns to microseconds for readability
    df["mean_us"] = df["mean_ns"] / 1000
    df["stddev_us"] = df["stddev_ns"] / 1000
    df["min_us"] = df["min_ns"] / 1000
    df["max_us"] = df["max_ns"] / 1000
    return df


def plot_groups_vs_time(df: pd.DataFrame, conv_type: str, output_prefix: str):
    """Plot groups vs execution time for different channel counts."""
    type_df = df[df["type"] == conv_type]

    if type_df.empty:
        print(f"No data for {conv_type}")
        return

    frames_list = sorted(type_df["frames"].unique())
    channels_list = sorted(type_df["channels"].unique())

    for frames in frames_list:
        fig, ax = plt.subplots(figsize=(10, 6))

        for channels in channels_list:
            subset = type_df[(type_df["frames"] == frames) & (type_df["channels"] == channels)]
            if subset.empty:
                continue

            # Sort by groups
            subset = subset.sort_values("groups")

            ax.errorbar(
                subset["groups"],
                subset["mean_us"],
                yerr=subset["stddev_us"],
                marker="o",
                capsize=3,
                label=f"{channels} channels",
            )

        ax.set_xlabel("Number of Groups")
        ax.set_ylabel("Execution Time (microseconds)")
        ax.set_title(f"{conv_type}: Groups vs Time (frames={frames})")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f"{output_prefix}_{conv_type.lower()}_frames{frames}_groups_vs_time.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()


def plot_speedup_vs_baseline(df: pd.DataFrame, conv_type: str, output_prefix: str):
    """Plot speedup relative to groups=1 baseline."""
    type_df = df[df["type"] == conv_type]

    if type_df.empty:
        print(f"No data for {conv_type}")
        return

    frames_list = sorted(type_df["frames"].unique())
    channels_list = sorted(type_df["channels"].unique())

    for frames in frames_list:
        fig, ax = plt.subplots(figsize=(10, 6))

        for channels in channels_list:
            subset = type_df[(type_df["frames"] == frames) & (type_df["channels"] == channels)]
            if subset.empty:
                continue

            # Get baseline (groups=1)
            baseline = subset[subset["groups"] == 1]
            if baseline.empty:
                continue
            baseline_time = baseline["mean_us"].values[0]

            # Calculate speedup
            subset = subset.sort_values("groups")
            speedup = baseline_time / subset["mean_us"]

            ax.plot(
                subset["groups"],
                speedup,
                marker="o",
                label=f"{channels} channels",
            )

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (groups=1)")
        ax.set_xlabel("Number of Groups")
        ax.set_ylabel("Speedup (relative to groups=1)")
        ax.set_title(f"{conv_type}: Speedup vs Groups (frames={frames})")
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f"{output_prefix}_{conv_type.lower()}_frames{frames}_speedup.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()


def plot_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame, conv_type: str, output_prefix: str):
    """Compare before/after benchmark results."""
    before = df_before[df_before["type"] == conv_type]
    after = df_after[df_after["type"] == conv_type]

    if before.empty or after.empty:
        print(f"No data for {conv_type}")
        return

    frames_list = sorted(before["frames"].unique())
    channels_list = sorted(before["channels"].unique())

    for frames in frames_list:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Absolute times
        ax1 = axes[0]
        for channels in channels_list:
            before_subset = before[(before["frames"] == frames) & (before["channels"] == channels)]
            after_subset = after[(after["frames"] == frames) & (after["channels"] == channels)]

            if before_subset.empty or after_subset.empty:
                continue

            before_subset = before_subset.sort_values("groups")
            after_subset = after_subset.sort_values("groups")

            ax1.plot(
                before_subset["groups"],
                before_subset["mean_us"],
                marker="o",
                linestyle="--",
                alpha=0.7,
                label=f"{channels}ch (before)",
            )
            ax1.plot(
                after_subset["groups"],
                after_subset["mean_us"],
                marker="s",
                label=f"{channels}ch (after)",
            )

        ax1.set_xlabel("Number of Groups")
        ax1.set_ylabel("Execution Time (microseconds)")
        ax1.set_title(f"{conv_type}: Before vs After (frames={frames})")
        ax1.legend(fontsize=8)
        ax1.set_xscale("log", base=2)
        ax1.grid(True, alpha=0.3)

        # Right plot: Speedup (after vs before)
        ax2 = axes[1]
        for channels in channels_list:
            before_subset = before[(before["frames"] == frames) & (before["channels"] == channels)]
            after_subset = after[(after["frames"] == frames) & (after["channels"] == channels)]

            if before_subset.empty or after_subset.empty:
                continue

            # Merge on groups
            merged = pd.merge(
                before_subset[["groups", "mean_us"]],
                after_subset[["groups", "mean_us"]],
                on="groups",
                suffixes=("_before", "_after"),
            )

            speedup = merged["mean_us_before"] / merged["mean_us_after"]

            ax2.plot(
                merged["groups"],
                speedup,
                marker="o",
                label=f"{channels} channels",
            )

        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No change")
        ax2.set_xlabel("Number of Groups")
        ax2.set_ylabel("Speedup (before/after)")
        ax2.set_title(f"{conv_type}: Optimization Speedup (frames={frames})")
        ax2.legend(fontsize=8)
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = f"{output_prefix}_{conv_type.lower()}_frames{frames}_comparison.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize convolution benchmark results")
    parser.add_argument("csv_files", nargs="+", help="CSV file(s) with benchmark results")
    parser.add_argument("-o", "--output-prefix", default="benchmark", help="Output file prefix")
    args = parser.parse_args()

    if len(args.csv_files) == 1:
        # Single file mode
        df = load_results(args.csv_files[0])

        for conv_type in ["Conv1x1", "Conv1D"]:
            plot_groups_vs_time(df, conv_type, args.output_prefix)
            plot_speedup_vs_baseline(df, conv_type, args.output_prefix)

        print("\nSummary statistics:")
        print(df.groupby(["type", "channels", "groups"])["mean_us"].mean().unstack())

    elif len(args.csv_files) == 2:
        # Comparison mode
        df_before = load_results(args.csv_files[0])
        df_after = load_results(args.csv_files[1])

        for conv_type in ["Conv1x1", "Conv1D"]:
            plot_comparison(df_before, df_after, conv_type, args.output_prefix)

        # Calculate overall improvement
        print("\nOverall speedup (before/after):")
        for conv_type in ["Conv1x1", "Conv1D"]:
            before_mean = df_before[df_before["type"] == conv_type]["mean_us"].mean()
            after_mean = df_after[df_after["type"] == conv_type]["mean_us"].mean()
            print(f"  {conv_type}: {before_mean/after_mean:.2f}x")
    else:
        print("Error: Provide 1 or 2 CSV files")
        sys.exit(1)


if __name__ == "__main__":
    main()
