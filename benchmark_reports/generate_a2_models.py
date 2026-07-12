#!/usr/bin/env python3
"""Generate A2-shaped .nam files for bench_a2_fast benchmarking."""

import json
import random
from pathlib import Path

K_NUM_LAYERS = 23
K_HEAD_KERNEL_SIZE = 16
K_LEAKY_SLOPE = 0.01
K_KERNEL_SIZES = [6] * 14 + [15, 15] + [6] * 7
K_DILATIONS = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]


def build_a2_config(channels: int) -> dict:
    film_inactive = {"active": False, "shift": True, "groups": 1}
    layer = {
        "input_size": 1,
        "condition_size": 1,
        "channels": channels,
        "bottleneck": channels,
        "kernel_sizes": K_KERNEL_SIZES,
        "dilations": K_DILATIONS,
        "activation": [{"type": "LeakyReLU", "negative_slope": K_LEAKY_SLOPE} for _ in range(K_NUM_LAYERS)],
        "gating_mode": ["none"] * K_NUM_LAYERS,
        "secondary_activation": [None] * K_NUM_LAYERS,
        "head": {"out_channels": 1, "kernel_size": K_HEAD_KERNEL_SIZE, "bias": True},
        "head1x1": {"active": False, "out_channels": 1, "groups": 1},
        "layer1x1": {"active": True, "groups": 1},
        "conv_pre_film": film_inactive,
        "conv_post_film": film_inactive,
        "input_mixin_pre_film": film_inactive,
        "input_mixin_post_film": film_inactive,
        "activation_pre_film": film_inactive,
        "activation_post_film": film_inactive,
        "layer1x1_post_film": film_inactive,
        "head1x1_post_film": film_inactive,
        "groups_input": 1,
        "groups_input_mixin": 1,
    }
    return {"layers": [layer], "head_scale": 0.01}


def a2_weight_count(channels: int) -> int:
    bn = channels
    total = channels
    for i in range(K_NUM_LAYERS):
        k = K_KERNEL_SIZES[i]
        total += bn * channels * k + bn  # conv1d
        total += bn  # input mixin
        total += channels * bn + channels  # layer1x1
    total += channels * K_HEAD_KERNEL_SIZE + 1  # head rechannel
    total += 1  # trailing head_scale
    return total


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "tmp_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    for channels, name in [(3, "a2_nano"), (8, "a2_standard")]:
        n = a2_weight_count(channels)
        weights = [rng.uniform(-0.3, 0.3) for _ in range(n)]
        nam = {
            "version": "0.6.0",
            "architecture": "WaveNet",
            "config": build_a2_config(channels),
            "weights": weights,
            "sample_rate": 48000,
            "metadata": {"name": f"A2 benchmark {name}"},
        }
        path = out_dir / f"{name}.nam"
        path.write_text(json.dumps(nam))
        print(f"Wrote {path} ({n} weights)")


if __name__ == "__main__":
    main()
