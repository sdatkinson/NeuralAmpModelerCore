#!/usr/bin/env python3
"""
Utility script to create WaveNet .nam files with given configurations and random weights.
The script ensures the correct number of weights based on the network architecture.
"""

import json
import random
import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any


class ExampleConfig(Enum):
    """Example WaveNet configuration presets."""
    SIMPLE = "simple"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


def count_conv1d_weights(in_channels: int, out_channels: int, kernel_size: int, 
                         has_bias: bool, groups: int = 1) -> int:
    """Count weights for a Conv1D layer."""
    # For grouped convolutions: kernel_size * (out_channels * in_channels / groups) + bias
    weight_count = kernel_size * (out_channels * in_channels // groups)
    if has_bias:
        weight_count += out_channels
    return weight_count


def count_conv1x1_weights(in_channels: int, out_channels: int, 
                          has_bias: bool, groups: int = 1) -> int:
    """Count weights for a Conv1x1 layer."""
    # For grouped convolutions: (out_channels * in_channels / groups) + bias
    weight_count = (out_channels * in_channels // groups)
    if has_bias:
        weight_count += out_channels
    return weight_count


def count_layer_array_weights(layer_config: Dict[str, Any]) -> int:
    """
    Count the total number of weights for a layer array.
    
    Each layer array consists of:
    1. Rechannel Conv1x1: (input_size, channels, bias=False)
    2. For each layer (for each dilation):
       - Conv1D: (channels, bottleneck*(2 if gated else 1), kernel_size, bias=True, groups_input)
       - Input mixin Conv1x1: (condition_size, bottleneck*(2 if gated else 1), bias=False)
       - 1x1 Conv1x1: (bottleneck, channels, bias=True, groups_1x1)
    3. Head rechannel Conv1x1: (bottleneck, head_size, bias=head_bias)
    """
    input_size = layer_config["input_size"]
    condition_size = layer_config["condition_size"]
    head_size = layer_config["head_size"]
    channels = layer_config["channels"]
    bottleneck = layer_config.get("bottleneck", channels)
    kernel_size = layer_config["kernel_size"]
    dilations = layer_config["dilations"]
    gated = layer_config.get("gated", False)
    head_bias = layer_config.get("head_bias", False)
    groups_input = layer_config.get("groups", 1)
    groups_1x1 = layer_config.get("groups_1x1", 1)
    
    num_layers = len(dilations)
    
    # 1. Rechannel weights
    weight_count = count_conv1x1_weights(input_size, channels, has_bias=False, groups=1)
    
    # 2. For each layer in the array
    for _ in range(num_layers):
        # Conv1D: output channels are 2*bottleneck if gated, bottleneck otherwise
        conv_out_channels = 2 * bottleneck if gated else bottleneck
        weight_count += count_conv1d_weights(
            channels, conv_out_channels, kernel_size, 
            has_bias=True, groups=groups_input
        )
        
        # Input mixin Conv1x1
        weight_count += count_conv1x1_weights(
            condition_size, conv_out_channels, 
            has_bias=False, groups=1
        )
        
        # 1x1 Conv1x1
        weight_count += count_conv1x1_weights(
            bottleneck, channels, 
            has_bias=True, groups=groups_1x1
        )
    
    # 3. Head rechannel weights
    weight_count += count_conv1x1_weights(
        bottleneck, head_size, 
        has_bias=head_bias, groups=1
    )
    
    return weight_count


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the WaveNet configuration.
    
    Validates:
    1. All layers have the same condition_size
    2. For each layer i (except the last), layers[i]["channels"] == layers[i+1]["input_size"]
    3. For each layer i (except the last), layers[i]["head_size"] == layers[i+1]["channels"]
    
    Raises:
        ValueError: If validation fails
    """
    layers = config["layers"]
    
    if not layers:
        raise ValueError("Config must have at least one layer")
    
    # 1. Check that all condition_size values are the same
    condition_sizes = [layer["condition_size"] for layer in layers]
    if len(set(condition_sizes)) > 1:
        raise ValueError(
            f"All layers must have the same condition_size. "
            f"Found: {dict(enumerate(condition_sizes))}"
        )
    
    # 2. Check that channels[i] == input_size[i+1] for each layer (except last)
    # 3. Check that head_size[i] == channels[i+1] for each layer (except last)
    for i in range(len(layers) - 1):
        current_layer = layers[i]
        next_layer = layers[i + 1]
        
        if current_layer["channels"] != next_layer["input_size"]:
            raise ValueError(
                f"Layer {i} channels ({current_layer['channels']}) must match "
                f"layer {i+1} input_size ({next_layer['input_size']})"
            )
        
        if current_layer["head_size"] != next_layer["channels"]:
            raise ValueError(
                f"Layer {i} head_size ({current_layer['head_size']}) must match "
                f"layer {i+1} channels ({next_layer['channels']})"
            )


def generate_weights(weight_count: int, seed: int = None) -> List[float]:
    """Generate random weights."""
    if seed is not None:
        random.seed(seed)
    # Generate weights in a reasonable range, similar to typical neural network initialization
    return [random.uniform(-1.0, 1.0) for _ in range(weight_count)]


def create_wavenet_nam(config: Dict[str, Any], output_path: Path, 
                       seed: int = None, sample_rate: int = 48000,
                       version: str = "0.6.0") -> None:
    """
    Create a WaveNet .nam file with the given configuration and random weights.
    
    Args:
        config: WaveNet configuration dictionary with 'layers' and optionally 'head_scale'
        output_path: Path to output .nam file
        seed: Random seed for weight generation (None for random)
        sample_rate: Sample rate for the model
        version: Version string for the .nam file
    
    Raises:
        ValueError: If configuration validation fails
    """
    # Validate the configuration
    validate_config(config)
    
    layers = config["layers"]
    head_scale = config.get("head_scale", 0.02)
    
    # Calculate total weight count
    total_weights = 0
    
    # Count weights for each layer array
    for layer_config in layers:
        total_weights += count_layer_array_weights(layer_config)
    
    # Add head_scale (1 float)
    total_weights += 1
    
    # Generate weights
    weights = generate_weights(total_weights, seed)
    
    # Create metadata
    now = datetime.now()
    metadata = {
        "date": {
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second
        },
        "loudness": -20.0,
        "gain": 0.2,
        "name": "Generated WaveNet Model",
        "modeled_by": "create_wavenet.py"
    }
    
    # Create the .nam file structure
    nam_file = {
        "version": version,
        "metadata": metadata,
        "architecture": "WaveNet",
        "config": {
            "layers": layers,
            "head": None,
            "head_scale": head_scale
        },
        "weights": weights,
        "sample_rate": sample_rate
    }
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(nam_file, f, indent=4)
    
    print(f"Created WaveNet .nam file: {output_path}")
    print(f"Total weights: {total_weights}")
    print(f"Layers: {len(layers)}")
    for i, layer in enumerate(layers):
        layer_weights = count_layer_array_weights(layer)
        print(f"  Layer array {i+1}: {layer_weights} weights")


def main():
    parser = argparse.ArgumentParser(
        description="Create WaveNet .nam files with given configurations and random weights"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output .nam file path"
    )
    parser.add_argument(
        "--config",
        help="JSON file with WaveNet configuration (if not provided, uses example config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for weight generation (default: random)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate (default: 48000)"
    )
    parser.add_argument(
        "--version",
        default="0.5.4",
        help="Version string (default: 0.5.4)"
    )
    parser.add_argument(
        "--output-channels",
        type=int,
        default=1,
        help="Output channels for condition_dsp (default: 1)"
    )
    parser.add_argument(
        "--condition-dim",
        type=int,
        default=None,
        help="Condition dimension (overrides condition_size in all layers, default: use layer config)"
    )
    
    # Example layer configurations
    def example_type(value: str) -> ExampleConfig:
        """Convert string to ExampleConfig enum."""
        try:
            return ExampleConfig[value.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError(
                f"Invalid example config: {value}. Must be one of: {', '.join([e.value for e in ExampleConfig])}"
            )
    
    parser.add_argument(
        "--example",
        type=example_type,
        help="Use an example configuration (simple, small, medium, or large)"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.example:
        # Create example configurations
        if args.example == ExampleConfig.SIMPLE:
            config = {
                "layers": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "head_size": 2,
                        "channels": 3,
                        "kernel_size": 3,
                        "dilations": [1, 2],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False
                    },
                    {
                        "input_size": 3,
                        "condition_size": 1,
                        "head_size": 1,
                        "channels": 2,
                        "kernel_size": 3,
                        "dilations": [8],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True
                    }
                ],
                "head_scale": 0.02
            }
        elif args.example == ExampleConfig.SMALL:
            config = {
                "layers": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "head_size": 8,
                        "channels": 16,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False
                    },
                    {
                        "input_size": 16,
                        "condition_size": 1,
                        "head_size": 1,
                        "channels": 8,
                        "kernel_size": 3,
                        "dilations": [64, 128, 256],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True
                    }
                ],
                "head_scale": 0.02
            }
        elif args.example == ExampleConfig.MEDIUM:
            config = {
                "layers": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "head_size": 16,
                        "channels": 32,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False
                    },
                    {
                        "input_size": 32,
                        "condition_size": 1,
                        "head_size": 1,
                        "channels": 16,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True
                    }
                ],
                "head_scale": 0.02
            }
        else:  # ExampleConfig.LARGE
            config = {
                "layers": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "head_size": 32,
                        "channels": 64,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": False
                    },
                    {
                        "input_size": 64,
                        "condition_size": 1,
                        "head_size": 1,
                        "channels": 32,
                        "kernel_size": 3,
                        "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                        "activation": "Tanh",
                        "gated": False,
                        "head_bias": True
                    }
                ],
                "head_scale": 0.02
            }
    else:
        # Default to simple config
        config = {
            "layers": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 2,
                    "channels": 3,
                    "kernel_size": 3,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False
                },
                {
                    "input_size": 3,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 3,
                    "dilations": [8],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True
                }
            ],
            "head_scale": 0.02
        }
    
    # Override condition_size in all layers if condition_dim is specified
    if args.condition_dim is not None:
        for layer in config["layers"]:
            layer["condition_size"] = args.condition_dim
    
    # Override head_size in last layer array if output_channels is specified
    if args.output_channels is not None:
        config["layers"][-1]["head_size"] = args.output_channels
    
    create_wavenet_nam(
        config,
        args.output,
        seed=args.seed,
        sample_rate=args.sample_rate,
        version=args.version
    )


if __name__ == "__main__":
    main()
