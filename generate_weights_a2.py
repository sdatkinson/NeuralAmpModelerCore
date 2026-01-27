#!/usr/bin/env python3
"""
Generate weights for wavenet_a2_max.nam file.
This script handles the full A2 architecture including:
- FiLM (Feature-wise Linear Modulation) modules
- head1x1 modules  
- condition_dsp (nested WaveNet)
- Advanced gating modes (GATED, BLENDED, NONE)
- Complex activation configurations
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List


def count_conv1d_weights(in_channels: int, out_channels: int, kernel_size: int, 
                         has_bias: bool, groups: int = 1) -> int:
    """Count weights for a Conv1D layer."""
    weight_count = kernel_size * (out_channels * in_channels // groups)
    if has_bias:
        weight_count += out_channels
    return weight_count


def count_conv1x1_weights(in_channels: int, out_channels: int, 
                          has_bias: bool, groups: int = 1) -> int:
    """Count weights for a Conv1x1 layer (kernel_size=1)."""
    weight_count = (out_channels * in_channels // groups)
    if has_bias:
        weight_count += out_channels
    return weight_count


def count_film_weights(condition_dim: int, input_dim: int, has_shift: bool, groups: int = 1) -> int:
    """
    Count weights for a FiLM (Feature-wise Linear Modulation) module.
    FiLM uses a Conv1x1: condition_dim -> (2*input_dim if shift else input_dim), with bias.
    
    Args:
        condition_dim: Size of the conditioning input
        input_dim: Size of the input to be modulated
        has_shift: Whether to apply both scale and shift (true) or only scale (false)
        groups: Number of groups for grouped convolution (default: 1)
    """
    out_channels = (2 * input_dim) if has_shift else input_dim
    return count_conv1x1_weights(condition_dim, out_channels, has_bias=True, groups=groups)


def parse_gating_mode(layer_config: Dict[str, Any], layer_index: int = 0) -> str:
    """
    Parse gating mode from layer config (handles both old and new formats).
    
    Args:
        layer_config: Layer configuration dictionary
        layer_index: Index of the layer (for array-valued gating_mode)
    """
    if "gating_mode" in layer_config:
        gating_mode_value = layer_config["gating_mode"]
        if isinstance(gating_mode_value, list):
            # Array of gating modes - use the one at layer_index
            gating_mode_str = gating_mode_value[layer_index]
        else:
            # Single gating mode - use for all layers
            gating_mode_str = gating_mode_value
        
        if isinstance(gating_mode_str, str):
            if gating_mode_str in ["GATED", "BLENDED", "NONE"]:
                return gating_mode_str
            # Handle lowercase versions
            return gating_mode_str.upper()
        return "NONE"
    elif "gated" in layer_config:
        # Backward compatibility
        return "GATED" if layer_config["gated"] else "NONE"
    else:
        return "NONE"


def count_layer_weights(layer_config: Dict[str, Any], condition_size: int, layer_index: int = 0) -> int:
    """
    Count weights for a single layer (one dilation).
    
    A layer consists of:
    1. Conv1D: (channels, bottleneck*(2 if gated/blended else 1), kernel_size, bias=True, groups_input)
    2. Input mixin Conv1x1: (condition_size, bottleneck*(2 if gated/blended else 1), bias=False, groups_input_mixin)
    3. 1x1 Conv1x1: (bottleneck, channels, bias=True, groups_1x1)
    4. Optional head1x1 Conv1x1: (bottleneck, head1x1_out_channels, bias=True, head1x1_groups)
    5. FiLM modules (optional, various configurations)
    
    Args:
        layer_config: Layer configuration dictionary
        condition_size: Size of the conditioning input
        layer_index: Index of the layer within the layer array (for array-valued configs)
    """
    channels = layer_config["channels"]
    bottleneck = layer_config.get("bottleneck", channels)
    kernel_size = layer_config["kernel_size"]
    groups_input = layer_config.get("groups_input", 1)
    groups_input_mixin = layer_config.get("groups_input_mixin", 1)
    groups_1x1 = layer_config.get("groups_1x1", 1)
    
    gating_mode = parse_gating_mode(layer_config, layer_index)
    
    # Output channels are doubled for GATED and BLENDED modes
    conv_out_channels = 2 * bottleneck if gating_mode in ["GATED", "BLENDED"] else bottleneck
    
    weight_count = 0
    
    # 1. Conv1D weights
    weight_count += count_conv1d_weights(
        channels, conv_out_channels, kernel_size,
        has_bias=True, groups=groups_input
    )
    
    # 2. Input mixin Conv1x1 weights
    weight_count += count_conv1x1_weights(
        condition_size, conv_out_channels,
        has_bias=False, groups=groups_input_mixin
    )
    
    # 3. 1x1 Conv1x1 weights
    weight_count += count_conv1x1_weights(
        bottleneck, channels,
        has_bias=True, groups=groups_1x1
    )
    
    # 4. Optional head1x1 weights
    head1x1_config = layer_config.get("head_1x1") or layer_config.get("head1x1")
    if head1x1_config and head1x1_config.get("active", False):
        head1x1_out_channels = head1x1_config.get("out_channels", channels)
        head1x1_groups = head1x1_config.get("groups", 1)
        weight_count += count_conv1x1_weights(
            bottleneck, head1x1_out_channels,
            has_bias=True, groups=head1x1_groups
        )
    
    # 5. FiLM module weights
    # Parse all possible FiLM configurations
    film_configs = [
        ("conv_pre_film", channels),
        ("conv_post_film", conv_out_channels),
        ("input_mixin_pre_film", condition_size),
        ("input_mixin_post_film", conv_out_channels),
        ("activation_pre_film", conv_out_channels),
        ("activation_post_film", bottleneck),
        ("1x1_post_film", channels),
        ("head1x1_post_film", head1x1_config.get("out_channels", channels) if head1x1_config and head1x1_config.get("active") else 0)
    ]
    
    for film_key, input_dim in film_configs:
        if film_key in layer_config and layer_config[film_key]:
            film_params = layer_config[film_key]
            if isinstance(film_params, dict) and film_params.get("active", True):
                has_shift = film_params.get("shift", True)
                groups = film_params.get("groups", 1)
                if input_dim > 0:  # Only count if input_dim is valid
                    weight_count += count_film_weights(condition_size, input_dim, has_shift, groups)
    
    return weight_count


def count_layer_array_weights(layer_config: Dict[str, Any]) -> int:
    """
    Count the total number of weights for a layer array.
    
    Each layer array consists of:
    1. Rechannel Conv1x1: (input_size, channels, bias=False)
    2. Layers (one per dilation)
    3. Head rechannel Conv1x1: (head_output_size, head_size, bias=head_bias)
       where head_output_size = head_1x1.out_channels if head_1x1 active, else bottleneck
    """
    input_size = layer_config["input_size"]
    condition_size = layer_config["condition_size"]
    head_size = layer_config["head_size"]
    channels = layer_config["channels"]
    bottleneck = layer_config.get("bottleneck", channels)
    dilations = layer_config["dilations"]
    head_bias = layer_config.get("head_bias", False)
    
    # Determine head output size: head_1x1.out_channels if active, else bottleneck
    head1x1_config = layer_config.get("head_1x1") or layer_config.get("head1x1")
    if head1x1_config and head1x1_config.get("active", False):
        head_output_size = head1x1_config.get("out_channels", channels)
    else:
        head_output_size = bottleneck
    
    num_layers = len(dilations)
    
    # Validate array-valued configs match number of layers
    if "activation" in layer_config and isinstance(layer_config["activation"], list):
        if len(layer_config["activation"]) != num_layers:
            raise ValueError(f"activation array size ({len(layer_config['activation'])}) must match dilations size ({num_layers})")
    
    if "gating_mode" in layer_config and isinstance(layer_config["gating_mode"], list):
        if len(layer_config["gating_mode"]) != num_layers:
            raise ValueError(f"gating_mode array size ({len(layer_config['gating_mode'])}) must match dilations size ({num_layers})")
    
    if "secondary_activation" in layer_config and isinstance(layer_config["secondary_activation"], list):
        if len(layer_config["secondary_activation"]) != num_layers:
            raise ValueError(f"secondary_activation array size ({len(layer_config['secondary_activation'])}) must match dilations size ({num_layers})")
    
    weight_count = 0
    
    # 1. Rechannel weights
    weight_count += count_conv1x1_weights(input_size, channels, has_bias=False, groups=1)
    
    # 2. For each layer in the array
    for layer_idx in range(num_layers):
        weight_count += count_layer_weights(layer_config, condition_size, layer_idx)
    
    # 3. Head rechannel weights (input is head_output_size, not bottleneck)
    weight_count += count_conv1x1_weights(
        head_output_size, head_size,
        has_bias=head_bias, groups=1
    )
    
    return weight_count


def count_wavenet_weights(config: Dict[str, Any]) -> int:
    """
    Count total weights for a WaveNet model (including optional condition_dsp).
    """
    weight_count = 0
    
    # Count weights for each layer array
    for layer_config in config["layers"]:
        weight_count += count_layer_array_weights(layer_config)
    
    # Add head_scale (1 float)
    weight_count += 1
    
    return weight_count


def generate_weights(weight_count: int, seed: int = None, 
                    weight_range: tuple = (-1.0, 1.0)) -> List[float]:
    """Generate random weights in the specified range."""
    if seed is not None:
        random.seed(seed)
    return [random.uniform(*weight_range) for _ in range(weight_count)]


def process_model(input_path: Path, output_path: Path, seed: int = None) -> None:
    """
    Load a .nam file with empty weights and generate random weights for it.
    """
    # Load the input file
    with open(input_path, 'r') as f:
        model_data = json.load(f)
    
    print(f"Processing: {input_path}")
    print(f"Architecture: {model_data.get('architecture', 'Unknown')}")
    
    # Process condition_dsp if present
    if "config" in model_data and "condition_dsp" in model_data["config"]:
        condition_dsp = model_data["config"]["condition_dsp"]
        if condition_dsp and "config" in condition_dsp:
            print("\nCounting weights for condition_dsp...")
            condition_weights = count_wavenet_weights(condition_dsp["config"])
            print(f"  Condition DSP weights: {condition_weights}")
            
            # Generate weights for condition_dsp
            condition_dsp["weights"] = generate_weights(condition_weights, seed)
            print(f"  Generated {len(condition_dsp['weights'])} weights for condition_dsp")
    
    # Count main model weights
    print("\nCounting weights for main model...")
    main_weights = count_wavenet_weights(model_data["config"])
    print(f"  Main model weights: {main_weights}")
    
    # Generate weights for main model
    model_data["weights"] = generate_weights(main_weights, seed)
    print(f"  Generated {len(model_data['weights'])} weights for main model")
    
    # Print detailed breakdown
    print("\nWeight breakdown:")
    total_weights = 0
    
    # Condition DSP breakdown
    if "config" in model_data and "condition_dsp" in model_data["config"]:
        condition_dsp = model_data["config"]["condition_dsp"]
        if condition_dsp and "config" in condition_dsp:
            print("  Condition DSP:")
            for i, layer in enumerate(condition_dsp["config"]["layers"]):
                layer_weights = count_layer_array_weights(layer)
                print(f"    Layer array {i+1}: {layer_weights} weights")
                total_weights += layer_weights
            total_weights += 1  # head_scale
    
    # Main model breakdown
    print("  Main model:")
    for i, layer in enumerate(model_data["config"]["layers"]):
        layer_weights = count_layer_array_weights(layer)
        print(f"    Layer array {i+1}: {layer_weights} weights")
        total_weights += layer_weights
    total_weights += 1  # head_scale
    
    print(f"\nTotal weights generated: {total_weights}")
    
    # Write output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=4)
    
    print(f"\nOutput written to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate weights for A2 WaveNet models with empty weight arrays"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("example_models/wavenet_a2_max.nam"),
        help="Input .nam file with empty weights (default: example_models/wavenet_a2_max.nam)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("example_models/wavenet_a2_max_withweights.nam"),
        help="Output .nam file (default: example_models/wavenet_a2_max_withweights.nam)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight generation (default: 42)"
    )
    
    args = parser.parse_args()
    
    process_model(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
