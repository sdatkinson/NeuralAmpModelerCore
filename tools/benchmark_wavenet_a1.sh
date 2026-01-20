#!/bin/bash

# Script to benchmark all wavenet_a1 models and report median results
# Usage: ./tools/benchmark_wavenet_a1.sh

set -e  # Exit on error

BUILD_DIR="build"
BENCHMARK_EXEC="build/tools/benchmodel"
NUM_RUNS=10
EXAMPLE_MODELS_DIR="example_models"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to extract milliseconds from benchmodel output
extract_ms() {
    local output="$1"
    # Extract the double precision milliseconds value (the second one)
    echo "$output" | grep -E "^[0-9]+\.[0-9]+ms$" | head -1 | sed 's/ms$//'
}

# Function to calculate median
calculate_median() {
    local results_file="$1"
    local n=$(wc -l < "$results_file" | tr -d ' ')
    
    if [ "$n" -eq 0 ]; then
        echo "0"
        return
    fi
    
    if [ $((n % 2)) -eq 0 ]; then
        # Even number of values: average of middle two
        local mid1=$((n / 2))
        local mid2=$((n / 2 + 1))
        local val1=$(sort -n "$results_file" | sed -n "${mid1}p")
        local val2=$(sort -n "$results_file" | sed -n "${mid2}p")
        echo "scale=3; ($val1 + $val2) / 2" | bc
    else
        # Odd number of values: middle value
        local mid=$((n / 2 + 1))
        sort -n "$results_file" | sed -n "${mid}p"
    fi
}

# Function to run benchmark for a single model
benchmark_model() {
    local model_path="$1"
    local model_name=$(basename "$model_path")
    local results_file=$(mktemp)
    
    echo -e "${YELLOW}Benchmarking: ${model_name}${NC}"
    
    # Verify model file exists
    if [ ! -f "$model_path" ]; then
        echo "Error: Model file not found at $model_path"
        rm -f "$results_file"
        return 1
    fi
    
    # Verify benchmodel executable exists
    if [ ! -f "$BENCHMARK_EXEC" ]; then
        echo "Error: benchmodel executable not found at $BENCHMARK_EXEC"
        echo "Please build the project first: cd $BUILD_DIR && cmake .. && make benchmodel"
        rm -f "$results_file"
        return 1
    fi
    
    # Run benchmark multiple times
    > "$results_file"  # Clear results file
    
    for i in $(seq 1 $NUM_RUNS); do
        echo -n "  Run $i/$NUM_RUNS... "
        output=$("$BENCHMARK_EXEC" "$model_path" 2>&1)
        ms=$(extract_ms "$output")
        
        if [ -z "$ms" ]; then
            echo "Failed to extract timing"
            echo "Output was:"
            echo "$output"
            rm -f "$results_file"
            return 1
        fi
        
        echo "$ms" >> "$results_file"
        echo "${ms}ms"
    done
    
    # Calculate median
    median=$(calculate_median "$results_file")
    
    echo -e "${GREEN}  Median: ${median}ms${NC}"
    echo ""
    
    # Store result for reporting
    echo "${model_name},${median}" >> "$RESULTS_SUMMARY"
    
    rm -f "$results_file"
}

# Main execution
main() {
    # Ensure we're in the project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_ROOT"
    
    echo "=========================================="
    echo "Wavenet A1 Models Benchmark"
    echo "=========================================="
    echo "Running each model $NUM_RUNS times"
    echo "Reporting median results"
    echo ""
    
    # Create temporary file for results summary
    RESULTS_SUMMARY=$(mktemp)
    > "$RESULTS_SUMMARY"  # Clear file
    
    # Cleanup function
    cleanup() {
        rm -f "$RESULTS_SUMMARY"
    }
    trap cleanup EXIT
    
    # Find all wavenet_a1 models
    models=$(find "$EXAMPLE_MODELS_DIR" -name "wavenet_a1*.nam" -type f | sort)
    
    if [ -z "$models" ]; then
        echo "Error: No wavenet_a1*.nam files found in $EXAMPLE_MODELS_DIR"
        rm -f "$RESULTS_SUMMARY"
        exit 1
    fi
    
    # Benchmark each model
    while read -r model; do
        benchmark_model "$model"
    done <<< "$models"
    
    # Print summary
    echo "=========================================="
    echo "Summary (Median Results)"
    echo "=========================================="
    echo ""
    printf "%-30s %15s\n" "Model" "Median (ms)"
    echo "----------------------------------------------"
    
    sort "$RESULTS_SUMMARY" | while IFS=',' read -r model_name median; do
        printf "%-30s %15s\n" "$model_name" "$median"
    done
    
    echo ""
    rm -f "$RESULTS_SUMMARY"
    
    echo -e "${GREEN}Benchmark complete!${NC}"
}

# Run main function
main "$@"
