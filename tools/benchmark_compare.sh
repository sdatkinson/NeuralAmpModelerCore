#!/bin/bash

# Script to compare performance of current branch against another branch (default: main)
# Usage: ./tools/benchmark_compare.sh [--model MODEL_PATH] [--branch BRANCH_NAME]

set -e  # Exit on error

MODEL_PATH="example_models/wavenet_a1_standard.nam"
BUILD_DIR="build"
BENCHMARK_EXEC="build/tools/benchmodel"
NUM_RUNS=10
COMPARE_BRANCH="main"  # Default branch to compare against
# Report file will be set with timestamp in main()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to extract milliseconds from benchmodel output
extract_ms() {
    local output="$1"
    # Extract the double precision milliseconds value (the second one)
    echo "$output" | grep -E "^[0-9]+\.[0-9]+ms$" | head -1 | sed 's/ms$//'
}

# Function to run benchmark multiple times and collect results
run_benchmark() {
    local branch_name="$1"
    local results_file="$2"
    local project_root="$PWD"  # Save current directory
    
    echo -e "${YELLOW}Running benchmark on branch: ${branch_name}${NC}"
    
    # Clean build directory - remove only untracked files, preserve tracked files like .gitignore
    if [ -d "$BUILD_DIR" ]; then
        # Remove files/directories that aren't tracked by git (process depth-first)
        find "$BUILD_DIR" -mindepth 1 -depth -exec sh -c 'if ! git ls-files --error-unmatch "$1" >/dev/null 2>&1; then rm -rf "$1"; fi' _ {} \;
    fi
    mkdir -p "$BUILD_DIR"
    
    # Configure and build in release mode
    echo "Configuring CMake..."
    cd "$BUILD_DIR" || exit 1
    cmake -DCMAKE_BUILD_TYPE=Release ..
    
    echo "Building benchmodel..."
    cmake --build . --target benchmodel -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd "$project_root" || exit 1
    
    # Verify executable exists
    if [ ! -f "$BENCHMARK_EXEC" ]; then
        echo -e "${RED}Error: benchmodel executable not found at $BENCHMARK_EXEC${NC}"
        exit 1
    fi
    
    # Verify model file exists (use absolute path to be sure)
    local abs_model_path="$project_root/$MODEL_PATH"
    if [ ! -f "$abs_model_path" ]; then
        echo -e "${RED}Error: Model file not found at $abs_model_path${NC}"
        echo "Available model files:"
        find "$project_root/example_models" -name "*.nam" -type f 2>/dev/null || echo "  (none found)"
        exit 1
    fi
    
    # Run benchmark multiple times
    echo "Running benchmark $NUM_RUNS times..."
    > "$results_file"  # Clear results file
    
    for i in $(seq 1 $NUM_RUNS); do
        echo -n "  Run $i/$NUM_RUNS... "
        output=$("$BENCHMARK_EXEC" "$abs_model_path" 2>&1)
        ms=$(extract_ms "$output")
        
        if [ -z "$ms" ]; then
            echo -e "${RED}Failed to extract timing${NC}"
            echo "Output was:"
            echo "$output"
            exit 1
        fi
        
        echo "$ms" >> "$results_file"
        echo "${ms}ms"
    done
    
    echo -e "${GREEN}Completed benchmark for ${branch_name}${NC}"
    echo ""
}

# Function to calculate statistics
calculate_stats() {
    local results_file="$1"
    
    # Calculate mean, min, max, stddev with awk
    local stats=$(awk '
    {
        sum += $1
        sumsq += $1 * $1
        if (NR == 1 || $1 < min) min = $1
        if (NR == 1 || $1 > max) max = $1
    }
    END {
        n = NR
        mean = sum / n
        variance = (sumsq / n) - (mean * mean)
        stddev = sqrt(variance)
        printf "%.3f %.3f %.3f %.3f %d", mean, min, max, stddev, n
    }' "$results_file")
    
    # Calculate median using sort (works with BSD awk)
    local n=$(echo "$stats" | awk '{print $5}')
    local median
    if [ $((n % 2)) -eq 0 ]; then
        # Even number of values: average of middle two
        local mid1=$((n / 2))
        local mid2=$((n / 2 + 1))
        local val1=$(sort -n "$results_file" | sed -n "${mid1}p")
        local val2=$(sort -n "$results_file" | sed -n "${mid2}p")
        median=$(echo "scale=3; ($val1 + $val2) / 2" | bc)
    else
        # Odd number of values: middle value
        local mid=$((n / 2 + 1))
        median=$(sort -n "$results_file" | sed -n "${mid}p")
    fi
    
    # Output: mean median min max stddev
    echo "$stats" | awk -v median="$median" '{printf "%.3f %.3f %.3f %.3f %.3f", $1, median, $2, $3, $4}'
}

# Function to generate report
generate_report() {
    local compare_results="$1"
    local current_results="$2"
    local current_branch="$3"
    local compare_branch="$4"
    local compare_commit="$5"
    local current_commit="$6"
    local report_file="$7"
    
    echo "Generating performance comparison report..."
    
    # Calculate statistics for both branches
    read compare_mean compare_median compare_min compare_max compare_stddev <<< $(calculate_stats "$compare_results")
    read current_mean current_median current_min current_max current_stddev <<< $(calculate_stats "$current_results")
    
    # Calculate percentage difference
    diff_mean=$(echo "scale=2; (($current_mean - $compare_mean) / $compare_mean) * 100" | bc)
    diff_median=$(echo "scale=2; (($current_median - $compare_median) / $compare_median) * 100" | bc)
    
    # Generate report
    {
        echo "=========================================="
        echo "Performance Benchmark Comparison Report"
        echo "=========================================="
        echo ""
        echo "Model: $MODEL_PATH"
        echo "Number of runs per branch: $NUM_RUNS"
        echo "Date: $(date)"
        echo ""
        echo "----------------------------------------"
        echo "Branch: $compare_branch"
        echo "----------------------------------------"
        echo "Commit:   ${compare_commit}"
        echo "Mean:     ${compare_mean} ms"
        echo "Median:   ${compare_median} ms"
        echo "Min:      ${compare_min} ms"
        echo "Max:      ${compare_max} ms"
        echo "Std Dev:  ${compare_stddev} ms"
        echo ""
        echo "----------------------------------------"
        echo "Branch: $current_branch"
        echo "----------------------------------------"
        echo "Commit:   ${current_commit}"
        echo "Mean:     ${current_mean} ms"
        echo "Median:   ${current_median} ms"
        echo "Min:      ${current_min} ms"
        echo "Max:      ${current_max} ms"
        echo "Std Dev:  ${current_stddev} ms"
        echo ""
        echo "----------------------------------------"
        echo "Comparison"
        echo "----------------------------------------"
        if (( $(echo "$diff_mean > 0" | bc -l) )); then
            echo "Mean:     ${current_branch} is ${diff_mean}% SLOWER than ${compare_branch}"
        else
            echo "Mean:     ${current_branch} is ${diff_mean#-}% FASTER than ${compare_branch}"
        fi
        if (( $(echo "$diff_median > 0" | bc -l) )); then
            echo "Median:   ${current_branch} is ${diff_median}% SLOWER than ${compare_branch}"
        else
            echo "Median:   ${current_branch} is ${diff_median#-}% FASTER than ${compare_branch}"
        fi
        echo ""
        echo "Raw Results ($compare_branch):"
        cat "$compare_results" | awk '{printf "  %.3f ms\n", $1}'
        echo ""
        echo "Raw Results ($current_branch):"
        cat "$current_results" | awk '{printf "  %.3f ms\n", $1}'
    } > "$report_file"
    
    echo -e "${GREEN}Report written to: $report_file${NC}"
    echo ""
    cat "$report_file"
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                if [ -z "$2" ]; then
                    echo -e "${RED}Error: --model requires a path argument${NC}"
                    echo "Use --help for usage information"
                    exit 1
                fi
                MODEL_PATH="$2"
                shift 2
                ;;
            --branch)
                if [ -z "$2" ]; then
                    echo -e "${RED}Error: --branch requires a branch name argument${NC}"
                    echo "Use --help for usage information"
                    exit 1
                fi
                COMPARE_BRANCH="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [--model MODEL_PATH] [--branch BRANCH_NAME]"
                echo ""
                echo "Options:"
                echo "  --model MODEL_PATH    Path to the model file to benchmark (default: example_models/wavenet_a1_standard.nam)"
                echo "  --branch BRANCH_NAME  Branch to compare against (default: main)"
                echo "  --help, -h            Show this help message"
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Ensure we're in the project root (parent of tools/)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_ROOT"
    
    # Verify we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}Error: Not in a git repository${NC}"
        exit 1
    fi
    
    # Get current branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    if [ "$current_branch" = "$COMPARE_BRANCH" ]; then
        echo -e "${RED}Error: Already on $COMPARE_BRANCH branch. Please checkout a different branch first.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Current branch: ${current_branch}${NC}"
    echo -e "${YELLOW}Comparing against: ${COMPARE_BRANCH}${NC}"
    echo ""
    
    # Generate timestamped report filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    REPORT_FILE="benchmark_report_${TIMESTAMP}.txt"
    
    # Create temporary files for results
    compare_results=$(mktemp)
    current_results=$(mktemp)
    
    # Variables to store commit hashes
    compare_commit=""
    current_commit=""
    
    # Save untracked model file if it exists (to preserve it across branch switches)
    model_backup=""
    if [ -f "$MODEL_PATH" ] && ! git ls-files --error-unmatch "$MODEL_PATH" > /dev/null 2>&1; then
        echo -e "${YELLOW}Preserving untracked model file: $MODEL_PATH${NC}"
        model_backup=$(mktemp)
        cp "$MODEL_PATH" "$model_backup"
    fi
    
    # Track if we stashed anything
    stashed=false
    
    # Cleanup function
    cleanup() {
        rm -f "$compare_results" "$current_results"
        # Restore original branch if we're not on it
        if [ -n "$current_branch" ] && [ "$(git rev-parse --abbrev-ref HEAD)" != "$current_branch" ]; then
            git checkout "$current_branch" > /dev/null 2>&1 || true
        fi
        # Restore stashed changes if we stashed anything
        if [ "$stashed" = true ]; then
            git stash pop > /dev/null 2>&1 || true
        fi
        # Restore untracked model file if we backed it up
        if [ -n "$model_backup" ] && [ -f "$model_backup" ]; then
            mkdir -p "$(dirname "$MODEL_PATH")"
            cp "$model_backup" "$MODEL_PATH"
            rm -f "$model_backup"
            echo -e "${GREEN}Restored untracked model file: $MODEL_PATH${NC}"
        fi
    }
    trap cleanup EXIT
    
    # Test comparison branch
    echo -e "${YELLOW}=== Testing ${COMPARE_BRANCH} branch ===${NC}"
    # Stash any uncommitted changes (only if there are any)
    if ! git diff-index --quiet HEAD -- 2>/dev/null || ! git diff-index --quiet --cached HEAD -- 2>/dev/null; then
        git stash push -m "benchmark_compare.sh temporary stash" > /dev/null 2>&1
        stashed=true
    fi
    # Restore model file to comparison branch if we backed it up (so it's available for benchmarking)
    if [ -n "$model_backup" ] && [ -f "$model_backup" ]; then
        mkdir -p "$(dirname "$MODEL_PATH")"
        cp "$model_backup" "$MODEL_PATH"
    fi
    # Use --force to allow overwriting untracked files if needed
    git checkout "$COMPARE_BRANCH" --force 2>/dev/null || git checkout "$COMPARE_BRANCH"
    compare_commit=$(git rev-parse HEAD)
    echo "Commit: ${compare_commit}"
    run_benchmark "$COMPARE_BRANCH" "$compare_results"
    
    # Test current branch
    echo -e "${YELLOW}=== Testing ${current_branch} branch ===${NC}"
    git checkout "$current_branch" --force 2>/dev/null || git checkout "$current_branch"
    # Restore model file if we backed it up
    if [ -n "$model_backup" ] && [ -f "$model_backup" ]; then
        mkdir -p "$(dirname "$MODEL_PATH")"
        cp "$model_backup" "$MODEL_PATH"
    fi
    if [ "$stashed" = true ]; then
        git stash pop > /dev/null 2>&1 || true
        stashed=false
    fi
    current_commit=$(git rev-parse HEAD)
    echo "Commit: ${current_commit}"
    run_benchmark "$current_branch" "$current_results"
    
    # Generate report
    generate_report "$compare_results" "$current_results" "$current_branch" "$COMPARE_BRANCH" "$compare_commit" "$current_commit" "$REPORT_FILE"
    
    echo -e "${GREEN}Benchmark comparison complete!${NC}"
}

# Run main function with all command line arguments
main "$@"
