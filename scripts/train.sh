#!/bin/bash
# =============================================================================
# train.sh - Train the customer segmentation model
# =============================================================================
# Usage: ./scripts/train.sh [OPTIONS]
#
# Options:
#   --input FILE     Input CSV file (default: data/raw/data.csv)
#   --output DIR     Output directory (default: models/)
#   --n-clusters N   Number of clusters (default: 4)
#   --verbose        Enable verbose output
# =============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment if exists
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "  Olist Customer Segmentation - Training"
echo "=============================================="

# Default values
INPUT_FILE="data/raw/data.csv"
OUTPUT_DIR="models/"
N_CLUSTERS=4
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-clusters)
            N_CLUSTERS="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input FILE     Input CSV file (default: data/raw/data.csv)"
            echo "  --output DIR     Output directory (default: models/)"
            echo "  --n-clusters N   Number of clusters (default: 4)"
            echo "  --verbose        Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo ""
    echo "Please ensure your data file exists or run:"
    echo "  ./scripts/download_data.sh"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Clusters: $N_CLUSTERS"
echo ""

# Run training
python -m src.cli train \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --n-clusters "$N_CLUSTERS" \
    $VERBOSE

echo ""
echo "Training complete!"
echo "To start the dashboard: make serve"
