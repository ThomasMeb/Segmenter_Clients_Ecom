#!/bin/bash
# =============================================================================
# setup.sh - Complete project setup
# =============================================================================
# Usage: ./scripts/setup.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Olist Customer Segmentation - Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo -e "\nProject root: $PROJECT_ROOT"

# Check Python version
echo -e "\n[1/5] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    echo -e "${RED}Error: Python 3.10+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION OK${NC}"

# Create virtual environment if needed
echo -e "\n[2/5] Setting up virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo -e "${GREEN}Created .venv${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate venv
source .venv/bin/activate
echo -e "${GREEN}Activated .venv${NC}"

# Install dependencies
echo -e "\n[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q
echo -e "${GREEN}Dependencies installed${NC}"

# Install pre-commit hooks
echo -e "\n[4/5] Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo -e "${GREEN}Pre-commit hooks installed${NC}"
else
    echo -e "${YELLOW}pre-commit not found, skipping hooks${NC}"
fi

# Create data directories
echo -e "\n[5/5] Creating directories..."
mkdir -p data/raw data/processed models
echo -e "${GREEN}Directories created${NC}"

# Summary
echo -e "\n=============================================="
echo -e "${GREEN}  Setup completed successfully!${NC}"
echo "=============================================="
echo -e "\nNext steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Download data: ./scripts/download_data.sh"
echo "  3. Train model: make train"
echo "  4. Start dashboard: make serve"
echo ""
