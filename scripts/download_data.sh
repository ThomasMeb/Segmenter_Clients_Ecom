#!/bin/bash
# =============================================================================
# download_data.sh - Download Olist dataset from Kaggle
# =============================================================================
# Usage: ./scripts/download_data.sh
#
# Prerequisites:
#   - Kaggle API credentials (~/.kaggle/kaggle.json)
#   - kaggle package installed: pip install kaggle
#
# Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "  Olist Data Download"
echo "=============================================="

# Check for kaggle credentials
if [[ ! -f ~/.kaggle/kaggle.json ]]; then
    echo -e "${RED}Error: Kaggle credentials not found${NC}"
    echo ""
    echo "To set up Kaggle API:"
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Create New API Token"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Alternatively, download manually from:"
    echo "  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
    exit 1
fi

# Check for kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo -e "${YELLOW}Installing kaggle package...${NC}"
    pip install kaggle -q
fi

# Create data directory
mkdir -p data/raw

echo -e "\n[1/3] Downloading Olist dataset..."
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

echo -e "\n[2/3] Listing downloaded files..."
ls -la data/raw/

echo -e "\n[3/3] Creating merged dataset..."

# If Python merge script is needed, run it
if [[ -f "scripts/merge_data.py" ]]; then
    python scripts/merge_data.py
else
    # Create a simple merge using Python inline
    python << 'EOF'
import pandas as pd
from pathlib import Path

raw_dir = Path("data/raw")

# Check if individual files exist
orders_file = raw_dir / "olist_orders_dataset.csv"
items_file = raw_dir / "olist_order_items_dataset.csv"
customers_file = raw_dir / "olist_customers_dataset.csv"

if all(f.exists() for f in [orders_file, items_file, customers_file]):
    print("Merging datasets...")

    orders = pd.read_csv(orders_file)
    items = pd.read_csv(items_file)
    customers = pd.read_csv(customers_file)

    # Merge
    df = items.merge(orders, on="order_id")
    df = df.merge(customers, on="customer_id")

    # Select and rename columns
    df = df[[
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "price"
    ]]

    # Save
    output_path = raw_dir / "data.csv"
    df.to_csv(output_path, index=False)
    print(f"Created: {output_path}")
    print(f"Total rows: {len(df):,}")
else:
    print("Required files not found. Please check the download.")
EOF
fi

# Copy olist files to data/ directory for notebook compatibility
echo -e "\n[4/4] Copying files for notebook compatibility..."
cp data/raw/olist_*.csv data/
cp data/raw/product_category_name_translation.csv data/

echo -e "\n${GREEN}=============================================="
echo "  Download complete!"
echo "==============================================${NC}"
echo ""
echo "Files available in both data/ and data/raw/"
echo ""
echo "Next steps:"
echo "  1. Train the model: make train"
echo "  2. Start dashboard: make serve"
echo ""
