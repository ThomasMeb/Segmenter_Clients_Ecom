# Raw Data

This directory contains the original **Olist Brazilian E-Commerce Dataset** from Kaggle.

## Source

Dataset: [Olist Brazilian E-Commerce Public Dataset by André Sionek](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Files

| File | Rows | Description |
|------|------|-------------|
| `olist_orders_dataset.csv` | 99,442 | Order information with status and timestamps |
| `olist_order_items_dataset.csv` | 112,651 | Items purchased per order |
| `olist_order_reviews_dataset.csv` | 104,720 | Customer reviews with satisfaction scores (1-5) |
| `olist_customers_dataset.csv` | 99,442 | Customer information including location |
| `olist_order_payments_dataset.csv` | 103,887 | Payment details (type, installments, value) |
| `olist_products_dataset.csv` | 32,952 | Product catalog with categories and dimensions |
| `olist_sellers_dataset.csv` | 3,096 | Seller information and locations |
| `olist_geolocation_dataset.csv` | 1,000,164 | Brazilian ZIP code coordinates |
| `product_category_name_translation.csv` | 71 | Portuguese to English category translations |

## Download Instructions

1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Download the dataset (requires Kaggle account)
3. Extract all CSV files to this directory

## Data Period

**September 2016 - August 2018** (2 years)

## License

This dataset is made available under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Notes

- Files in this directory are gitignored (too large for version control)
- Total dataset size: ~126 MB compressed
- Data is in Brazilian Portuguese with some English translations provided
