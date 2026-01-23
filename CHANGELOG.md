# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CLI with Click for train, predict, evaluate, serve commands
- Makefile with standard targets
- Automation scripts (setup.sh, train.sh, download_data.sh)
- Comprehensive test suite (152 tests, 92% coverage)
- Pre-commit hooks for code quality
- CI/CD workflows for GitHub Actions
- **Phase 5: Documentation**
  - CODE_OF_CONDUCT.md (Contributor Covenant)
  - Sphinx documentation setup with Furo theme
  - API reference documentation with autodoc
  - User guides: Quick Start, Installation, CLI, API, Dashboard
  - FAQ with common troubleshooting
  - `make docs` and `make docs-live` commands
- **Phase 6: MLOps & Monitoring**
  - Model Registry with versioning and metadata
  - Drift Detection (ARI-based model drift, KS-based data drift)
  - DriftDetector class with fit/detect workflow
  - Maintenance notebook (notebooks/05_maintenance.ipynb)
  - GitHub Actions workflow for scheduled maintenance
  - 23 new tests for monitoring module

### Changed
- Migrated from black to ruff-format
- Refactored imports to use absolute paths
- Updated Streamlit API (use_container_width → width)
- Updated pyproject.toml with docs dependencies

### Fixed
- Palette bug in RFM boxplots visualization
- Test loader missing columns test

## [1.0.0] - 2024-01-23

### Added
- Initial release
- RFM (Recency, Frequency, Monetary) feature engineering
- KMeans clustering with 4 customer segments
- Interactive Streamlit dashboard with 4 pages:
  - Overview: Key metrics and segment distribution
  - Segments: Detailed segment analysis
  - Explorer: 3D visualization
  - About: Project documentation
- Model persistence (save/load)
- Clustering evaluation metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Data preprocessing pipeline
- Visualization functions (elbow curve, silhouette plot, radar chart, etc.)

### Customer Segments
- **Clients Récents** (54%): Recent purchase, low frequency
- **Clients Fidèles** (3%): Regular purchases
- **Clients Dormants** (40%): Inactive customers
- **Clients VIP** (3%): High value customers

### Technical
- Python 3.10+ support
- Modular architecture (src/, app/, tests/)
- pytest test suite
- Type hints throughout codebase
- Configuration centralized in src/config.py

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2024-01-23 | Initial release with dashboard |

[Unreleased]: https://github.com/ThomasMeb/olist-customer-segmentation/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ThomasMeb/olist-customer-segmentation/releases/tag/v1.0.0
