# Contributing Guide

Thank you for your interest in contributing to the Customer Segmentation project!

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Development Setup](#development-setup)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to:

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other contributors

---

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

**🐛 Bug Fixes**
- Report bugs via GitHub Issues
- Submit pull requests with fixes

**✨ New Features**
- Propose new clustering algorithms
- Add visualization improvements
- Enhance dashboard functionality

**📚 Documentation**
- Improve existing documentation
- Add usage examples
- Translate to other languages

**🧪 Testing**
- Write unit tests for untested code
- Improve test coverage
- Add integration tests

**🎨 Code Quality**
- Refactor for better readability
- Optimize performance
- Improve error handling

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Segmenter_Clients_Ecom.git
cd Segmenter_Clients_Ecom

# Add upstream remote
git remote add upstream https://github.com/ThomasMeb/Segmenter_Clients_Ecom.git
```

### 2. Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n customer-segmentation python=3.9
conda activate customer-segmentation
```

### 3. Install Development Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install black flake8 pytest pytest-cov
```

### 4. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

---

## Code Style

### Python Style Guide

We follow **PEP 8** with some additional conventions:

**Imports**
```python
# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Local imports
from src.data.data_loader import OlistDataLoader
from src.models.clustering import CustomerSegmenter
```

**Naming Conventions**
- **Classes:** PascalCase (`CustomerSegmenter`, `RFMCalculator`)
- **Functions/Methods:** snake_case (`calculate_rfm`, `load_data`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_CLUSTERS`, `DEFAULT_K`)
- **Private methods:** Leading underscore (`_initialize_model`)

**Docstrings**

Use Google-style docstrings:

```python
def calculate_rfm(data: pd.DataFrame, customer_id_col: str = 'customer_id') -> pd.DataFrame:
    """
    Calculate RFM metrics from transaction data.

    Parameters:
        data (pd.DataFrame): Transaction-level data with customer IDs and dates.
        customer_id_col (str): Column name for customer identifiers.

    Returns:
        pd.DataFrame: RFM metrics per customer with columns [Recency, Frequency, Monetary].

    Raises:
        ValueError: If required columns are missing.

    Example:
        >>> rfm = calculate_rfm(transactions, customer_id_col='customer_unique_id')
        >>> print(rfm.head())
    """
    pass
```

**Type Hints**

Use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple

def find_optimal_k(
    data: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42
) -> Tuple[List[int], List[float]]:
    pass
```

### Code Formatting

Use **Black** for automatic formatting:

```bash
# Format a single file
black src/models/clustering.py

# Format entire project
black .

# Check without modifying
black --check .
```

### Linting

Use **flake8** to check for style issues:

```bash
# Check a file
flake8 src/models/clustering.py

# Check entire project
flake8 src/

# Configuration in .flake8 (create if needed)
[flake8]
max-line-length = 100
ignore = E203, W503
exclude = venv, .git, __pycache__
```

---

## Testing

### Writing Tests

Place tests in a `tests/` directory (create if needed):

```
tests/
├── __init__.py
├── test_data_loader.py
├── test_clustering.py
├── test_rfm_engineering.py
└── test_metrics.py
```

**Example Test:**

```python
# tests/test_clustering.py
import pytest
import pandas as pd
from src.models.clustering import CustomerSegmenter

def test_kmeans_initialization():
    """Test K-Means model initialization."""
    segmenter = CustomerSegmenter(algorithm='kmeans', n_clusters=4)
    assert segmenter.algorithm == 'kmeans'
    assert segmenter.model.n_clusters == 4

def test_fit_predict():
    """Test fit_predict on sample data."""
    data = pd.DataFrame({
        'Recency': [10, 100, 200, 300],
        'Frequency': [5, 3, 1, 1],
        'Monetary': [500, 300, 100, 50]
    })

    segmenter = CustomerSegmenter(algorithm='kmeans', n_clusters=2, random_state=42)
    labels = segmenter.fit_predict(data)

    assert len(labels) == 4
    assert len(set(labels)) <= 2  # At most 2 unique clusters
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_clustering.py

# Run specific test
pytest tests/test_clustering.py::test_kmeans_initialization

# Verbose output
pytest -v
```

### Test Coverage

Aim for >80% test coverage on new code:

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# Open htmlcov/index.html in browser to view
```

---

## Submitting Changes

### Pull Request Process

1. **Update Your Branch**

```bash
git fetch upstream
git rebase upstream/main
```

2. **Commit Your Changes**

Use clear, descriptive commit messages:

```bash
git add src/models/clustering.py
git commit -m "Add HDBSCAN clustering algorithm

- Implement HDBSCAN as new clustering option
- Add min_cluster_size and min_samples parameters
- Update tests and documentation
- Fixes #42"
```

**Commit Message Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, style, refactor, test, chore

3. **Push to Your Fork**

```bash
git push origin feature/your-feature-name
```

4. **Create Pull Request**

- Go to GitHub and click "New Pull Request"
- Fill in the PR template:
  - Description of changes
  - Related issue number (#123)
  - Testing performed
  - Screenshots (if applicable)

5. **Code Review**

- Address reviewer comments
- Make requested changes
- Push updates to the same branch

6. **Merge**

Once approved, a maintainer will merge your PR.

---

## Reporting Issues

### Bug Reports

Use the GitHub issue tracker with this template:

**Title:** Clear, concise description

**Body:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Load data with '...'
2. Run command '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.9.7]
- Package versions: [paste output of `pip freeze`]

**Additional context**
Add any other context, screenshots, or logs.
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
Describe the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Mockups, examples, or related issues.
```

---

## Project Structure

When adding new features, follow this structure:

```
src/
├── data/           # Data loading and preprocessing
├── features/       # Feature engineering (RFM, etc.)
├── models/         # Clustering algorithms
├── evaluation/     # Metrics and evaluation
└── visualization/  # Plotting functions

scripts/            # Executable scripts (train.py, predict.py)
tests/              # Unit and integration tests
docs/               # Additional documentation
notebooks/          # Jupyter notebooks for exploration
```

---

## Best Practices

### Code Quality

- **Single Responsibility:** Each function should do one thing well
- **DRY Principle:** Don't Repeat Yourself
- **Error Handling:** Use try/except with specific exceptions
- **Logging:** Use `logging` module instead of print statements
- **Configuration:** Use `config.yaml` for configurable values

### Documentation

- **Docstrings:** Every public function/class needs a docstring
- **README:** Update if adding new features
- **Examples:** Provide usage examples in docstrings
- **Changelog:** Update CHANGELOG.md (if exists)

### Git

- **Atomic Commits:** One logical change per commit
- **Meaningful Messages:** Explain why, not just what
- **Small PRs:** Easier to review (< 500 lines ideal)
- **Branch Naming:** `feature/add-dbscan`, `fix/issue-123`

---

## Questions?

- **Documentation:** Check `docs/` folder
- **Examples:** Review `notebooks/` for usage examples
- **Discussions:** Open a GitHub Discussion
- **Maintainer:** Contact Thomas Mebarki

Thank you for contributing! 🎉
