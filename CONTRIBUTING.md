# Contributing to Olist Customer Segmentation

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Make (optional, but recommended)

### Setting Up the Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/ThomasMeb/olist-customer-segmentation.git
   cd olist-customer-segmentation
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   ```

3. **Install development dependencies**

   ```bash
   pip install -e ".[dev]"
   # or
   make install-dev
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

5. **Verify your setup**

   ```bash
   make check  # Runs lint + tests
   ```

## Development Workflow

### Branch Naming Convention

Use descriptive branch names with prefixes:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat/` | New features | `feat/export-csv` |
| `fix/` | Bug fixes | `fix/clustering-error` |
| `docs/` | Documentation | `docs/api-reference` |
| `refactor/` | Code refactoring | `refactor/rfm-calculator` |
| `test/` | Test additions | `test/edge-cases` |
| `chore/` | Maintenance tasks | `chore/update-deps` |

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
feat(cli): add export command for predictions
fix(clustering): handle edge case with single customer
docs(readme): update installation instructions
test(rfm): add tests for edge cases
```

### Workflow Steps

1. **Create a feature branch**

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes**

   - Write code
   - Add/update tests
   - Update documentation if needed

3. **Run checks locally**

   ```bash
   make check  # Runs lint + tests
   ```

4. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

5. **Push and create PR**

   ```bash
   git push origin feat/your-feature-name
   ```

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Documentation is updated (if applicable)
- [ ] CHANGELOG.md is updated (for significant changes)

### PR Template

When creating a PR, include:

```markdown
## Summary
Brief description of the changes.

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
```

### Review Process

1. A maintainer will review your PR
2. Address any requested changes
3. Once approved, the PR will be merged

## Coding Standards

### Python Style

We use **Ruff** for linting and formatting:

```bash
# Format code
make format

# Check linting
make lint
```

### Key Guidelines

- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings
- **Line length**: Maximum 88 characters
- **Imports**: Sorted automatically by ruff

### Example Code Style

```python
def calculate_rfm(
    df: pd.DataFrame,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Calculate RFM features from transaction data.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction data with customer_id, date, and amount columns.
    reference_date : datetime, optional
        Reference date for recency calculation.

    Returns
    -------
    pd.DataFrame
        DataFrame with recency, frequency, and monetary columns.

    Raises
    ------
    ValueError
        If required columns are missing.

    Examples
    --------
    >>> rfm = calculate_rfm(transactions)
    >>> rfm.head()
    """
    ...
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_rfm.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test edge cases and error conditions

### Test Structure

```python
class TestRFMCalculator:
    """Tests for RFMCalculator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return pd.DataFrame(...)

    def test_basic_calculation(self, sample_data):
        """Test basic RFM calculation."""
        calculator = RFMCalculator()
        result = calculator.fit_transform(sample_data)
        assert "recency" in result.columns

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError):
            calculator = RFMCalculator()
            calculator.fit_transform(pd.DataFrame())
```

## Documentation

### Updating Documentation

- **README.md**: Project overview and quick start
- **CHANGELOG.md**: Version history (Keep a Changelog format)
- **Docstrings**: Google-style for all public functions
- **docs/**: Detailed documentation (Sphinx)

### Building Documentation

```bash
# Build HTML docs
cd docs
make html

# View locally
open _build/html/index.html
```

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

Thank you for contributing!
