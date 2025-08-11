# Contributing to WaveAnim

Thank you for your interest in contributing to WaveAnim! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/waveAnim.git
   cd waveAnim
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
black sine_wave_animator/
isort sine_wave_animator/
flake8 sine_wave_animator/
mypy sine_wave_animator/
```

### Testing

Run tests with pytest:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sine_wave_animator

# Run specific test file
pytest tests/test_sine_wave_animator.py

# Run with verbose output
pytest -v
```

### Type Hints

All code must include comprehensive type hints. Use:
- `typing` module for generic types
- `numpy.typing` for NumPy arrays
- `Optional[]` for nullable types
- `Union[]` for multiple possible types

Example:
```python
def generate_wave(self, wave_idx: int, x_points: Optional[npt.NDArray[np.float64]] = None) -> npt.NDArray[np.float64]:
    # Implementation here
    pass
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-envelope-algorithm`
- `fix/animation-speed-bug`
- `docs/update-readme`

### Commit Messages

Follow conventional commit format:
- `feat: add new envelope calculation method`
- `fix: resolve animation timing issue`
- `docs: update API documentation`
- `test: add tests for edge cases`
- `refactor: simplify wave generation logic`

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run the test suite**:
   ```bash
   pytest
   black --check sine_wave_animator/
   isort --check-only sine_wave_animator/
   flake8 sine_wave_animator/
   mypy sine_wave_animator/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: your descriptive commit message"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Pull Request Guidelines

- **Title**: Use a descriptive title that summarizes the change
- **Description**: Explain what changes were made and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update docstrings and README if needed
- **Breaking Changes**: Clearly mark any breaking changes

## Code Guidelines

### General Principles

1. **Readability**: Code should be self-documenting
2. **Type Safety**: Use comprehensive type hints
3. **Testing**: All new code should include tests
4. **Documentation**: Public APIs must be documented
5. **Performance**: Consider performance implications

### Documentation

- Use Google-style docstrings
- Include parameter types and descriptions
- Provide usage examples for complex functions
- Update README.md for user-facing changes

Example docstring:
```python
def calculate_pointwise_maximum(self) -> npt.NDArray[np.float64]:
    """
    Calculate the pointwise maximum across all sine waves.
    
    This method evaluates all sine waves at each x-coordinate and returns
    the maximum value at each point, creating an envelope that follows
    the peaks of the highest wave at each location.
    
    Returns:
        np.ndarray: Array of maximum values at each x-coordinate
        
    Example:
        >>> animator = SineWaveAnimator(num_waves=5)
        >>> envelope = animator.calculate_pointwise_maximum()
        >>> len(envelope) == len(animator.x)
        True
    """
```

### Testing Guidelines

- Write tests for all public methods
- Include edge cases and error conditions
- Use descriptive test names
- Group related tests in classes
- Mock external dependencies when needed

Example test:
```python
def test_generate_wave_invalid_index(self):
    """Test that invalid wave indices raise errors."""
    animator = SineWaveAnimator(num_waves=3)
    
    with pytest.raises(IndexError):
        animator.generate_wave(-1)
    
    with pytest.raises(IndexError):
        animator.generate_wave(3)
```

## Reporting Issues

### Bug Reports

Include:
- Python version
- Package versions (`pip list`)
- Minimal reproducible example
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

Include:
- Use case description
- Proposed API or interface
- Examples of how it would be used
- Alternatives considered

## Getting Help

- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the maintainers directly for sensitive issues

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- GitHub repository insights

Thank you for contributing to WaveAnim!
