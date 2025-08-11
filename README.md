# WaveAnim

An animated sine wave generator with envelope tracing, built with Python and matplotlib. This tool creates real-time visualizations of multiple sine waves with different amplitudes, frequencies, and phases, culminating in an animated upper envelope that traces the pointwise maximum across all waves.

## Features

- **Real-time Animation**: Progressive drawing of sine waves from left to right
- **Customizable Parameters**: Control amplitude ranges, frequency ranges, and animation speed
- **Envelope Tracing**: Animated upper envelope showing pointwise maximum across all waves
- **Visual Effects**: Thick lines during drawing, thin lines when complete
- **Dark Theme**: Professional dark background with colorful waves
- **Command Line Interface**: Full CLI support for all parameters
- **Type Safety**: Complete type hints throughout the codebase

## Demo

![Wave Animation Demo](docs/demo.gif)

*Animated sine waves with envelope tracing*

## Installation

### From PyPI (when published)
```bash
pip install wave-anim
```

### From Source
```bash
git clone https://github.com/FNNDSC/waveAnim.git
cd waveAnim
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/FNNDSC/waveAnim.git
cd waveAnim
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Command Line Usage
```bash
# Basic usage with defaults
sine-animator

# Custom parameters
sine-animator --waves 15 --amp-min 0.1 --amp-max 4.0 --freq-min 0.2 --freq-max 6.0

# Tight y-axis margins
sine-animator --y-margin 0.05 --speed 30
```

### Programmatic Usage
```python
from sine_wave_animator import create_envelope_tracer

# Basic usage
anim = create_envelope_tracer()

# Custom parameters
anim = create_envelope_tracer(
    num_waves=12,
    amplitude_range=(0.5, 3.5),
    frequency_range=(0.1, 5.0),
    animation_speed=25,
    y_margin=0.08
)
```

### Advanced Usage
```python
from sine_wave_animator import SineWaveAnimator

# Full control with class interface
animator = SineWaveAnimator(
    num_waves=20,
    x_range=(-3*np.pi, 3*np.pi),
    num_points=1500,
    amplitude_range=(0.2, 4.0),
    frequency_range=(0.1, 8.0),
    y_margin=0.15
)

animator.animation_speed = 20
anim = animator.run_animation(interval=30, save_gif=True)
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--waves` | int | 8 | Number of sine waves to generate |
| `--speed` | int | 25 | Animation speed (points per frame) |
| `--amp-min` | float | 0.2 | Minimum wave amplitude |
| `--amp-max` | float | 3.0 | Maximum wave amplitude |
| `--freq-min` | float | 0.2 | Minimum wave frequency |
| `--freq-max` | float | 4.0 | Maximum wave frequency |
| `--y-margin` | float | 0.1 | Y-axis margin as fraction of max amplitude |

## Examples

### Wide Amplitude Range
```bash
sine-animator --waves 10 --amp-min 0.1 --amp-max 5.0 --freq-min 0.5 --freq-max 2.0
```

### High Frequency Waves
```bash
sine-animator --waves 20 --freq-min 1.0 --freq-max 10.0 --speed 40
```

### Slow, Detailed Animation
```bash
sine-animator --waves 6 --speed 10 --y-margin 0.05
```

## API Reference

### Classes

#### `SineWaveAnimator`
Main class for creating animated sine wave visualizations.

**Parameters:**
- `num_waves` (int): Number of sine waves to generate
- `x_range` (Tuple[float, float]): X-axis range in radians
- `num_points` (int): Number of sample points
- `amplitude_range` (Tuple[float, float]): Min/max amplitudes
- `frequency_range` (Tuple[float, float]): Min/max frequencies  
- `y_margin` (float): Y-axis margin fraction

**Methods:**
- `generate_wave(wave_idx, x_points=None)`: Generate single sine wave
- `calculate_pointwise_maximum()`: Calculate envelope
- `run_animation(interval=50, save_gif=False)`: Start animation

### Functions

#### `create_envelope_tracer(**kwargs)`
High-level function to create and run animations with sensible defaults.

#### `main()`
Command-line entry point with argument parsing.

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Pillow >= 8.3.0 (for GIF export)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/FNNDSC/waveAnim.git
cd waveAnim
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

### Running Tests
```bash
pytest tests/
```

### Type Checking
```bash
mypy sine_wave_animator/
```

### Code Formatting
```bash
black sine_wave_animator/
isort sine_wave_animator/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [matplotlib](https://matplotlib.org/) for animation
- Uses [NumPy](https://numpy.org/) for mathematical operations
- Inspired by signal processing and wave interference visualization

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## Support

- Create an [issue](https://github.com/FNNDSC/waveAnim/issues) for bug reports
- Start a [discussion](https://github.com/FNNDSC/waveAnim/discussions) for questions
- Check the [wiki](https://github.com/FNNDSC/waveAnim/wiki) for additional documentation
