"""
Animated Sine Wave Generator with Envelope Tracing

This module provides a real-time animation system for generating multiple sine waves
with different amplitudes, frequencies, and phases. Each wave is drawn progressively
from left to right with thick lines during animation, then switches to thin lines
when complete. After all waves are drawn, an upper envelope is animated that traces
the pointwise maximum across all sine waves.

The animation uses matplotlib's FuncAnimation for smooth real-time rendering and
supports both programmatic control and command-line interface for parameter adjustment.

Classes:
   SineWaveAnimator: Main class handling wave generation and animation

Functions:
   create_envelope_tracer: High-level function to create and run animations
   main: Command-line entry point with argument parsing

Example:
   # Basic usage
   from sine_wave_animator import create_envelope_tracer
   anim = create_envelope_tracer(num_waves=10, amplitude_range=(0.5, 3.0))

   # Command line usage
   sine-animator --waves 15 --amp-min 0.1 --amp-max 4.0 --freq-min 0.2 --freq-max 6.0

Author: Generated for sine wave visualization
Version: 1.0.0
"""

from typing import Tuple, List, Optional, Union
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.figure
import matplotlib.axes
import matplotlib.lines
from matplotlib.colors import hsv_to_rgb
import time
import argparse
import pudb


def pause_keep_display() -> None:
    """Pause while keeping current display intact."""
    input("Press Enter to contine...")


class SineWaveAnimator:
    """
    A class for creating animated sine wave visualizations with envelope tracing.

    This class generates multiple sine waves with random parameters and animates
    them progressively from left to right. After all waves are complete, it
    calculates and animates the upper envelope showing either the pointwise maximum
    across all waves or the sum of all waves.

    Attributes:
        num_waves (int): Number of sine waves to generate
        x_range (tuple): Range of x-axis values as (min, max)
        num_points (int): Number of points to sample along x-axis
        x (np.ndarray): Array of x-coordinates for plotting
        amplitudes (np.ndarray): Array of wave amplitudes
        frequencies (np.ndarray): Array of wave frequencies
        phases (np.ndarray): Array of wave phase shifts
        colors (list): List of RGB color tuples for each wave
        fig (matplotlib.figure.Figure): The matplotlib figure object
        ax (matplotlib.axes.Axes): The matplotlib axes object
        wave_lines (list): List of matplotlib line objects for final thin waves
        thick_lines (list): List of matplotlib line objects for animated thick waves
        envelope_line (matplotlib.lines.Line2D): Line object for the envelope
        current_wave (int): Index of currently animating wave
        current_point (int): Current point index within the animating wave
        animation_speed (int): Number of points to draw per animation frame
        all_waves_complete (bool): Flag indicating if all animations are finished
        envelope_point (int): Current point index for envelope animation
        envelope_animating (bool): Flag indicating if envelope is currently animating
        envelope_first (bool): Flag to show envelope animation first
        envelope_first_complete (bool): Flag indicating initial envelope is complete
        pause_enabled (bool): Flag to enable pausing at key points
        animation_obj (matplotlib.animation.FuncAnimation): Reference to animation object
        full_envelope (np.ndarray): Pre-calculated envelope data
        envelope_type (str): Type of envelope calculation ("max" or "sum")
    """

    def __init__(
        self,
        num_waves: int = 8,
        x_range: Tuple[float, float] = (-2 * np.pi, 2 * np.pi),
        num_points: int = 1000,
        amplitude_range: Tuple[float, float] = (0.2, 3.0),
        frequency_range: Tuple[float, float] = (0.2, 4.0),
        y_margin: float = 0.1,
        envelope_type: str = "max",
        envelope_first: bool = False,
        pause_enabled: bool = False,
    ) -> None:
        """
        Initialize the SineWaveAnimator with specified parameters.

        Args:
            num_waves (int, optional): Number of sine waves to generate. Defaults to 8.
            x_range (tuple, optional): Range of x-axis as (min, max) in radians.
                                      Defaults to (-2π, 2π).
            num_points (int, optional): Number of sample points along x-axis.
                                       Defaults to 1000.
            amplitude_range (tuple, optional): Range of wave amplitudes as (min, max).
                                              Defaults to (0.2, 3.0).
            frequency_range (tuple, optional): Range of wave frequencies as (min, max).
                                              Defaults to (0.2, 4.0).
            y_margin (float, optional): Fraction of max amplitude to add as y-axis margin.
                                       Defaults to 0.1 (10%).
            envelope_type (str, optional): Type of envelope ("max" or "sum").
                                          Defaults to "max".
            envelope_first (bool, optional): Show envelope animation first.
                                            Defaults to False.
            pause_enabled (bool, optional): Enable pausing at key animation points.
                                           Defaults to False.

        Raises:
            ValueError: If num_waves < 1 or if range tuples have min >= max
        """
        if num_waves < 1:
            raise ValueError("num_waves must be at least 1")
        if amplitude_range[0] >= amplitude_range[1]:
            raise ValueError("amplitude_range must have min < max")
        if frequency_range[0] >= frequency_range[1]:
            raise ValueError("frequency_range must have min < max")
        if envelope_type not in ["max", "sum"]:
            raise ValueError("envelope_type must be 'max' or 'sum'")

        self.num_waves: int = num_waves
        self.x_range: Tuple[float, float] = x_range
        self.num_points: int = num_points
        self.envelope_type: str = envelope_type
        self.envelope_first: bool = envelope_first
        self.pause_enabled: bool = pause_enabled

        self.x: npt.NDArray[np.float64] = np.linspace(x_range[0], x_range[1], num_points)

        # Generate random wave parameters with specified ranges
        np.random.seed(42)  # For reproducible results
        self.amplitudes: npt.NDArray[np.float64] = np.random.uniform(
            amplitude_range[0], amplitude_range[1], num_waves
        )
        self.frequencies: npt.NDArray[np.float64] = np.random.uniform(
            frequency_range[0], frequency_range[1], num_waves
        )
        self.phases: npt.NDArray[np.float64] = np.random.uniform(0, 2 * np.pi, num_waves)

        # Generate colors using HSV for better distribution
        hues: npt.NDArray[np.float64] = np.linspace(0, 1, num_waves, endpoint=False)
        self.colors: List[npt.NDArray[np.float64]] = [hsv_to_rgb([h, 0.8, 0.9]) for h in hues]

        # Setup figure and axis
        plt.style.use("dark_background")
        self.fig: matplotlib.figure.Figure
        self.ax: matplotlib.axes.Axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(x_range[0], x_range[1])

        # Set y-axis limits based on envelope type and amplitude with margin
        if envelope_type == "max":
            max_y_value: float = amplitude_range[1]
            envelope_label: str = "Max Envelope"
        else:  # envelope_type == "sum"
            max_y_value = amplitude_range[1] * num_waves
            envelope_label = "Wave Sum"

        y_range: float = max_y_value * (1 + y_margin)
        self.ax.set_ylim(-y_range, y_range)

        self.ax.set_xlabel("x (radians)", color="white")
        self.ax.set_ylabel("Amplitude", color="white")
        self.ax.set_title(f"Animated Sine Waves with {envelope_label}", color="white")
        self.ax.grid(True, alpha=0.3)

        # Initialize line objects
        self.wave_lines: List[matplotlib.lines.Line2D] = []
        self.thick_lines: List[matplotlib.lines.Line2D] = []

        for i in range(num_waves):
            line: matplotlib.lines.Line2D
            (line,) = self.ax.plot([], [], color=self.colors[i], linewidth=1, alpha=0.7)
            self.wave_lines.append(line)

            thick_line: matplotlib.lines.Line2D
            (thick_line,) = self.ax.plot([], [], color=self.colors[i], linewidth=5, alpha=1.0)
            self.thick_lines.append(thick_line)

        # Envelope line
        self.envelope_line: matplotlib.lines.Line2D
        (self.envelope_line,) = self.ax.plot(
            [], [], color="yellow", linewidth=2, alpha=0.9, label=envelope_label
        )
        self.ax.legend()

        # Animation state
        self.current_wave: int = 0
        self.current_point: int = 0
        self.animation_speed: int = 20
        self.all_waves_complete: bool = False
        self.envelope_point: int = 0
        self.envelope_animating: bool = False
        self.envelope_first_complete: bool = False
        self.animation_obj: Optional[matplotlib.animation.FuncAnimation] = None
        self.full_envelope: Optional[npt.NDArray[np.float64]] = None

    def generate_wave(
        self, wave_idx: int, x_points: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray[np.float64]:
        """
        Generate a single sine wave with the specified parameters.

        Args:
            wave_idx (int): Index of the wave to generate (0 to num_waves-1)
            x_points (np.ndarray, optional): Array of x-coordinates to evaluate.

        Returns:
            np.ndarray: Array of y-values for the sine wave

        Raises:
            IndexError: If wave_idx is out of range
        """
        if wave_idx < 0 or wave_idx >= self.num_waves:
            raise IndexError(f"wave_idx {wave_idx} out of range [0, {self.num_waves-1}]")

        if x_points is None:
            x_points = self.x

        result: npt.NDArray[np.float64] = self.amplitudes[wave_idx] * np.sin(
            self.frequencies[wave_idx] * x_points + self.phases[wave_idx]
        )
        return result

    def calculate_envelope(self) -> npt.NDArray[np.float64]:
        """Calculate the envelope based on the specified envelope type."""
        if self.envelope_type == "max":
            return self.calculate_pointwise_maximum()
        else:
            return self.calculate_wave_sum()

    def calculate_wave_sum(self) -> npt.NDArray[np.float64]:
        """Calculate the sum of all sine waves at each point."""
        wave_sum: npt.NDArray[np.float64] = np.zeros_like(self.x)
        for i in range(self.num_waves):
            wave_sum += self.generate_wave(i)
        return wave_sum

    def calculate_pointwise_maximum(self) -> npt.NDArray[np.float64]:
        """Calculate the pointwise maximum across all sine waves."""
        all_waves: npt.NDArray[np.float64] = np.zeros((self.num_waves, len(self.x)))
        for i in range(self.num_waves):
            all_waves[i] = self.generate_wave(i)
        envelope: npt.NDArray[np.float64] = np.max(all_waves, axis=0)
        return envelope

    def animate(self, frame: int) -> List[Union[matplotlib.lines.Line2D]]:
        """Animation function called for each frame by matplotlib's FuncAnimation."""
        # Phase 1: Initial envelope animation (if enabled)
        if self.envelope_first and not self.envelope_first_complete:
            if not self.envelope_animating:
                self.envelope_animating = True
                self.envelope_point = 0
                self.full_envelope = self.calculate_envelope()

            if self.full_envelope is not None:
                end_point: int = min(self.envelope_point + self.animation_speed, len(self.x))
                x_data: npt.NDArray[np.float64] = self.x[:end_point]
                y_data: npt.NDArray[np.float64] = self.full_envelope[:end_point]
                self.envelope_line.set_data(x_data, y_data)
                self.envelope_point = end_point

                if self.envelope_point >= len(self.x):
                    self.envelope_first_complete = True
                    self.envelope_animating = False
                    if self.pause_enabled and self.animation_obj is not None:
                        # Stop animation completely
                        self.animation_obj.event_source.stop()
                        print("Press any key to continue after first envelope...")
                        pause_keep_display()
                        # Fade envelope and restart
                        self.envelope_line.set_color("#404040")
                        self.envelope_line.set_alpha(0.3)
                        self.animation_obj.event_source.start()
                    else:
                        self.envelope_line.set_color("#404040")
                        self.envelope_line.set_alpha(0.3)

        # Phase 2: Individual wave animations
        elif self.current_wave < self.num_waves:
            wave_idx: int = self.current_wave
            end_point: int = min(self.current_point + self.animation_speed, len(self.x))

            x_data: npt.NDArray[np.float64] = self.x[:end_point]
            y_data: npt.NDArray[np.float64] = self.generate_wave(wave_idx, x_data)

            self.thick_lines[wave_idx].set_data(x_data, y_data)
            self.current_point = end_point

            if self.current_point >= len(self.x):
                self.thick_lines[wave_idx].set_data([], [])
                self.wave_lines[wave_idx].set_data(self.x, self.generate_wave(wave_idx))
                self.current_wave += 1
                self.current_point = 0

                # Pause after all waves if enabled
                if (
                    self.current_wave >= self.num_waves
                    and self.pause_enabled
                    and self.animation_obj is not None
                ):
                    # Stop animation completely
                    self.animation_obj.event_source.stop()
                    print("Press any key to continue after wave animations...")
                    pause_keep_display()
                    # Restart animation
                    self.animation_obj.event_source.start()

        # Phase 3: Final envelope animation
        else:
            if not self.envelope_animating:
                self.envelope_animating = True
                self.envelope_point = 0
                if self.full_envelope is None:
                    self.full_envelope = self.calculate_envelope()
                self.envelope_line.set_color("yellow")
                self.envelope_line.set_alpha(0.9)

            if (
                self.envelope_animating
                and not self.all_waves_complete
                and self.full_envelope is not None
            ):
                end_point: int = min(self.envelope_point + self.animation_speed, len(self.x))
                x_data: npt.NDArray[np.float64] = self.x[:end_point]
                y_data: npt.NDArray[np.float64] = self.full_envelope[:end_point]
                self.envelope_line.set_data(x_data, y_data)
                self.envelope_point = end_point

                if self.envelope_point >= len(self.x):
                    self.all_waves_complete = True

        return self.wave_lines + self.thick_lines + [self.envelope_line]

    def run_animation(
        self, interval: int = 50, save_gif: bool = False
    ) -> matplotlib.animation.FuncAnimation:
        """Start and run the sine wave animation."""
        self.animation_obj = animation.FuncAnimation(
            self.fig, self.animate, interval=interval, blit=True, repeat=False
        )

        if save_gif:
            self.animation_obj.save("sine_waves_animation.gif", writer="pillow", fps=20)

        plt.tight_layout()
        plt.show()

        return self.animation_obj


def create_envelope_tracer(
    num_waves: int = 10,
    animation_speed: int = 30,
    amplitude_range: Tuple[float, float] = (0.2, 3.0),
    frequency_range: Tuple[float, float] = (0.2, 4.0),
    y_margin: float = 0.1,
    envelope_type: str = "max",
    envelope_first: bool = False,
    pause_enabled: bool = False,
) -> matplotlib.animation.FuncAnimation:
    """Create and run a sine wave animation with envelope tracing."""
    animator: SineWaveAnimator = SineWaveAnimator(
        num_waves=num_waves,
        amplitude_range=amplitude_range,
        frequency_range=frequency_range,
        y_margin=y_margin,
        envelope_type=envelope_type,
        envelope_first=envelope_first,
        pause_enabled=pause_enabled,
    )
    animator.animation_speed = animation_speed

    print(f"Generating {num_waves} sine waves...")
    print("Wave parameters:")
    for i in range(num_waves):
        print(
            f"  Wave {i+1}: A={animator.amplitudes[i]:.2f}, "
            f"f={animator.frequencies[i]:.2f}, φ={animator.phases[i]:.2f}"
        )

    return animator.run_animation()


def main() -> matplotlib.animation.FuncAnimation:
    """Main entry point for command line usage."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Animate sine waves with upper envelope"
    )
    parser.add_argument("--waves", type=int, default=8, help="Number of sine waves (default: 8)")
    parser.add_argument(
        "--speed", type=int, default=25, help="Animation speed - points per frame (default: 25)"
    )
    parser.add_argument(
        "--amp-min", type=float, default=0.2, help="Minimum amplitude (default: 0.2)"
    )
    parser.add_argument(
        "--amp-max", type=float, default=3.0, help="Maximum amplitude (default: 3.0)"
    )
    parser.add_argument(
        "--freq-min", type=float, default=0.2, help="Minimum frequency (default: 0.2)"
    )
    parser.add_argument(
        "--freq-max", type=float, default=4.0, help="Maximum frequency (default: 4.0)"
    )
    parser.add_argument(
        "--y-margin",
        type=float,
        default=0.1,
        help="Y-axis margin as fraction of max amplitude (default: 0.1)",
    )
    parser.add_argument(
        "--envelope",
        choices=["max", "sum"],
        default="max",
        help="Envelope type: 'max' for pointwise maximum, 'sum' for wave sum (default: max)",
    )
    parser.add_argument(
        "--envelope-first",
        action="store_true",
        help="Show envelope animation first, then waves, then envelope retrace",
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Pause for user input at key animation points",
    )

    args: argparse.Namespace = parser.parse_args()

    print(f"Starting sine wave animation with {args.waves} waves...")
    print(f"Amplitude range: {args.amp_min} to {args.amp_max}")
    print(f"Frequency range: {args.freq_min} to {args.freq_max}")
    print(f"Y-axis margin: {args.y_margin * 100}%")
    print(f"Envelope type: {args.envelope}")
    print(f"Envelope first: {args.envelope_first}")
    print(f"Pause enabled: {args.pause}")

    anim: matplotlib.animation.FuncAnimation = create_envelope_tracer(
        num_waves=args.waves,
        animation_speed=args.speed,
        amplitude_range=(args.amp_min, args.amp_max),
        frequency_range=(args.freq_min, args.freq_max),
        y_margin=args.y_margin,
        envelope_type=args.envelope,
        envelope_first=args.envelope_first,
        pause_enabled=args.pause,
    )

    return anim


if __name__ == "__main__":
    main()
