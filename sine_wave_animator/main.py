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
        full_envelope (np.ndarray): Pre-calculated envelope data
        envelope_type (str): Type of envelope calculation ("max" or "sum")
    """
    
    def __init__(self, num_waves: int = 8, x_range: Tuple[float, float] = (-2*np.pi, 2*np.pi), 
                 num_points: int = 1000, amplitude_range: Tuple[float, float] = (0.2, 3.0), 
                 frequency_range: Tuple[float, float] = (0.2, 4.0), y_margin: float = 0.1,
                 envelope_type: str = "max") -> None:
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
        self.x: npt.NDArray[np.float64] = np.linspace(x_range[0], x_range[1], num_points)
        
        # Generate random wave parameters with specified ranges
        np.random.seed(42)  # For reproducible results
        self.amplitudes: npt.NDArray[np.float64] = np.random.uniform(amplitude_range[0], amplitude_range[1], num_waves)
        self.frequencies: npt.NDArray[np.float64] = np.random.uniform(frequency_range[0], frequency_range[1], num_waves)
        self.phases: npt.NDArray[np.float64] = np.random.uniform(0, 2*np.pi, num_waves)
        
        # Generate colors using HSV for better distribution
        hues: npt.NDArray[np.float64] = np.linspace(0, 1, num_waves, endpoint=False)
        self.colors: List[npt.NDArray[np.float64]] = [hsv_to_rgb([h, 0.8, 0.9]) for h in hues]
        
        # Setup figure and axis
        plt.style.use('dark_background')
        self.fig: matplotlib.figure.Figure
        self.ax: matplotlib.axes.Axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_xlim(x_range[0], x_range[1])
        
        # Set y-axis limits based on envelope type and amplitude with margin
        if envelope_type == "max":
            max_y_value = amplitude_range[1]
            envelope_label = "Max Envelope"
        else:  # envelope_type == "sum"
            # For sum, waves can interfere constructively or destructively
            # Maximum possible value is sum of all max amplitudes
            # Minimum possible value is negative sum of all max amplitudes
            max_y_value = amplitude_range[1] * num_waves
            envelope_label = "Wave Sum"
            
        y_range: float = max_y_value * (1 + y_margin)
        self.ax.set_ylim(-y_range, y_range)
        
        self.ax.set_xlabel('x (radians)', color='white')
        self.ax.set_ylabel('Amplitude', color='white')
        self.ax.set_title(f'Animated Sine Waves with {envelope_label}', color='white')
        self.ax.grid(True, alpha=0.3)
        
        # Initialize line objects
        self.wave_lines: List[matplotlib.lines.Line2D] = []
        self.thick_lines: List[matplotlib.lines.Line2D] = []
        for i in range(num_waves):
            # Thin line (final state)
            line: matplotlib.lines.Line2D
            line, = self.ax.plot([], [], color=self.colors[i], linewidth=1, alpha=0.7)
            self.wave_lines.append(line)
            # Thick line (drawing state)
            thick_line: matplotlib.lines.Line2D
            thick_line, = self.ax.plot([], [], color=self.colors[i], linewidth=5, alpha=1.0)
            self.thick_lines.append(thick_line)
        
        # Envelope line
        self.envelope_line: matplotlib.lines.Line2D
        self.envelope_line, = self.ax.plot([], [], color='yellow', linewidth=2, alpha=0.9, label=envelope_label)
        self.ax.legend()
        
        # Animation state
        self.current_wave: int = 0
        self.current_point: int = 0
        self.animation_speed: int = 20  # points per frame
        self.all_waves_complete: bool = False
        self.envelope_point: int = 0
        self.envelope_animating: bool = False
        self.full_envelope: Optional[npt.NDArray[np.float64]] = None
        
    def generate_wave(self, wave_idx: int, x_points: Optional[npt.NDArray[np.float64]] = None) -> npt.NDArray[np.float64]:
        """
        Generate a single sine wave with the specified parameters.
        
        Args:
            wave_idx (int): Index of the wave to generate (0 to num_waves-1)
            x_points (np.ndarray, optional): Array of x-coordinates to evaluate.
                                           If None, uses self.x. Defaults to None.
        
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
        else:  # envelope_type == "sum"
            return self.calculate_wave_sum()
    
    def calculate_wave_sum(self) -> npt.NDArray[np.float64]:
        """
        Calculate the sum of all sine waves at each point.
        
        Returns:
            np.ndarray: Array of summed wave values at each x-coordinate
        """
        # Sum all waves at each x point
        wave_sum: npt.NDArray[np.float64] = np.zeros_like(self.x)
        for i in range(self.num_waves):
            wave_sum += self.generate_wave(i)
        
    def calculate_pointwise_maximum(self) -> npt.NDArray[np.float64]:
        """
        Calculate the pointwise maximum across all sine waves.
        
        This method evaluates all sine waves at each x-coordinate and returns
        the maximum value at each point, creating an envelope that follows
        the peaks of the highest wave at each location.
        
        Returns:
            np.ndarray: Array of maximum values at each x-coordinate
        """
        # Get all wave values at each x point
        all_waves: npt.NDArray[np.float64] = np.zeros((self.num_waves, len(self.x)))
        for i in range(self.num_waves):
            all_waves[i] = self.generate_wave(i)
        
        # Take maximum across all waves at each x point
        envelope: npt.NDArray[np.float64] = np.max(all_waves, axis=0)
        return envelope
    
    def calculate_envelope_legacy(self, signal: npt.NDArray[np.float64], window_size: int = 50) -> npt.NDArray[np.float64]:
        """
        Calculate upper envelope of a signal using local maxima.
        
        This method is kept for compatibility but is not used in the current
        implementation which uses pointwise maximum instead.
        
        Args:
            signal (np.ndarray): Input signal to calculate envelope for
            window_size (int, optional): Size of sliding window for local maxima.
                                       Defaults to 50.
        
        Returns:
            np.ndarray: Upper envelope of the input signal
        """
        envelope: npt.NDArray[np.float64] = np.zeros_like(signal)
        half_window: int = window_size // 2
        
        for i in range(len(signal)):
            start_idx: int = max(0, i - half_window)
            end_idx: int = min(len(signal), i + half_window + 1)
            envelope[i] = np.max(signal[start_idx:end_idx])
        
        return envelope
    
    def animate(self, frame: int) -> List[Union[matplotlib.lines.Line2D]]:
        """
        Animation function called for each frame by matplotlib's FuncAnimation.
        
        This method handles the progressive drawing of sine waves and the envelope.
        It draws each wave from left to right with thick lines, then replaces them
        with thin lines when complete. After all waves are drawn, it animates
        the envelope tracing.
        
        Args:
            frame (int): Frame number (automatically provided by FuncAnimation)
        
        Returns:
            list: List of matplotlib artist objects that were modified
        """
        if self.current_wave < self.num_waves:
            # Currently drawing a wave
            wave_idx: int = self.current_wave
            
            # Calculate how many points to show
            end_point: int = min(self.current_point + self.animation_speed, len(self.x))
            
            # Get x and y data for current progress
            x_data: npt.NDArray[np.float64] = self.x[:end_point]
            y_data: npt.NDArray[np.float64] = self.generate_wave(wave_idx, x_data)
            
            # Update thick line (currently drawing)
            self.thick_lines[wave_idx].set_data(x_data, y_data)
            
            # Update progress
            self.current_point = end_point
            
            # Check if current wave is complete
            if self.current_point >= len(self.x):
                # Replace thick line with thin line
                self.thick_lines[wave_idx].set_data([], [])  # Clear thick line
                self.wave_lines[wave_idx].set_data(self.x, self.generate_wave(wave_idx))
                
                # Move to next wave
                self.current_wave += 1
                self.current_point = 0
                
        else:
            # All waves complete, start animating envelope
            if not self.envelope_animating:
                self.envelope_animating = True
                self.envelope_point = 0
                # Pre-calculate the full envelope once
                self.full_envelope = self.calculate_envelope()
            
            if self.envelope_animating and not self.all_waves_complete and self.full_envelope is not None:
                # Animate envelope drawing
                end_point: int = min(self.envelope_point + self.animation_speed, len(self.x))
                
                # Show envelope up to current point
                x_data: npt.NDArray[np.float64] = self.x[:end_point]
                y_data: npt.NDArray[np.float64] = self.full_envelope[:end_point]
                self.envelope_line.set_data(x_data, y_data)
                
                self.envelope_point = end_point
                
                # Check if envelope animation is complete
                if self.envelope_point >= len(self.x):
                    self.all_waves_complete = True
        
        return self.wave_lines + self.thick_lines + [self.envelope_line]
    
    def run_animation(self, interval: int = 50, save_gif: bool = False) -> matplotlib.animation.FuncAnimation:
        """
        Start and run the sine wave animation.
        
        This method creates a matplotlib FuncAnimation and starts the display.
        The animation will continue until all waves and the envelope are drawn.
        
        Args:
            interval (int, optional): Time between frames in milliseconds. 
                                    Defaults to 50.
            save_gif (bool, optional): Whether to save animation as GIF. 
                                     Defaults to False.
        
        Returns:
            matplotlib.animation.FuncAnimation: The animation object
        """
        anim: matplotlib.animation.FuncAnimation = animation.FuncAnimation(
            self.fig, self.animate, interval=interval, 
            blit=True, repeat=False
        )
        
        if save_gif:
            anim.save('sine_waves_animation.gif', writer='pillow', fps=20)
        
        plt.tight_layout()
        plt.show()
        return anim


def create_envelope_tracer(num_waves: int = 10, animation_speed: int = 30, 
                          amplitude_range: Tuple[float, float] = (0.2, 3.0), 
                          frequency_range: Tuple[float, float] = (0.2, 4.0), 
                          y_margin: float = 0.1, envelope_type: str = "max") -> matplotlib.animation.FuncAnimation:
    """
    Create and run a sine wave animation with envelope tracing.
    
    This is a high-level convenience function that creates a SineWaveAnimator
    instance with the specified parameters and runs the animation.
    
    Args:
        num_waves (int, optional): Number of sine waves to generate. Defaults to 10.
        animation_speed (int, optional): Points drawn per frame (higher = faster). 
                                       Defaults to 30.
        amplitude_range (tuple, optional): Range of wave amplitudes as (min, max). 
                                         Defaults to (0.2, 3.0).
        frequency_range (tuple, optional): Range of wave frequencies as (min, max). 
                                         Defaults to (0.2, 4.0).
        y_margin (float, optional): Fraction of max amplitude to add as y-axis margin. 
                                  Defaults to 0.1 (10%).
        envelope_type (str, optional): Type of envelope ("max" or "sum"). 
                                     Defaults to "max".
    
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
        
    Example:
        # Create animation with custom parameters
        anim = create_envelope_tracer(
            num_waves=15,
            amplitude_range=(0.5, 4.0),
            frequency_range=(0.1, 6.0),
            y_margin=0.05,
            envelope_type="sum"
        )
    """
    animator: SineWaveAnimator = SineWaveAnimator(num_waves=num_waves, 
                                                 amplitude_range=amplitude_range,
                                                 frequency_range=frequency_range,
                                                 y_margin=y_margin,
                                                 envelope_type=envelope_type)
    animator.animation_speed = animation_speed
    
    print(f"Generating {num_waves} sine waves...")
    print("Wave parameters:")
    for i in range(num_waves):
        print(f"  Wave {i+1}: A={animator.amplitudes[i]:.2f}, "
              f"f={animator.frequencies[i]:.2f}, φ={animator.phases[i]:.2f}")
    
    return animator.run_animation()


def main() -> matplotlib.animation.FuncAnimation:
    """
    Main entry point for command line usage.
    
    Parses command line arguments and creates a sine wave animation with
    the specified parameters. Provides a full command-line interface for
    controlling all animation parameters.
    
    Command line arguments:
        --waves: Number of sine waves (default: 8)
        --speed: Animation speed in points per frame (default: 25)
        --amp-min: Minimum amplitude (default: 0.2)
        --amp-max: Maximum amplitude (default: 3.0)
        --freq-min: Minimum frequency (default: 0.2)
        --freq-max: Maximum frequency (default: 4.0)
        --y-margin: Y-axis margin as fraction of max amplitude (default: 0.1)
        --envelope: Envelope type - 'max' for pointwise maximum, 'sum' for wave sum (default: max)
    
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
        
    Example:
        Command line usage:
        $ sine-animator --waves 15 --amp-min 0.1 --amp-max 4.0 --y-margin 0.05 --envelope sum
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Animate sine waves with upper envelope")
    parser.add_argument("--waves", type=int, default=8, help="Number of sine waves (default: 8)")
    parser.add_argument("--speed", type=int, default=25, help="Animation speed - points per frame (default: 25)")
    parser.add_argument("--amp-min", type=float, default=0.2, help="Minimum amplitude (default: 0.2)")
    parser.add_argument("--amp-max", type=float, default=3.0, help="Maximum amplitude (default: 3.0)")
    parser.add_argument("--freq-min", type=float, default=0.2, help="Minimum frequency (default: 0.2)")
    parser.add_argument("--freq-max", type=float, default=4.0, help="Maximum frequency (default: 4.0)")
    parser.add_argument("--y-margin", type=float, default=0.1, help="Y-axis margin as fraction of max amplitude (default: 0.1)")
    parser.add_argument("--envelope", choices=["max", "sum"], default="max", help="Envelope type: 'max' for pointwise maximum, 'sum' for wave sum (default: max)")
    
    args: argparse.Namespace = parser.parse_args()
    
    print(f"Starting sine wave animation with {args.waves} waves...")
    print(f"Amplitude range: {args.amp_min} to {args.amp_max}")
    print(f"Frequency range: {args.freq_min} to {args.freq_max}")
    print(f"Y-axis margin: {args.y_margin * 100}%")
    print(f"Envelope type: {args.envelope}")
    
    anim: matplotlib.animation.FuncAnimation = create_envelope_tracer(num_waves=args.waves, 
                                                                     animation_speed=args.speed,
                                                                     amplitude_range=(args.amp_min, args.amp_max),
                                                                     frequency_range=(args.freq_min, args.freq_max),
                                                                     y_margin=args.y_margin,
                                                                     envelope_type=args.envelope)
    return anim


# Run the animation
if __name__ == "__main__":
    # Example with more varied amplitudes and frequencies
    # anim = create_envelope_tracer(num_waves=12, animation_speed=20,
    #                              amplitude_range=(0.1, 4.0), frequency_range=(0.1, 6.0))
    main()
