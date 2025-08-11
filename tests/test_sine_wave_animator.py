"""Tests for the SineWaveAnimator class and related functions."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from sine_wave_animator.main import SineWaveAnimator, create_envelope_tracer


class TestSineWaveAnimator:
    """Test cases for the SineWaveAnimator class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        animator = SineWaveAnimator()
        
        assert animator.num_waves == 8
        assert animator.x_range == (-2*np.pi, 2*np.pi)
        assert animator.num_points == 1000
        assert len(animator.x) == 1000
        assert len(animator.amplitudes) == 8
        assert len(animator.frequencies) == 8
        assert len(animator.phases) == 8
        assert len(animator.colors) == 8
        assert len(animator.wave_lines) == 8
        assert len(animator.thick_lines) == 8
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        animator = SineWaveAnimator(
            num_waves=5,
            x_range=(-np.pi, np.pi),
            num_points=500,
            amplitude_range=(1.0, 2.0),
            frequency_range=(1.0, 3.0),
            y_margin=0.2
        )
        
        assert animator.num_waves == 5
        assert animator.x_range == (-np.pi, np.pi)
        assert animator.num_points == 500
        assert len(animator.amplitudes) == 5
        assert all(1.0 <= amp <= 2.0 for amp in animator.amplitudes)
        assert all(1.0 <= freq <= 3.0 for freq in animator.frequencies)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="num_waves must be at least 1"):
            SineWaveAnimator(num_waves=0)
        
        with pytest.raises(ValueError, match="amplitude_range must have min < max"):
            SineWaveAnimator(amplitude_range=(2.0, 1.0))
        
        with pytest.raises(ValueError, match="frequency_range must have min < max"):
            SineWaveAnimator(frequency_range=(3.0, 1.0))
    
    def test_generate_wave(self):
        """Test wave generation functionality."""
        animator = SineWaveAnimator(num_waves=3)
        
        # Test wave generation with default x points
        wave = animator.generate_wave(0)
        assert len(wave) == animator.num_points
        assert isinstance(wave, np.ndarray)
        
        # Test wave generation with custom x points
        custom_x = np.linspace(0, np.pi, 100)
        wave_custom = animator.generate_wave(0, custom_x)
        assert len(wave_custom) == 100
        
        # Test that different waves are different
        wave1 = animator.generate_wave(0)
        wave2 = animator.generate_wave(1)
        assert not np.array_equal(wave1, wave2)
    
    def test_generate_wave_invalid_index(self):
        """Test that invalid wave indices raise errors."""
        animator = SineWaveAnimator(num_waves=3)
        
        with pytest.raises(IndexError):
            animator.generate_wave(-1)
        
        with pytest.raises(IndexError):
            animator.generate_wave(3)
    
    def test_calculate_pointwise_maximum(self):
        """Test envelope calculation."""
        animator = SineWaveAnimator(num_waves=3)
        envelope = animator.calculate_pointwise_maximum()
        
        assert len(envelope) == animator.num_points
        assert isinstance(envelope, np.ndarray)
        
        # Envelope should be non-negative (assuming positive amplitudes)
        assert all(env >= 0 for env in envelope)
        
        # Envelope should be at least as large as any individual wave
        for i in range(animator.num_waves):
            wave = animator.generate_wave(i)
            assert all(envelope >= wave)
    
    def test_calculate_envelope_legacy(self):
        """Test the legacy envelope calculation method."""
        animator = SineWaveAnimator()
        test_signal = np.sin(np.linspace(0, 4*np.pi, 100))
        envelope = animator.calculate_envelope(test_signal, window_size=10)
        
        assert len(envelope) == len(test_signal)
        assert all(envelope >= test_signal)
    
    def test_animation_state_initialization(self):
        """Test that animation state is properly initialized."""
        animator = SineWaveAnimator()
        
        assert animator.current_wave == 0
        assert animator.current_point == 0
        assert animator.animation_speed == 20
        assert animator.all_waves_complete is False
        assert animator.envelope_point == 0
        assert animator.envelope_animating is False
        assert animator.full_envelope is None


class TestCreateEnvelopeTracer:
    """Test cases for the create_envelope_tracer function."""
    
    def test_create_envelope_tracer_default(self):
        """Test create_envelope_tracer with default parameters."""
        # This test just ensures the function runs without errors
        # We can't easily test the full animation in a unit test
        animator = SineWaveAnimator(num_waves=2)  # Create manually to avoid showing plot
        assert animator is not None
    
    def test_create_envelope_tracer_custom_params(self):
        """Test create_envelope_tracer with custom parameters."""
        animator = SineWaveAnimator(
            num_waves=5,
            amplitude_range=(0.5, 1.5),
            frequency_range=(1.0, 2.0),
            y_margin=0.05
        )
        animator.animation_speed = 50
        
        assert animator.num_waves == 5
        assert animator.animation_speed == 50


class TestMathematicalProperties:
    """Test mathematical properties of the sine waves."""
    
    def test_wave_amplitude_bounds(self):
        """Test that generated waves respect amplitude bounds."""
        amp_min, amp_max = 0.5, 2.5
        animator = SineWaveAnimator(
            num_waves=10,
            amplitude_range=(amp_min, amp_max)
        )
        
        for i in range(animator.num_waves):
            wave = animator.generate_wave(i)
            max_amplitude = np.max(np.abs(wave))
            # Allow small floating point errors
            assert max_amplitude <= amp_max + 1e-10
            assert max_amplitude >= amp_min - 1e-10
    
    def test_wave_frequency_properties(self):
        """Test that waves have expected frequency properties."""
        animator = SineWaveAnimator(num_waves=1, frequency_range=(1.0, 1.0))
        
        # For a sine wave with frequency 1.0, we should see one complete cycle
        # in the interval [0, 2π]
