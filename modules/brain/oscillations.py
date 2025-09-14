import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Wave:
    """Represents a bound oscillatory wave."""

    region: str
    frequency: float
    phase: float
    data: np.ndarray


class NeuralOscillations:
    """Simulate neural oscillatory generators with basic waveform support."""

    class AlphaGenerator:
        def generate_wave(
            self,
            frequency: float = 10.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 10.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    class BetaGenerator:
        def generate_wave(
            self,
            frequency: float = 20.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 20.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    class GammaGenerator:
        def generate_wave(
            self,
            frequency: float = 40.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 40.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    class ThetaGenerator:
        def generate_wave(
            self,
            frequency: float = 6.0,
            duration: float = 1.0,
            phase: float = 0.0,
            sample_rate: int = 1000,
        ) -> np.ndarray:
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            return np.sin(2 * np.pi * frequency * t + phase)

        def bind(
            self,
            region: str,
            frequency: float = 6.0,
            phase: float = 0.0,
            duration: float = 1.0,
            sample_rate: int = 1000,
            lock_to: Optional[Wave] = None,
        ) -> Wave:
            if lock_to is not None:
                phase = lock_to.phase
            data = self.generate_wave(frequency, duration, phase, sample_rate)
            return Wave(region, frequency, phase, data)

    def __init__(self) -> None:
        self.alpha_waves = self.AlphaGenerator()
        self.beta_waves = self.BetaGenerator()
        self.gamma_waves = self.GammaGenerator()
        self.theta_waves = self.ThetaGenerator()

    def synchronize_regions(
        self,
        regions,
        frequency: float = 40.0,
        phase: float = 0.0,
        duration: float = 1.0,
        sample_rate: int = 1000,
    ):
        """Synchronize a list of regions using gamma oscillations with phase locking."""
        if not regions:
            return []
        reference = self.gamma_waves.bind(
            regions[0], frequency=frequency, phase=phase, duration=duration, sample_rate=sample_rate
        )
        bindings = [reference]
        for region in regions[1:]:
            bindings.append(
                self.gamma_waves.bind(
                    region,
                    frequency=frequency,
                    duration=duration,
                    sample_rate=sample_rate,
                    lock_to=reference,
                )
            )
        return bindings

    @staticmethod
    def oscillatory_modulation(carrier_wave: np.ndarray, modulator_wave: np.ndarray) -> np.ndarray:
        """Amplitude modulation of a carrier wave using a modulating wave."""
        return carrier_wave * (1 + modulator_wave)

    def cross_frequency_coupling(self, low_wave: np.ndarray, high_wave: np.ndarray) -> np.ndarray:
        """Simple cross-frequency coupling via phase-amplitude modulation."""
        return self.oscillatory_modulation(high_wave, low_wave)
