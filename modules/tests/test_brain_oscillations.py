import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.oscillations import KuramotoModel, NeuralOscillations


def test_kuramoto_synchronization():
    model = KuramotoModel()
    natural_freqs = [2 * np.pi * 1.0, 2 * np.pi * 1.1, 2 * np.pi * 0.9]
    phases = model.simulate(
        natural_freqs,
        coupling_strength=10.0,
        initial_phases=[0.0, 1.0, 2.0],
        duration=1.0,
        sample_rate=1000,
    )
    final = phases[-1]
    initial_spread = 2.0 - 0.0
    final_spread = np.max(final) - np.min(final)
    assert final_spread < initial_spread


def test_generate_realistic_oscillations():
    osc = NeuralOscillations()
    waves = osc.generate_realistic_oscillations(
        num_oscillators=3, duration=0.5, sample_rate=1000, coupling_strength=5.0
    )
    assert waves.shape[0] == 3
    corr = np.corrcoef(waves)
    assert corr[0, 1] > 0 and corr[1, 2] > 0


def test_cross_frequency_coupling():
    osc = NeuralOscillations()
    duration = 0.5
    sample_rate = 1000
    result = osc.cross_frequency_coupling(
        low_freq=5.0,
        high_freq=40.0,
        duration=duration,
        sample_rate=sample_rate,
        coupling_strength=5.0,
    )
    dt = 1.0 / sample_rate
    E_low, I_low = osc.wilson_cowan.simulate(duration=duration, dt=dt, P_e=1.25, P_i=0.5)
    low_amp = E_low - I_low
    phases = osc.kuramoto.simulate(
        [2 * np.pi * 5.0, 2 * np.pi * 40.0],
        5.0,
        [0.0, np.pi / 2],
        duration,
        sample_rate,
    )
    low_signal = low_amp * np.sin(phases[:, 0])
    high_signal = np.sin(phases[:, 1])
    expected = osc.oscillatory_modulation(high_signal, low_signal)
    np.testing.assert_allclose(result, expected)
