from __future__ import annotations

from typing import Dict, List, Tuple


def latency_encode(
    signal: List[float],
    *,
    t_start: float = 0.0,
    t_scale: float = 1.0,
) -> List[Tuple[float, List[int]]]:
    """Encode an analog input vector into spike-time events using latency coding.

    Each value in ``signal`` is interpreted as an amplitude in the range ``[0, 1]``.
    Larger values produce earlier spikes according to
    ``time = t_start + t_scale * (1 - value)``.

    Parameters
    ----------
    signal
        Analog input vector. Values are clamped to ``[0, 1]``.
    t_start
        Base timestamp for all spikes.
    t_scale
        Scaling factor mapping amplitudes to spike latencies.

    Returns
    -------
    list of tuple
        A list of ``(time, spikes)`` pairs sorted by time. ``spikes`` is a list
        with the same length as ``signal`` containing ``1`` at the index of the
        neuron that spikes and ``0`` elsewhere.
    """

    n = len(signal)
    events: Dict[float, List[int]] = {}
    for i, value in enumerate(signal):
        clamped = max(0.0, min(1.0, value))
        time = t_start + t_scale * (1.0 - clamped)
        spikes = events.setdefault(time, [0] * n)
        spikes[i] = 1

    return sorted(events.items())
