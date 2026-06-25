"""StreamingResampler — chunk alignment, length, fidelity."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio.resample import StreamingResampler, resample_clip


def _sine(n: int, rate: int, freq: float = 220.0) -> bytes:
    t = np.arange(n) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype("<i2").tobytes()


def test_resample_24k_to_16k_total_length():
    # 1 second at 24 kHz -> ~1 second at 16 kHz.
    data = _sine(24_000, 24_000)
    out = resample_clip(data, 24_000, 16_000)
    samples = len(out) // 2
    assert samples == pytest.approx(16_000, abs=4)


def test_resample_22050_to_16k_total_length():
    data = _sine(22_050, 22_050)
    out = resample_clip(data, 22_050, 16_000)
    samples = len(out) // 2
    assert samples == pytest.approx(16_000, abs=4)


def test_chunked_output_equals_whole():
    """Feeding the same input in many small chunks must yield the SAME bytes as
    one push — the carry/fractional-position math must be seam-correct."""
    import itertools

    data = _sine(24_000, 24_000)
    whole = resample_clip(data, 24_000, 16_000)

    rs = StreamingResampler(24_000, 16_000)
    out = bytearray()
    # deliberately awkward chunk sizes, incl. odd lengths (half-frame carry).
    # Single pass over the data, varying the boundary each step.
    sizes = itertools.cycle((1, 3, 7, 100, 1, 333, 2, 1000))
    i = 0
    while i < len(data):
        s = next(sizes)
        out += rs.push(data[i : i + s])
        i += s
    out += rs.flush()
    assert bytes(out) == whole


def test_resample_preserves_low_freq_signal():
    """A low-frequency sine (well below both Nyquists) resamples to the same
    sine within a few LSBs — confirms interpolation is sane, not just length."""
    n_in = 24_000
    data = _sine(n_in, 24_000, freq=200.0)
    out = resample_clip(data, 24_000, 16_000)
    expected = _sine(16_000, 16_000, freq=200.0)
    a = np.frombuffer(out, dtype="<i2").astype(np.float32)
    b = np.frombuffer(expected, dtype="<i2").astype(np.float32)
    # ignore the very edges (interp endpoints differ by a sample or two)
    core = np.abs(a[10:-10] - b[10:-10])
    assert float(np.max(core)) < 80.0  # < ~0.25% full-scale error over the body


def test_small_chunks_dont_lose_data():
    """One-byte pushes (worst case) still reconstruct the full-length output."""
    data = _sine(4800, 24_000)  # 0.2s
    whole = resample_clip(data, 24_000, 16_000)
    rs = StreamingResampler(24_000, 16_000)
    out = bytearray()
    for byte in data:
        out += rs.push(bytes([byte]))
    out += rs.flush()
    assert bytes(out) == whole


def test_empty_push_returns_empty():
    rs = StreamingResampler(24_000, 16_000)
    assert rs.push(b"") == b""
    assert rs.flush() == b""
