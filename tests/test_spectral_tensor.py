"""Tests for SpectralTensor dataclass."""

import pytest
import torch

from bifrost.spectral_tensor import SpectralTensor


@pytest.fixture
def sample_st():
    """Create a small SpectralTensor for testing."""
    n_freq = 33  # e.g. from n_fft=64
    return SpectralTensor(
        amplitude=torch.rand(2, n_freq),
        phase=torch.rand(2, n_freq) * 2 * 3.14159 - 3.14159,
        scale=torch.linspace(0, 8000, n_freq).unsqueeze(0).expand(2, -1),
        uncertainty=torch.ones(2, n_freq),
        metadata={"sample_rate": 16000},
    )


class TestSpectralTensor:
    def test_shape_property(self, sample_st):
        assert sample_st.shape == torch.Size([2, 33])

    def test_num_bands(self, sample_st):
        assert sample_st.num_bands == 33

    def test_device(self, sample_st):
        assert sample_st.device == torch.device("cpu")

    def test_validate_passes(self, sample_st):
        sample_st.validate()  # should not raise

    def test_validate_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            SpectralTensor(
                amplitude=torch.rand(2, 10),
                phase=torch.rand(2, 5),  # wrong shape
                scale=torch.rand(2, 10),
                uncertainty=torch.rand(2, 10),
            ).validate()

    def test_complex_spectrum(self, sample_st):
        cs = sample_st.complex_spectrum()
        assert cs.is_complex()
        assert cs.shape == sample_st.shape

    def test_energy(self, sample_st):
        e = sample_st.energy()
        assert e.item() > 0

    def test_to_device(self, sample_st):
        moved = sample_st.to("cpu")
        assert moved.device == torch.device("cpu")

    def test_detach(self, sample_st):
        d = sample_st.detach()
        assert not d.amplitude.requires_grad

    def test_repr(self, sample_st):
        r = repr(sample_st)
        assert "SpectralTensor" in r
        assert "bands=33" in r
