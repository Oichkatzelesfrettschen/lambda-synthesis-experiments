"""Unit tests for USS pipeline configuration and setup."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.experiments.uss_pipeline import (
    NeuralLambdaModel,
    ShardedDataset,
    USSConfig,
    auto_gpu_adjust,
)


@pytest.mark.unit
class TestUSSConfig:
    """Test suite for USS configuration."""

    def test_config_defaults(self) -> None:
        """Test that configuration has sensible defaults."""
        assert USSConfig.MODEL_DIM == 768
        assert USSConfig.NUM_HEADS == 12
        assert isinstance(USSConfig.LOG_DIR, Path)
        assert isinstance(USSConfig.SHARD_DIR, Path)

    def test_config_device_type(self) -> None:
        """Test that device is either cuda or cpu."""
        assert USSConfig.DEVICE in ["cuda", "cpu"]


@pytest.mark.unit
class TestAutoGPUAdjust:
    """Test suite for GPU auto-adjustment."""

    def setup_method(self) -> None:
        """Save original config values before each test."""
        self.original_device = USSConfig.DEVICE
        self.original_batch_size = USSConfig.BATCH_SIZE

    def teardown_method(self) -> None:
        """Restore original config values after each test."""
        USSConfig.DEVICE = self.original_device
        USSConfig.BATCH_SIZE = self.original_batch_size

    @patch("torch.cuda.is_available")
    def test_cpu_fallback(self, mock_cuda_available: Mock) -> None:
        """Test that CPU fallback reduces batch size."""
        mock_cuda_available.return_value = False
        USSConfig.DEVICE = "cpu"

        auto_gpu_adjust()

        assert USSConfig.BATCH_SIZE == 64

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_small_vram_adjustment(self, mock_props: Mock, mock_cuda_available: Mock) -> None:
        """Test batch size adjustment for GPUs with limited VRAM."""
        mock_cuda_available.return_value = True
        USSConfig.DEVICE = "cuda"

        # Mock a GPU with 12GB VRAM
        mock_device_props = Mock()
        mock_device_props.total_memory = 12 * 1024**3
        mock_props.return_value = mock_device_props

        auto_gpu_adjust()

        assert USSConfig.BATCH_SIZE == 512


@pytest.mark.unit
class TestShardedDataset:
    """Test suite for ShardedDataset."""

    def test_dataset_length(self, tmp_path: Path) -> None:
        """Test that dataset reports correct length."""
        dataset = ShardedDataset(tmp_path)
        assert len(dataset) == 10_000_000

    def test_dataset_getitem_shape(self, tmp_path: Path) -> None:
        """Test that __getitem__ returns correct tensor shapes."""
        dataset = ShardedDataset(tmp_path)
        x, y = dataset[0]

        assert x.shape == (USSConfig.MODEL_DIM,)
        assert y.shape == (1,)

    def test_dataset_getitem_types(self, tmp_path: Path) -> None:
        """Test that __getitem__ returns correct tensor types."""
        dataset = ShardedDataset(tmp_path)
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)


@pytest.mark.unit
class TestNeuralLambdaModel:
    """Test suite for NeuralLambdaModel."""

    def test_model_initialization(self) -> None:
        """Test that model initializes correctly."""
        model = NeuralLambdaModel(USSConfig)

        assert hasattr(model, "encoder")
        assert hasattr(model, "transformer")
        assert hasattr(model, "head")

    def test_model_forward_shape(self) -> None:
        """Test that forward pass produces correct output shape."""
        model = NeuralLambdaModel(USSConfig)
        batch_size = 4
        seq_len = 32

        x = torch.randn(batch_size, seq_len, USSConfig.MODEL_DIM)
        output = model(x)

        assert output.shape == (batch_size, seq_len, 100)

    def test_model_forward_no_nan(self) -> None:
        """Test that forward pass doesn't produce NaN values."""
        model = NeuralLambdaModel(USSConfig)
        model.eval()

        x = torch.randn(2, 16, USSConfig.MODEL_DIM)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda_transfer(self) -> None:
        """Test that model can be transferred to CUDA."""
        model = NeuralLambdaModel(USSConfig)
        model = model.to("cuda")

        x = torch.randn(2, 16, USSConfig.MODEL_DIM).to("cuda")
        output = model(x)

        assert output.device.type == "cuda"
