"""Unit tests for the Triton tensor contraction kernel."""

import sys
from pathlib import Path

import importlib
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Check if Triton is available without importing it directly
    if importlib.util.find_spec("triton") is not None:
        from src.kernels.tensor_contraction import uss_tensor_contract
        TRITON_AVAILABLE = True
    else:
        TRITON_AVAILABLE = False
except ImportError:
    TRITON_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.gpu
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTensorContraction:
    """Test suite for Triton tensor contraction kernel."""

    def test_basic_matrix_multiply(self) -> None:
        """Test basic matrix multiplication correctness."""
        M, K, N = 128, 64, 128

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Our custom kernel result
        c_custom = uss_tensor_contract(a, b)

        # PyTorch reference result
        c_reference = torch.matmul(a, b)

        # Check shapes match
        assert c_custom.shape == c_reference.shape

        # Check numerical correctness (allow for some floating point error)
        torch.testing.assert_close(c_custom, c_reference, rtol=1e-2, atol=1e-2)

    def test_shape_validation(self) -> None:
        """Test that incompatible dimensions raise an error."""
        a = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # Incompatible K dimension

        with pytest.raises(AssertionError, match="Incompatible dimensions"):
            uss_tensor_contract(a, b)

    def test_different_sizes(self) -> None:
        """Test correctness with different matrix sizes."""
        test_sizes = [
            (64, 32, 64),
            (256, 128, 256),
            (512, 256, 512),
        ]

        for M, K, N in test_sizes:
            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)

            c_custom = uss_tensor_contract(a, b)
            c_reference = torch.matmul(a, b)

            torch.testing.assert_close(c_custom, c_reference, rtol=1e-2, atol=1e-2)

    def test_output_dtype(self) -> None:
        """Test that output dtype is float16."""
        M, K, N = 128, 64, 128

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = uss_tensor_contract(a, b)

        assert c.dtype == torch.float16

    def test_deterministic_output(self) -> None:
        """Test that same inputs produce same outputs."""
        M, K, N = 128, 64, 128

        torch.manual_seed(42)
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c1 = uss_tensor_contract(a, b)
        c2 = uss_tensor_contract(a, b)

        torch.testing.assert_close(c1, c2)


@pytest.mark.unit
class TestTensorContractionCPU:
    """CPU-only tests that don't require GPU."""

    def test_import_module(self) -> None:
        """Test that the module can be imported."""
        from src.kernels import tensor_contraction

        assert hasattr(tensor_contraction, "uss_tensor_contract")
