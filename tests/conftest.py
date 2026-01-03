"""Conftest for pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture
def sample_lambda_term() -> str:
    """Provide a sample lambda term for testing."""
    return "(λ x. x)"


@pytest.fixture
def sample_complex_term() -> str:
    """Provide a complex lambda term for testing."""
    return "(λ f. (λ x. f (f x)))"
