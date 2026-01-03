"""Unit tests for the lambda term generator."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.generator import generate_shard


@pytest.mark.unit
class TestGenerator:
    """Test suite for lambda term generator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_generate_shard_creates_file(self) -> None:
        """Test that generate_shard creates a parquet file."""
        shard_id = 0
        count = 100
        
        result_path = generate_shard(shard_id, count, self.output_dir)
        
        assert result_path.exists()
        assert result_path.suffix == ".parquet"
        assert result_path.name == "uss_shard_0000.parquet"

    def test_generate_shard_correct_row_count(self) -> None:
        """Test that generated shard has correct number of rows."""
        shard_id = 1
        count = 250
        
        result_path = generate_shard(shard_id, count, self.output_dir)
        df = pd.read_parquet(result_path)
        
        assert len(df) == count

    def test_generate_shard_has_required_columns(self) -> None:
        """Test that generated shard has all required columns."""
        shard_id = 2
        count = 50
        
        result_path = generate_shard(shard_id, count, self.output_dir)
        df = pd.read_parquet(result_path)
        
        required_columns = ["id", "specification", "term", "complexity", "type_valid"]
        for col in required_columns:
            assert col in df.columns

    def test_generate_shard_unique_ids(self) -> None:
        """Test that generated terms have unique IDs."""
        shard_id = 3
        count = 100
        
        result_path = generate_shard(shard_id, count, self.output_dir)
        df = pd.read_parquet(result_path)
        
        assert df["id"].nunique() == count
        assert not df["id"].duplicated().any()

    def test_generate_shard_complexity_range(self) -> None:
        """Test that complexity values are in expected range."""
        shard_id = 4
        count = 100
        
        result_path = generate_shard(shard_id, count, self.output_dir)
        df = pd.read_parquet(result_path)
        
        assert df["complexity"].min() >= 1
        assert df["complexity"].max() < 50

    def test_generate_shard_type_valid_all_true(self) -> None:
        """Test that all generated terms are marked as type valid."""
        shard_id = 5
        count = 100
        
        result_path = generate_shard(shard_id, count, self.output_dir)
        df = pd.read_parquet(result_path)
        
        assert df["type_valid"].all()

    def test_generate_shard_different_shards_different_content(self) -> None:
        """Test that different shard IDs produce different content."""
        count = 50
        
        path1 = generate_shard(0, count, self.output_dir)
        path2 = generate_shard(1, count, self.output_dir)
        
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)
        
        # IDs should be different due to shard_id in the ID generation
        assert not df1["id"].equals(df2["id"])
