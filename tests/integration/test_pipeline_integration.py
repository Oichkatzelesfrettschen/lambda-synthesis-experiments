"""Integration tests for the complete USS pipeline."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Integration tests for complete data generation and processing pipeline."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_generate_and_load_data(self) -> None:
        """Test that generated data can be loaded and processed."""
        import pandas as pd

        from src.data.generator import generate_shard

        # Generate a small shard
        shard_path = generate_shard(0, 1000, self.output_dir)

        # Load and verify
        df = pd.read_parquet(shard_path)
        assert len(df) == 1000
        assert "term" in df.columns
        assert "complexity" in df.columns

    @pytest.mark.skipif(sys.platform == "darwin", reason="Skip on macOS due to multiprocessing")
    def test_parallel_shard_generation(self) -> None:
        """Test parallel generation of multiple shards."""
        from concurrent.futures import ProcessPoolExecutor

        import pandas as pd

        from src.data.generator import generate_shard

        num_shards = 4
        shard_size = 500

        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(generate_shard, i, shard_size, self.output_dir)
                for i in range(num_shards)
            ]

            paths = [future.result() for future in futures]

        # Verify all shards were created
        assert len(paths) == num_shards

        # Verify data integrity
        total_rows = 0
        all_ids = set()

        for path in paths:
            df = pd.read_parquet(path)
            total_rows += len(df)
            all_ids.update(df["id"].tolist())

        assert total_rows == num_shards * shard_size
        assert len(all_ids) == total_rows  # All IDs unique across shards


@pytest.mark.integration
class TestModelDatasetIntegration:
    """Integration tests for model and dataset interaction."""

    def test_dataset_model_compatibility(self, tmp_path: Path) -> None:
        """Test that dataset output is compatible with model input."""
        import torch

        from src.experiments.uss_pipeline import NeuralLambdaModel, ShardedDataset, USSConfig

        dataset = ShardedDataset(tmp_path)
        model = NeuralLambdaModel(USSConfig)

        # Get a batch of data
        batch_size = 4
        seq_len = 16
        batch = torch.stack([dataset[i][0] for i in range(batch_size)])
        batch = batch.unsqueeze(1).expand(-1, seq_len, -1)

        # Forward pass should work
        output = model(batch)
        assert output.shape[0] == batch_size
