"""Lambda term generator for USS data pipeline."""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def generate_shard(shard_id: int, count: int, output_dir: Path) -> Path:
    """
    Generates a single shard of synthetic lambda terms.
    """
    print(f"Starting shard {shard_id} ({count} terms)...")

    # Logic for generating "valid-looking" terms
    # In a real scenario, this would involve a recursive term generator
    data = {
        "id": [f"term_{shard_id}_{i}" for i in range(count)],
        "specification": [f"spec_type_{np.random.randint(0, 100)}" for _ in range(count)],
        "term": [f"(\u03bb x. x) term_{i}" for i in range(count)],  # Simplified for baseline
        "complexity": np.random.randint(1, 50, size=count),
        "type_valid": [True] * count,
    }

    df = pd.DataFrame(data)
    shard_path = output_dir / f"uss_shard_{shard_id:04d}.parquet"
    df.to_parquet(shard_path, engine="pyarrow", compression="snappy")
    return shard_path


def main() -> None:
    TOTAL_TERMS = 10_000_000
    SHARD_SIZE = 250_000
    NUM_SHARDS = TOTAL_TERMS // SHARD_SIZE
    OUTPUT_DIR = Path("src/data/shards")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scaling to {TOTAL_TERMS} terms across {NUM_SHARDS} shards...")
    start_time = time.time()

    # Use CPU count for parallel generation
    num_workers = mp.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(generate_shard, i, SHARD_SIZE, OUTPUT_DIR) for i in range(NUM_SHARDS)
        ]

        for future in futures:
            path = future.result()
            # print(f"Completed: {path.name}")

    end_time = time.time()
    print(f"Total Generation Time: {end_time - start_time:.2f} seconds")
    print(f"Data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
