"""Migrate an existing legacy dense Qdrant collection to native sparse hybrid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import migrate_legacy_index_to_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the native sparse sidecar for existing Qdrant points.")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    migrated = migrate_legacy_index_to_hybrid(batch_size=args.batch_size)
    print(f"Migrated {migrated} points to the native sparse hybrid collection.")


if __name__ == "__main__":
    main()
