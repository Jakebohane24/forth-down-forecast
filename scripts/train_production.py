"""Train and save the model used for future weekly predictions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import BettingConfig
from src.config import ModelConfig
from src.training import NFLModel


PRODUCTION_SEASONS = tuple(range(2018, 2026))
MODEL_VERSION = "production-2018-2025-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/production"),
    )
    args = parser.parse_args()

    config = ModelConfig(
        training_seasons=PRODUCTION_SEASONS,
        # These future masks remain empty by design. Production training uses
        # all completed data and is evaluated prospectively.
        validation_season=2026,
        test_season=2027,
    )
    model = NFLModel(config).train()
    model.save(args.output, model_version=MODEL_VERSION)
    manifest = {
        "model_version": MODEL_VERSION,
        "status": "production",
        "training_seasons": list(PRODUCTION_SEASONS),
        "moneyline_confidence_threshold": (
            BettingConfig().moneyline_confidence_threshold
        ),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = Path("reports/production_model.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Saved production model trained on {PRODUCTION_SEASONS[0]}–"
        f"{PRODUCTION_SEASONS[-1]} to {args.output}"
    )


if __name__ == "__main__":
    main()
