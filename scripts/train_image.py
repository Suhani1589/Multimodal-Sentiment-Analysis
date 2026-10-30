from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.experiment import training_cli


if __name__ == "__main__":
    training_cli("image")
