from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference.checkpoint import load_checkpoint_predictor, predict_batch, prepare_prediction_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment from text, image, or paired inputs.")
    parser.add_argument("--mode", choices=["text", "image", "multimodal"], required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    image = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    predictor = load_checkpoint_predictor(
        args.mode, args.checkpoint, fallback_config_path=args.config
    )
    batch = prepare_prediction_batch(predictor, text=args.text, image=image)
    print(json.dumps(predict_batch(predictor, batch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
