from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IndicSent-M inference with a trained checkpoint.")
    parser.add_argument("--mode", choices=["text", "image", "multimodal"], required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print("Inference CLI parsed successfully.")
    print("Load the requested model class and checkpoint matching --mode before production use.")
    print(vars(args))


if __name__ == "__main__":
    main()
