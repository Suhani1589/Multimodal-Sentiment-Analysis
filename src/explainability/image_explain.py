from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.explainability.attribution import grad_cam


def explain_image_prediction(model, image, **kwargs):
    """Return a Grad-CAM map from the final ResNet block."""
    return grad_cam(model, image, **kwargs)


def gradcam_unavailable_message(image_path: Optional[Union[str, Path]] = None) -> str:
    """Explain the current Grad-CAM limitation honestly."""
    suffix = f" for {image_path}" if image_path else ""
    return f"Grad-CAM requires a trained CNN checkpoint and target layer selection{suffix}."
