from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import colormaps
from PIL import Image, UnidentifiedImageError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.explainability.attribution import grad_cam, token_gradient_attribution
from src.inference.checkpoint import (
    LoadedPredictor,
    load_checkpoint_predictor,
    predict_batch,
    prepare_prediction_batch,
)


MAX_UPLOAD_MB = 10
DEFAULT_CHECKPOINTS = {
    "text": Path("checkpoints/text-baseline/text_best.pt"),
    "image": Path("checkpoints/image-subset-baseline/image_best.pt"),
    "multimodal": Path("checkpoints/multimodal-subset-baseline/multimodal_best.pt"),
}
EVALUATION_FILES = {
    "text": Path("results/text-baseline-evaluation/metrics/text_results.json"),
    "image": Path("results/image-subset-evaluation/metrics/image_results.json"),
    "multimodal": Path("results/multimodal-subset-evaluation/metrics/multimodal_results.json"),
}
RUN_FILES = {
    "text": Path("checkpoints/text-baseline/text_run.json"),
    "image": Path("checkpoints/image-subset-baseline/image_run.json"),
    "multimodal": Path("checkpoints/multimodal-subset-baseline/multimodal_run.json"),
}


@st.cache_resource(show_spinner="Loading checkpoint…")
def get_predictor(model_type: str, checkpoint_path: str) -> LoadedPredictor:
    return load_checkpoint_predictor(model_type, checkpoint_path)


def _load_json(path: Path) -> dict:
    import json

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _dataset_summary() -> tuple[pd.DataFrame, dict]:
    split_path = Path("data/processed/splits.csv")
    if not split_path.is_file():
        return pd.DataFrame(), {"status": "pending", "message": "Prepared split file is missing."}
    frame = pd.read_csv(split_path)
    summary = pd.crosstab(frame["split"], frame["sentiment"]).reindex(
        columns=["positive", "neutral", "negative"], fill_value=0
    )
    return summary, {
        "status": "available",
        "samples": len(frame),
        "missing_text": int(frame["text"].isna().sum()),
        "split_counts": frame["split"].value_counts().to_dict(),
    }


def _comparison_frame() -> pd.DataFrame:
    rows = []
    for name in ("text", "image", "multimodal"):
        result = _load_json(EVALUATION_FILES[name])
        run = _load_json(RUN_FILES[name])
        rows.append(
            {
                "model": {"text": "Text only", "image": "Image only", "multimodal": "Multimodal"}[name],
                "status": result.get("status", "pending"),
                "train samples": run.get("training_sample_count"),
                "val samples": run.get("validation_sample_count"),
                "test samples": result.get("test_samples_evaluated"),
                "accuracy": result.get("accuracy"),
                "macro precision": result.get("precision_macro"),
                "macro recall": result.get("recall_macro"),
                "macro F1": result.get("f1_macro"),
                "weighted F1": result.get("f1_weighted"),
            }
        )
    return pd.DataFrame(rows)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _display_explanations(predictor: LoadedPredictor, batch: dict, image: Optional[Image.Image]) -> None:
    tensors = _move_batch(batch, predictor.device)
    if predictor.model_type in {"text", "multimodal"}:
        st.markdown("**Text attribution** · gradient × embedding")
        try:
            scores = token_gradient_attribution(
                predictor.model,
                tensors["input_ids"],
                tensors["attention_mask"],
                image=tensors.get("image"),
            )["scores"]
            token_ids = tensors["input_ids"][0].detach().cpu().tolist()
            mask = tensors["attention_mask"][0].detach().cpu().tolist()
            tokens = predictor.tokenizer.convert_ids_to_tokens(token_ids)
            special_tokens = set(getattr(predictor.tokenizer, "all_special_tokens", []))
            attribution = [
                {"token": token, "relative attribution": round(float(score), 4)}
                for token, score, keep in zip(tokens, scores, mask)
                if keep and token not in special_tokens
            ]
            st.dataframe(pd.DataFrame(attribution), width="stretch", hide_index=True)
        except (RuntimeError, ValueError, AttributeError) as exc:
            st.warning(f"Text attribution is unavailable for this checkpoint: {exc}")

    if predictor.model_type in {"image", "multimodal"} and image is not None:
        st.markdown("**Image attribution** · Grad-CAM")
        try:
            explanation = grad_cam(
                predictor.model,
                tensors["image"],
                input_ids=tensors.get("input_ids"),
                attention_mask=tensors.get("attention_mask"),
            )
            heatmap = explanation["heatmap"].numpy()
            colors = (colormaps["magma"](heatmap)[..., :3] * 255).astype(np.uint8)
            base = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0])))
            overlay = (0.55 * base + 0.45 * colors).clip(0, 255).astype(np.uint8)
            st.image(overlay, caption="Grad-CAM overlay", width="stretch")
        except (RuntimeError, ValueError, AttributeError) as exc:
            st.warning(f"Image attribution is unavailable for this checkpoint: {exc}")
    st.caption(
        "Attribution highlights input regions that locally affect this model's score. It is not a "
        "human explanation, proof of correctness, or evidence of causal reasoning."
    )


def main() -> None:
    st.set_page_config(page_title="IndicSent-M | Research demo", page_icon="💬", layout="wide")
    st.title("IndicSent-M")
    st.caption("Experimental sentiment classification for Hinglish meme text and images")
    st.warning(
        "Research prototype: predictions can be wrong. The available models are early baselines and "
        "currently miss most neutral and negative examples."
    )

    with st.sidebar:
        st.header("Analyze a post")
        display_mode = st.radio("Model", ["Text only", "Image only", "Multimodal"])
        mode = {"Text only": "text", "Image only": "image", "Multimodal": "multimodal"}[display_mode]
        checkpoint = DEFAULT_CHECKPOINTS[mode]
        if checkpoint.is_file():
            st.success(f"Checkpoint ready: `{checkpoint}`")
        else:
            st.error(f"Checkpoint missing: `{checkpoint}`")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        text = st.text_area("Post text", placeholder="बहुत शानदार फिल्म है 😂", height=150)
        upload = st.file_uploader("Meme image", type=["png", "jpg", "jpeg", "webp"])
        image = None
        if upload is not None:
            if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Image is larger than the {MAX_UPLOAD_MB} MB limit.")
            else:
                try:
                    image = Image.open(upload).convert("RGB")
                    st.image(image, caption="Uploaded image", width="stretch")
                except (UnidentifiedImageError, OSError):
                    st.error("The uploaded file is not a readable image.")
        explain = st.checkbox("Show optional input attribution", value=False)
        text_ready = mode == "image" or bool(text.strip())
        image_ready = mode == "text" or image is not None
        if st.button("Analyze", type="primary", disabled=not (text_ready and image_ready and checkpoint.is_file())):
            try:
                predictor = get_predictor(mode, str(checkpoint))
                batch = prepare_prediction_batch(predictor, text=text, image=image)
                prediction = predict_batch(predictor, batch)
                st.subheader(f"Predicted: {prediction['predicted_sentiment'].title()}")
                st.metric("Model confidence", f"{prediction['confidence']:.1%}")
                st.progress(prediction["confidence"])
                probabilities = pd.DataFrame(
                    [{"sentiment": label.title(), "probability": probability}
                     for label, probability in prediction["probabilities"].items()]
                )
                st.dataframe(probabilities, width="stretch", hide_index=True)
                if explain:
                    _display_explanations(predictor, batch, image)
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")

    with right:
        st.subheader("Current held-out results")
        st.caption("These are recorded experiment results, not a live benchmark. Training scopes differ.")
        st.dataframe(_comparison_frame(), width="stretch", hide_index=True)
        st.subheader("Dataset split summary")
        split_table, info = _dataset_summary()
        if info["status"] == "available":
            st.caption(f"{info['samples']:,} samples · blank text rows: {info['missing_text']}")
            st.dataframe(split_table, width="stretch")
            st.caption("Current data is primarily Hinglish; it does not establish multilingual coverage.")
        else:
            st.info(info["message"])

    with st.expander("How to read the explanations"):
        st.write(
            "Text scores use gradient × embedding token saliency. Image overlays use Grad-CAM on the "
            "final ResNet block. For multimodal inference, each attribution is computed while holding "
            "the other input fixed. These are model attribution visualizations, not verified reasons."
        )


if __name__ == "__main__":
    main()
