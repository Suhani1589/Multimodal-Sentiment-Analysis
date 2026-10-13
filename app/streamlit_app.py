from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.explainability.image_explain import gradcam_unavailable_message
from src.explainability.text_explain import simple_token_importance


MAX_UPLOAD_MB = 10


def main() -> None:
    st.set_page_config(page_title="IndicSent-M", layout="wide")
    st.title("IndicSent-M")
    st.caption("Multimodal sentiment analysis for Indian-language social media and meme content.")

    mode = st.radio("Analysis mode", ["Text only", "Image only", "Multimodal"], horizontal=True)
    text = st.text_area("Text", placeholder="बहुत शानदार फिल्म है 😂")
    upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])

    image = None
    if upload is not None:
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"Uploaded image is larger than {MAX_UPLOAD_MB} MB.")
        else:
            try:
                image = Image.open(upload).convert("RGB")
                st.image(image, caption="Uploaded image", use_container_width=True)
            except (UnidentifiedImageError, OSError):
                st.error("Invalid image file.")

    if st.button("Analyze", type="primary"):
        st.info("Connect a trained checkpoint to enable real predictions. The interface is ready for inference output.")
        st.subheader("Prediction")
        st.write({"predicted_sentiment": "not_available", "confidence": None})
        if mode in {"Text only", "Multimodal"}:
            st.subheader("Text explanation")
            st.dataframe(simple_token_importance(text))
        if mode in {"Image only", "Multimodal"}:
            st.subheader("Image explanation")
            st.write(gradcam_unavailable_message())

    with st.expander("About the Model"):
        st.write(
            "IndicSent-M compares text-only multilingual Transformer classification, "
            "image-only ResNet classification, and text-image fusion for positive, neutral, "
            "and negative sentiment. Use the same train/validation/test split for all three models."
        )


if __name__ == "__main__":
    main()
