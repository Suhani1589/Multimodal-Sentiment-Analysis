from __future__ import annotations

import torch
from torch import nn

from src.multimodal.model import MultimodalSentimentModel
from src.text.model import TextSentimentModel
from src.vision.model import ImageSentimentModel


class DummyTextEncoder(nn.Module):
    hidden_size = 5

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.ones((input_ids.shape[0], self.hidden_size))


class DummyImageEncoder(nn.Module):
    output_dim = 7

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ones((image.shape[0], self.output_dim))


def test_text_model_output_shape() -> None:
    model = TextSentimentModel("dummy", encoder=DummyTextEncoder(), encoder_dim=5)
    logits = model(input_ids=torch.ones((2, 4), dtype=torch.long), attention_mask=torch.ones((2, 4), dtype=torch.long))
    assert logits.shape == (2, 3)


def test_image_model_output_shape() -> None:
    model = ImageSentimentModel(encoder=DummyImageEncoder(), encoder_dim=7)
    logits = model(image=torch.randn(2, 3, 16, 16))
    assert logits.shape == (2, 3)


def test_multimodal_model_output_shape() -> None:
    model = MultimodalSentimentModel(
        "dummy",
        text_encoder=DummyTextEncoder(),
        image_encoder=DummyImageEncoder(),
        text_dim=5,
        image_dim=7,
        projection_dim=4,
        hidden_dim=6,
    )
    logits = model(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        image=torch.randn(2, 3, 16, 16),
    )
    assert logits.shape == (2, 3)
