# IndicSent-M: Multimodal Sentiment Analysis for Indian Languages

IndicSent-M is a research-oriented, beginner-readable project for comparing sentiment classification from text, images, and fused text-image inputs. The target task is three-class sentiment classification: `positive`, `neutral`, and `negative`.

## Research Question

Does combining textual and visual information improve sentiment classification of Indian-language social media or meme content compared with text-only and image-only models?

The project is designed to answer this only from measured results. It writes per-model metrics and a final comparison table; it does not claim multimodal improvement unless macro F1 improves on the same test split.

## Architecture

- Text-only: multilingual Transformer encoder such as IndicBERT or XLM-R, pooled representation, dropout, linear classifier.
- Image-only: ImageNet-pretrained ResNet-50, feature vector, dropout, classifier.
- Multimodal: text encoder plus image encoder, projection layers, concatenation fusion, MLP, classifier.

Important settings live in `configs/config.yaml`.

## Dataset Format

This repository does not fabricate training data or download data automatically. Put your paired dataset at:

```text
data/raw/metadata.csv
data/raw/images/
```

CSV columns:

```csv
sample_id,image_path,text,language,sentiment
```

Example schema:

```csv
001,images/001.jpg,"यह फिल्म बहुत अच्छी है",hi,positive
002,images/002.jpg,"This is so bad yaar",hinglish,negative
```

Allowed sentiment labels are `positive`, `neutral`, and `negative`.

## Dataset Selection

The recommended dataset for the main experiment is Memotion 3, because it provides paired meme images and embedded text for Hindi-English / Hinglish memes, with Task A sentiment labels matching the project target classes. Access requires using the official shared-task source and following its registration/access terms; the project does not automatically download it.

After manual download, adapt the official annotation files with:

```bash
python scripts/adapt_memotion.py \
  --annotations data/raw/memotion3/annotations/train.csv data/raw/memotion3/annotations/val.csv data/raw/memotion3/annotations/test.csv \
  --splits train val test \
  --image-prefix images \
  --language hinglish \
  --output data/raw/metadata.csv
```

See `data/README.md` for the expected folder structure, label mapping, and validation report details.

## Candidate Datasets

Candidate datasets found during setup:

| Dataset | Coverage | Paired text/image | Labels | Suitability |
| --- | --- | --- | --- | --- |
| Memotion / SemEval-2020 Task 8 | Internet memes, largely English/Hinglish-style social media content | Yes | Sentiment plus emotion/humour/offense-style labels | Good general baseline if access/license terms are satisfied |
| Memotion 3 | Code-mixed Hindi-English memes | Yes | Sentiment, emotion, intensity | Strong fit for Hinglish sentiment research, but confirm access and license |
| Dravidian multimodal troll meme shared task | Tamil/Malayalam meme content | Yes | Task-specific sentiment/troll labels | Useful for Dravidian-language extension if the released labels match this 3-class setup |
| HinglishMemeX | Indian Hinglish memes | Yes | Veracity-style labels | Not directly sentiment-labeled, so not a drop-in dataset |
| IMUSA Punjabi meme task | Punjabi memes | Yes | Sentiment-focused shared task | Potentially relevant, but availability and license must be checked |

Use only datasets whose access terms permit your intended academic use.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

```bash
python scripts/prepare_data.py
```

This validates required columns, label values, image existence/readability, normalizes text, creates train/validation/test splits, and saves:

```text
data/processed/splits.csv
results/plots/class_distribution.png
```

## Training

Run all models on the same prepared split:

```bash
python scripts/train_text.py
python scripts/train_image.py
python scripts/train_multimodal.py
```

The trainer supports epochs, optimizer, learning rate, weight decay, validation, early stopping, checkpoint saving, best-model selection, and training history.

## Evaluation

Model-specific test evaluation can be attached after training checkpoints are selected. The comparison utility expects:

```text
results/metrics/text_results.json
results/metrics/image_results.json
results/metrics/multimodal_results.json
```

Then run:

```bash
python scripts/evaluate.py
```

It writes:

```text
results/metrics/model_comparison.csv
```

Main metrics are accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrix, and per-class metrics.

## Inference

The current CLI validates arguments and is ready to connect to trained checkpoints:

```bash
python scripts/predict.py --mode multimodal --text "बहुत शानदार फिल्म है 😂" --image data/raw/images/example.jpg --checkpoint checkpoints/multimodal_best.pt
```

Predictions should be reported as probabilities, not certainty.

## Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

The app accepts text, image upload, and mode selection. It includes honest placeholder messaging until a trained checkpoint is connected.

## Explainability

The project includes a simple text-token explanation fallback and a Grad-CAM integration point for CNN checkpoints. Do not present raw Transformer attention as automatically faithful explanation. For final reporting, generate qualitative examples with input text, image, true label, predicted label, confidence, important text features, and visual regions.

## Reproducibility

- Random seed is configured in `configs/config.yaml`.
- Device selection supports MPS, CUDA, and CPU.
- Dataset splits are saved and reused across text-only, image-only, and multimodal experiments.
- Results are saved in machine-readable JSON/CSV files.

## Limitations

Indian-language multimodal sentiment datasets are limited. Code-mixing, sarcasm, cultural context, meme templates, OCR noise, class imbalance, dataset bias, pretrained model bias, compute limits, image-text contradiction, and imperfect explainability can all affect conclusions. Classification performance does not mean the model truly understands culture or intent.

## Tests

```bash
pytest
```

The tests cover dataset validation, missing-image handling, text preprocessing, metrics, and text/image/multimodal output dimensions using dummy encoders.
