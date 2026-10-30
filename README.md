# IndicSent-M

IndicSent-M compares sentiment prediction from meme text, images, and paired text-image inputs. The current data is a **Hinglish** Memotion-derived dataset; this repository does not establish coverage across Indian languages. The task labels are `positive`, `neutral`, and `negative`.

This is an experimental research prototype. Its present one-epoch/subset checkpoints collapse mostly to the positive class and should not be treated as useful or production-ready classifiers.

## Research question

Does combining text and image features improve three-class sentiment classification on the same held-out meme test split, compared with text-only and image-only models? Answer this from validation-selected checkpoints and measured test results; do not tune on the test split.

## Pipeline

```text
Memotion annotations + images
        │ adapt and validate
        ▼
metadata.csv ── deterministic grouped split ── train / validation / test
        │                                           │
        ├── Text: XLM-R/IndicBERT ─ classifier      ├── train: optimize
        ├── Image: ResNet-50 ────── classifier      ├── validation: select
        └── Text + image ─ projection/concat/MLP    └── test: final report only
```

The multimodal model projects each encoder representation, concatenates them, then uses an MLP classifier. Image transforms use ImageNet normalization; training adds random resized crops and horizontal flips. Text is normalized with the same URL, username, and casing settings in training and inference.

## Dataset and access

The included preparation workflow expects manually obtained data at `data/raw/metadata.csv`, with columns `sample_id,image_path,text,language,sentiment`. Images are referenced relative to `data/raw/` unless an absolute path is provided. `data/README.md` documents the Memotion 3 adapter, source links, expected files, label mapping, and access caveats. Follow the source dataset's current registration, license, and redistribution terms; do not commit the data or images.

The checked split file contains 6,991 rows, all marked `hinglish`: 4,898 train, 1,046 validation, and 1,047 test. It has 4 rows without usable text and no repeated sample IDs. Exact duplicate image content does not cross splits. Five normalized text/label combinations recur across splits; these were not removed because they are not necessarily the same sample. Split creation groups by image content to prevent identical images crossing splits. Grouped assignment is deterministic, but does not guarantee stratification; the existing `splits.csv` is preserved for current experiments. Its class counts are:

| Split | Positive | Neutral | Negative |
| --- | ---: | ---: | ---: |
| Train | 2,918 | 1,527 | 453 |
| Validation | 616 | 339 | 91 |
| Test | 625 | 335 | 87 |

Class imbalance is substantial, especially for negative examples. Text-only and multimodal loaders omit rows without text; image-only can retain them. No split or label is silently changed during training.

## Environment

Python 3.9+ is supported by the project code. On macOS, create an environment and install the pinned project requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Choose `project.device` in `configs/config.yaml` as `auto`, `cpu`, `mps`, or `cuda` as appropriate. `auto` prefers available accelerators. The current machine used for the saved experiments was CPU-only with 16 GB physical memory.

## Prepare and validate data

```bash
.venv/bin/python scripts/adapt_memotion.py --help
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/smoke_test_dataset.py
```

Preparation refuses to replace an existing prepared split by default. To make a new split, select a versioned destination and point a separate YAML config's `dataset.prepared_csv` to it:

```bash
.venv/bin/python scripts/prepare_data.py --output-csv data/processed/splits_v2.csv
```

Use `--overwrite` only when intentionally replacing the selected split. Preparation writes validation details to `results/dataset_validation_report.json` and a class plot under `results/plots/`.

## Training

Training uses the configured seed, AdamW, and validation macro-F1 checkpoint selection. The default imbalance strategy is a reproducible weighted sampler on **training rows only**; it draws classes approximately evenly with replacement. Do not combine this with inverse-frequency class-weighted loss by default, since applying both can overcorrect. Set `training.imbalance_strategy` to `class_weights` or `none` for those alternatives. Validation and test distributions are never sampled. Ties retain the earliest epoch. A ReduceLROnPlateau scheduler is configured to reduce the learning rate when validation macro-F1 stops improving. Each epoch records class-wise metrics, confusion matrices, predicted class counts, learning rate, and duration. The first training batch also records representative gradient norms and parameter update magnitudes for the encoder and classifier. Run metadata records original training counts, sampling strategy, best epoch, configuration, and initialization request. A model that cannot load its requested pretrained weights exits with an error rather than being reported as pretrained.

The configured `ai4bharat/indic-bert` model returned an access error in this environment. The available public alternative is XLM-RoBERTa. Download/cache XLM-R before running offline, or allow network access on first use. Image and multimodal models request ImageNet ResNet-50 weights by default; this download is about 100 MB. The saved subset image and multimodal runs used random ResNet initialization, as recorded in their configs.

Before another full text run, the training-only memorization diagnostic can test the data/model/optimizer path on a fixed balanced subset. It does not sample validation or test rows, does not use weighted sampling or weighted loss, and writes JSON without a checkpoint:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_text_overfit.py \
  --samples-per-class 8 --max-steps 120 --batch-size 4 \
  --output-json results/diagnostics/text-overfit-xlmroberta-next.json
```

The seed, actual sample IDs, label/tokenizer previews, step metrics, gradients, and parameter updates are recorded. Use a fresh `--output-json` path for each run.

Example full-split commands (these are not launched automatically):

```bash
.venv/bin/python scripts/train.py --model text --text-model-name xlm-roberta-base --epochs 3 --batch-size 2 --output-dir checkpoints/text-xlmroberta
.venv/bin/python scripts/train.py --model image --epochs 3 --batch-size 4 --output-dir checkpoints/image-resnet50
.venv/bin/python scripts/train.py --model multimodal --text-model-name xlm-roberta-base --epochs 3 --batch-size 2 --output-dir checkpoints/multimodal-xlmroberta
```

On the CPU used for the current run, one full-split XLM-R epoch took about 25 minutes. Three epochs are therefore roughly 75 minutes for text alone. Image training with ResNet-50 and full multimodal fine-tuning take additional time; memory and throughput depend on batch size. Check disk space before downloads and expect roughly 1.1 GB for XLM-R model weights, plus checkpoint copies. Use a unique `--output-dir`; training protects existing artifacts unless `--overwrite` is explicit. Bounded `--max-train-samples` and `--max-val-samples` are intended for pipeline checks, not final comparisons.

## Evaluation and report

Evaluate best checkpoints without training:

```bash
.venv/bin/python scripts/evaluate.py --model all
```

For experiments in separate directories, evaluate each model into its own output directory, for example:

```bash
.venv/bin/python scripts/evaluate.py --model text --checkpoint-dir checkpoints/text-xlmroberta --output-dir results/text-xlmroberta-evaluation
.venv/bin/python scripts/evaluate.py --model image --checkpoint-dir checkpoints/image-resnet50 --output-dir results/image-resnet50-evaluation
.venv/bin/python scripts/evaluate.py --model multimodal --checkpoint-dir checkpoints/multimodal-xlmroberta --output-dir results/multimodal-xlmroberta-evaluation
```

Create a combined report, training curves, held-out confusion matrices, and metric comparison figure from the saved runs:

```bash
.venv/bin/python scripts/build_final_report.py \
  --text-run-dir checkpoints/text-xlmroberta \
  --image-run-dir checkpoints/image-resnet50 \
  --multimodal-run-dir checkpoints/multimodal-xlmroberta \
  --text-evaluation-dir results/text-xlmroberta-evaluation \
  --image-evaluation-dir results/image-resnet50-evaluation \
  --multimodal-evaluation-dir results/multimodal-xlmroberta-evaluation \
  --output-dir results/final
```

Missing checkpoints/evaluations are marked `not_evaluated`; metrics are not filled with zero. Reports include macro and weighted scores, per-class scores, support, confusion matrices, test class counts, and a test-set majority-class reference. Parameter count and inference latency are marked unmeasured unless profiled separately on a stated device. Existing `results/final` files are not overwritten.

## Existing experiment results

The current saved results are not a matched full-data model comparison. Text was trained for one epoch on 4,896 train and 1,044 validation rows. Image-only and multimodal used 128 train and 48 validation rows for one epoch. All were evaluated on the same 1,047 held-out test rows. The text encoder was XLM-RoBERTa; image initialization was random for the two subset vision runs.

A later three-epoch full-split XLM-RoBERTa run is recorded under `checkpoints/text-xlmroberta/`. Its validation macro-F1 was 0.2471 at each epoch and it predicted positive for every validation row; strict macro-F1 checkpoint selection therefore retained epoch 1. Its validation set had 616 positive, 339 neutral, and 91 negative examples. This run has not been evaluated on the held-out test split, so it is not included in the test table below.

| Run | Test accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Prediction behavior |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Text, full split | 0.5969 | 0.1990 | 0.3333 | 0.2492 | 0.4463 | Positive for all 1,047 rows |
| Text, subset | 0.5969 | 0.1990 | 0.3333 | 0.2492 | 0.4463 | Positive for all 1,047 rows |
| Image, subset | 0.5950 | 0.1989 | 0.3323 | 0.2489 | 0.4457 | 1,044 positive, 3 neutral |
| Multimodal, subset | 0.5969 | 0.1990 | 0.3333 | 0.2492 | 0.4463 | Positive for all 1,047 rows |

The test majority-positive baseline accuracy is 0.5969. Thus the present scores do not show useful class discrimination or multimodal benefit. Per-class metrics and confusion matrices are in the corresponding evaluation JSON/CSV files. Existing checkpoint histories and run metadata are retained under `checkpoints/`; held-out reports and per-example prediction CSVs remain under `results/*-evaluation/`.

## Inference and demo

Use a compatible saved checkpoint; no model is trained when predicting or starting the app:

```bash
.venv/bin/python scripts/predict.py --mode text --text "This is a lovely meme" --checkpoint checkpoints/text-baseline/text_best.pt
.venv/bin/python scripts/predict.py --mode image --image data/raw/memotion_dataset_7k/images/example.png --checkpoint checkpoints/image-subset-baseline/image_best.pt
.venv/bin/python scripts/predict.py --mode multimodal --text "बहुत शानदार" --image data/raw/memotion_dataset_7k/images/example.png --checkpoint checkpoints/multimodal-subset-baseline/multimodal_best.pt
streamlit run app/streamlit_app.py
```

The Streamlit demo loads checkpoints on demand and caches each model. It shows saved held-out results, dataset split counts, probabilities, and optional attribution maps. Text uses gradient-times-embedding saliency; images use Grad-CAM on the final ResNet block. For multimodal input, one modality is held fixed while attributing the other. These visualizations describe local model sensitivity; they are not verified human explanations or proof that a prediction is correct. Missing checkpoints are reported in the interface.

## Research report and limitations

See [`reports/research_report.md`](reports/research_report.md) for the current methodology, actual experiment scopes, results, and future work. This project is limited by small and imbalanced Hinglish meme data, OCR/text noise, sarcasm, cultural context, possible repeated captions, and early CPU-only experiments. It should not be described as covering all Indian languages or as having demonstrated multimodal improvement. Dataset licenses and use conditions remain those of the official source.

## Tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test_dataset.py
git diff --check
```

Tests use lightweight synthetic models for forward passes, checkpoint behavior, metrics, and attribution shapes; they do not download pretrained weights.
