# IndicSent-M: Preliminary Experimental Report

## Research objective

This project compares positive/neutral/negative sentiment classification from text, images, and paired text-image inputs in Hinglish meme content. The experiment asks whether image features add measurable value beyond text. No claim of novelty or broad Indian-language coverage is made.

## Data and protocol

The current prepared file has 6,991 examples: 4,898 train, 1,046 validation, and 1,047 test. All rows are labeled `hinglish`. The fixed label order is positive, neutral, negative. The data has substantial imbalance: train counts are 2,918/1,527/453 and test counts are 625/335/87. There are four rows without usable text. Exact duplicate image content is grouped so it does not cross splits; five normalized text-and-label combinations recur across splits and remain in place because they may represent distinct images/samples.

Existing split assignments were preserved. Split generation is seeded and grouped on image content, but GroupShuffleSplit does not guarantee stratification. No current test rows were used for model selection. Best checkpoints are selected using validation macro-F1; strict improvement is required, so the earliest epoch wins a tie.

## Models and training

- Text-only: XLM-RoBERTa base, fine-tuned for one epoch on 4,896 rows; the configured IndicBERT repository was gated and returned an access error.
- Image-only: ResNet-50, trained for one epoch on 128 rows, random encoder initialization.
- Multimodal: XLM-RoBERTa plus randomly initialized ResNet-50 with projected concatenation fusion and an MLP, trained for one epoch on 128 rows.

Image-only and multimodal validation used 48 rows. These runs used deterministic seed 42 and existing train/validation/test partitions, but train/validation subsets were sampled for the bounded runs. Every model was evaluated against the same 1,047 held-out test examples. This unequal training scope means the results are exploratory and not a controlled comparison of training data volume or initialization.

A separate three-epoch XLM-RoBERTa full-split run is present in `checkpoints/text-xlmroberta/`. Its validation predictions were all positive at every epoch (validation macro-F1 0.2471, accuracy 0.5891); strict macro-F1 checkpoint selection retained epoch 1 because all three values tied. This run was not evaluated on the held-out test set and is excluded from the test results below.

## Results

| Model | Test accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text, full split | 0.5969 | 0.1990 | 0.3333 | 0.2492 | 0.4463 |
| Text, subset | 0.5969 | 0.1990 | 0.3333 | 0.2492 | 0.4463 |
| Image, subset | 0.5950 | 0.1989 | 0.3323 | 0.2489 | 0.4457 |
| Multimodal, subset | 0.5969 | 0.1990 | 0.3333 | 0.2492 | 0.4463 |

The held-out test has a majority-positive baseline accuracy of 0.5969. Text and multimodal predicted positive for every test item. Image predicted 1,044 positive and 3 neutral; none were negative. Neutral and negative test F1 are zero for every run. These results indicate class collapse and no evidence that fusion helps. Full per-class scores and confusion matrices are saved with evaluation outputs.

Training took about 25 minutes for the full-split text epoch, 56 seconds for the text subset, 12 seconds for image subset, and 80 seconds for multimodal subset. Current image/multimodal duration estimates do not predict the full-data cost because the subset runs were small and hardware throughput varies. The full text run began before run-duration metadata was added; its history has epoch metrics, but its metadata does not record a best epoch or measured duration.

## Reproducibility and artifacts

Training and validation histories plus run configuration snapshots live beside the checkpoints in `checkpoints/`. Test metrics, class reports, confusion matrix CSVs, and predictions are in the corresponding `results/*-evaluation/` directories. `scripts/build_final_report.py` can consolidate these records and make comparison/training figures without training or changing existing metrics.

Random seed 42 is configured. Training requests deterministic PyTorch algorithms where available. An exact repeat experiment has not been run, so run-to-run variance is unknown. The guarded data preparation command will not overwrite an existing split unless explicitly requested.

The initial full-split text run used correctly ordered inverse-frequency cross-entropy weights of approximately `[0.560, 1.069, 3.603]` for positive, neutral, and negative. Those weights were derived from training counts and did not resolve minority-class collapse. The updated default training config uses a seeded weighted sampler on training examples instead of combining oversampling and weighted loss. This follow-up has not yet been trained or evaluated.

## Limitations and next work

The current dataset is Hinglish rather than a representative sample of Indian languages. It has class imbalance, OCR/text noise, sarcasm, repeated text, and possible source/template biases. One epoch and small vision/multimodal subsets are insufficient for a final model comparison. The gated IndicBERT weights were unavailable, and the subset vision runs did not load pretrained ResNet weights. Confidence scores are not calibrated. Gradient-times-embedding and Grad-CAM visualizations are input-attribution tools, not causal explanations or evidence of model correctness.

Before a final research claim, run approved multi-epoch experiments with XLM-R and verified pretrained ResNet weights where accessible, use comparable data and validation protocols, report repeated-seed variation, compare against the majority baseline, and inspect the collapse with per-class validation predictions. Any change to split methodology should create a separately named split and experiment set rather than replacing the current assignments.
