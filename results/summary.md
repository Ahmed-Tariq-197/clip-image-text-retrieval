# Summary

Seed 42; t-SNE perplexity 5.

## Core retrieval (7 images x 12 texts)

| Model | I2T Top-1 | I2T MeanRank | T2I Top-1 | T2I MeanRank | Excl. I2T/T2I |
|---|---:|---:|---:|---:|---:|
| clip-vit-base-patch32 | 100.0% | 1.00 | 100.0% | 1.00 | 0/4 |
| clip-vit-large-patch14 | 100.0% | 1.00 | 100.0% | 1.00 | 0/4 |

## Expanded set (+hard distractors), reported separately

| Model | I2T Top-1 | I2T MeanRank | T2I Top-1 | T2I MeanRank | # real top-1 errors |
|---|---:|---:|---:|---:|---:|
| clip-vit-base-patch32 | 100.0% | 1.00 | 100.0% | 1.00 | 0 |
| clip-vit-large-patch14 | 100.0% | 1.00 | 100.0% | 1.00 | 0 |

## Modality gap, core set

| Model | Centroid dist. | Cos matched | Cos unmatched | 2D nearest-valid PCA | 2D nearest-valid t-SNE | Retrieval Top-1 (I2T) |
|---|---:|---:|---:|---:|---:|---:|
| clip-vit-base-patch32 | 0.883 | 0.316 | 0.149 | 28.6% | 14.3% | 100.0% |
| clip-vit-large-patch14 | 0.905 | 0.258 | 0.091 | 42.9% | 14.3% | 100.0% |
