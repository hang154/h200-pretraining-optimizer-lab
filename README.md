# H200 Pretraining & Optimizer Lab

Reproducible evidence from scratch pretraining, optimizer studies, continued
pretraining, and recovery tests on NVIDIA H200 GPUs.

## Measured results

| Run | Parameters | Optimizer | Tokens | Validation loss | tok/s | Peak HBM GiB |
|---|---:|---|---:|---:|---:|---:|
| checkpoint resume | 405,334,016 | AdamW | 153,600 | 6.957358 | 3,484.7 | 3.33 |
| 1B baseline | 1,011,781,632 | AdamW | 16,384,000 | **4.755323** | 30,990.4 | 8.90 |
| 1B sweep-selected | 1,011,781,632 | AdamW | 16,384,000 | 4.783076 | 31,117.0 | 8.90 |
| 410M AdamW | 405,334,016 | AdamW | 2,457,600 | 5.772677 | 11,060.0 | 3.71 |
| 410M Adafactor | 405,334,016 | Adafactor | 2,457,600 | 11.015334 | 9,217.6 | 2.20 |

The short-budget sweep selected the configuration presented here as
`1B sweep-selected`. At the matched 16.384M-token endpoint it did **not**
outperform the baseline. This is a negative result and evidence that
short-horizon hyperparameter rankings can be unstable.

The equal-learning-rate AdamW/Adafactor run is an operational comparison, not
a fair general optimizer ranking. The CPT loss is measured on a distinct
GSM8K-derived corpus and is not compared directly with pretraining loss.

## What is included

- Executed training and sweep scripts.
- Public CSV/JSON summaries and the detailed benchmark report.
- A claim validator that enforces the public naming and negative-result text.
- Checkpoint interruption/resume evidence. A worker resumed from durable step
  50 and completed step 300.

Model weights, checkpoints, raw telemetry, and private infrastructure details
are intentionally excluded.

## Validate

```bash
python scripts/validate_claims.py
python -m py_compile scripts/*.py
```

## Claim boundary

This repository documents bounded experiments and hands-on execution. It does
not claim converged web-scale pretraining or years of production ownership.

