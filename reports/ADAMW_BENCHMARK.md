# AdamW 1B H200 benchmark

## Executed work

- 35-run staged 1.315B decoder-only feasibility sweep across LR, beta2,
  weight decay, warmup, batch, longer endpoints, six seeds, and CPT.
- Public-corpus 1.012B scratch run using 120M WikiText tokens as the prepared
  corpus, six 180-step trials, one 700-step endpoint, and four independent
  250-step seeds.
- CPT from the saved 700-step checkpoint on a GSM8K reasoning corpus.
- Qwen3-0.6B Base SFT followed by real two-GPU GRPO and a fixed 100-question
  GSM8K holdout evaluation.

## Public-corpus AdamW results

| Run | LR | beta2 | WD | warmup | validation loss | tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| lr4e4 | 4e-4 | .95 | .10 | 1% | 6.4736 | 31,998 |
| warm005 | 4e-4 | .95 | .10 | .5% | 6.4758 | 30,277 |
| wd05 | 4e-4 | .95 | .05 | 1% | 6.4769 | 31,129 |
| b2_98 | 4e-4 | .98 | .10 | 1% | 6.4904 | 30,869 |
| lr2e4 | 2e-4 | .95 | .10 | 1% | 6.5018 | 32,039 |
| lr6e4 | 6e-4 | .95 | .10 | 1% | 6.6069 | 31,180 |

The selected configuration was LR `4e-4`, beta2 `.95`, weight decay `.1`, and
1% warmup. The 700-step endpoint reached validation loss `5.4984` at 30,493
tokens/s. The four 250-step seeds had mean validation loss `6.19745` and sample
standard deviation `0.01769`; they are not compared directly with the longer
700-step endpoint.

The CPT run reached validation loss `4.5560` on its separate GSM8K-derived
validation corpus at 15,366 tokens/s. Cross-corpus loss values are not treated
as an apples-to-apples improvement claim.

## SFT and GRPO holdout

| Checkpoint | exact numeric accuracy | required `####` format rate |
|---|---:|---:|
| Qwen3-0.6B Base | 22% | 0% |
| 160-step SFT | 36% | 45% |
| 60-step two-GPU GRPO | 47% | 75% |

The GRPO reward combined exact numeric correctness and a separate format
reward. The final GRPO training runtime was 388.7 seconds for 60 steps.

## Recovery verification

The confirmation worker was deliberately killed after the 700-step model was
saved. Its guardian restarted once, skipped the completed endpoint, and resumed
the missing seed runs. GPU telemetry was also deliberately killed; its guardian
restarted once and continued appending to the same CSV.

## Limitations

- This is bounded evidence, not a converged web-scale pretraining run.
- The 35-run feasibility corpus is structured synthetic data and is reported
  separately from the public-corpus experiment.
- The supplied package's architecture is Pythia-style, not OLMo.
- MFU requires a defensible model-FLOP convention and is not fabricated here;
  raw throughput, power, utilization, HBM, and wall time are retained instead.

