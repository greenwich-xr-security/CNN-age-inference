# HPC job history

## Active and queued jobs

| Slurm job | Run directory | Purpose | Resources | State |
| --- | --- | --- | --- | --- |
| `1050704` | `runs/v2s_handrgbd_prolific_only_nll_4gpu_16cpu_20260727_384` | Pure Gaussian-NLL counterpart to job `1050688` | 4 GPUs, 16 CPU cores, 64 GB RAM; batch 16 per GPU | `RUNNING` |
| `1050705` | `runs/v2s_handrgbd_prolific_only_crps_4gpu_16cpu_20260727_384` | Pure Gaussian-CRPS counterpart to job `1050704` | 4 GPUs, 16 CPU cores, 64 GB RAM; batch 16 per GPU | `RUNNING` |
| `1050710` | `runs/v2s_handrgbd_prolific_synthetic_trainval_nll_4gpu_16cpu_20260727_384` | Pure Gaussian-NLL real + SyntheticDorsalHands counterpart to job `1050689` | 4 GPUs, 16 CPU cores, 64 GB RAM; batch 16 per GPU | `PENDING` (resources) |

This run reuses the fixed 96-user real test split, HandRGBD + ProlificHands
data, 20-images-per-user and 200-images-per-age caps, V2-S at 384 px, five
folds, seed 42, 240 epochs, and patience 10. Its only objective is Gaussian
NLL: MSE, MAE, within-user spread, and embedding losses are all disabled.
Job `1050705` has the identical protocol but uses CRPS as its only regression
objective. Together, the two runs compare the MAE/RMSE and adult-gate trade-offs
of two clean probabilistic objectives against the mixed-loss job `1050688`.

Job `1050710` extends the pure-NLL comparison to HandRGBD + ProlificHands +
SyntheticDorsalHands, using the same locked real-user test split as `1050689`.
It keeps only image-level aggregation (`n=1`) because synthetic images do not
have compatible user groups; all non-NLL losses and age reweighting are disabled.
It was submitted on 2026-07-27 and is awaiting the four-GPU allocation.

Submitted on 2026-07-24 to answer the real-versus-synthetic distribution
question. Both jobs use EfficientNet-V2-S at 384 px, five folds, seed 42,
240 epochs with patience 10, and the same fixed real-user test split:

`runs/v2s_handrgbd_prolific_only_2gpu_20260724_384/test_users.json`.

| Slurm job | Run directory | Training data | Resources | DDP port |
| --- | --- | --- | --- | --- |
| `1050688` | `runs/v2s_handrgbd_prolific_only_4gpu_16cpu_20260724_384` | HandRGBD + ProlificHands | 4 GPUs, 16 CPU cores, 64 GB RAM; batch 16 per GPU | 29685 |
| `1050689` | `runs/v2s_handrgbd_prolific_synthetic_trainval_4gpu_16cpu_20260724_384` | HandRGBD + ProlificHands + SyntheticDorsalHands | 4 GPUs, 16 CPU cores, 64 GB RAM; batch 16 per GPU | 29686 |

## Completion status

Both jobs completed successfully on 2026-07-24 with exit code 0.

| Slurm job | State | Elapsed time | Verified allocation |
| --- | --- | --- | --- |
| `1050688` | `COMPLETED` | 2:53:46 | 4 GPUs, 16 CPUs, 64 GB RAM |
| `1050689` | `COMPLETED` | 3:34:49 | 4 GPUs, 16 CPUs, 64 GB RAM |

## Held-out test results

Each fold was evaluated on the same locked real-data test set of 96 users and
1,069 images. Values below are the unweighted mean of the five per-fold test
metrics at image-level aggregation (`n=1`).

| Training data | MAE (years) | RMSE (years) | Adult-gate AUC |
| --- | ---: | ---: | ---: |
| HandRGBD + ProlificHands | 5.064 | 6.532 | 0.968 |
| HandRGBD + ProlificHands + SyntheticDorsalHands | 4.636 | 6.093 | 0.954 |

Adding SyntheticDorsalHands improved point age estimation: MAE decreased by
0.428 years (8.4%) and RMSE by 0.439 years (6.7%). Every fold had lower test
MAE with synthetic training data. Adult-gate ranking did not improve: AUC
decreased by 0.0135 (0.968 to 0.954). The synthetic data is therefore
promising for age regression accuracy, but should not yet be adopted for an
adult/minor gate without further gate-specific tuning and evaluation.

## Data-handling details

- Real data retains the pipeline's per-user cap of 20 images and per-age cap
  of 200 images.
- SyntheticDorsalHands is uncapped: all 13,451 images remain in the
  training/validation pool.
- Synthetic samples are image-level rather than identity-level. The
  submission script automatically uses `USER_GROUP_SIZES=1` and
  `AGG_SIZES=1` whenever SyntheticDorsalHands is included, preventing
  artificial duplication in training and dropped samples during evaluation.
- The 96 fixed real held-out users are excluded from both training and
  validation. Synthetic images are not part of that real-user test set.

## Monitoring

```bash
ssh -l 'rb3434w@staff' 100.96.122.39 "tail -f ~/CNN-age-inference/logs/v2s-hrgbd-prolific-4g16c-1050688.log"
ssh -l 'rb3434w@staff' 100.96.122.39 "tail -f ~/CNN-age-inference/logs/v2s-hrgbd-pro-synth-4g16c-1050689.log"
```

The preceding 4-GPU/4-core attempts were cancelled because GPU utilisation
showed data-loader starvation. These replacement jobs allocated 16 CPU cores
to support four DDP ranks and two loader workers per rank.
