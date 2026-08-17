# Q5 Archive Evaluation Results

Last updated: 2026-08-13T15:25:54+01:00

Archive-only inference for the Q5 paired fine-tuned model set. Each row is a five-fold unweighted `n=1` aggregate over the archive entries with resolved masks only.

This report excludes the 295 archive eval entries that lacked masks. The original full archive run is preserved at `runs\q5_archive_eval_local_masked_20260813`; the filtered masked-only result root is `runs\q5_archive_eval_local_masked_only_20260813`.

Masked-only sample set: 1,684 dorsal images from 99 archive users. Removed no-mask entries: 295 / 1,979.

Run root: `runs\q5_archive_eval_local_masked_only_20260813`
Execution context: `local`
Operating point: best fold-level threshold with FPR <= 5.0%; if unavailable, lowest-FPR threshold in the 10-30 age-threshold sweep.

## Seed-Level Results

| Seed | Arm | Folds | Samples/fold | MAE | RMSE | Adult-gate AUC | Mean FPR | Adult FNR | Mean tau | Folds <= target FPR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `R0` | 5 | 1684 | 10.938 | 12.486 | 0.5748 | 39.69% | 13.11% | 30.00 | 0/5 |
| 42 | `Pooled` | 5 | 1684 | 11.464 | 12.986 | 0.5053 | 46.34% | 14.19% | 30.00 | 0/5 |
| 43 | `R0` | 5 | 1684 | 11.101 | 12.618 | 0.5731 | 38.20% | 18.27% | 30.00 | 0/5 |
| 43 | `Pooled` | 5 | 1684 | 10.792 | 12.284 | 0.6208 | 33.03% | 20.06% | 30.00 | 0/5 |
| 44 | `R0` | 5 | 1684 | 11.376 | 13.198 | 0.5453 | 40.60% | 19.25% | 30.00 | 0/5 |
| 44 | `Pooled` | 5 | 1684 | 10.907 | 12.400 | 0.6252 | 33.37% | 16.16% | 30.00 | 0/5 |
| 45 | `R0` | 5 | 1684 | 10.336 | 11.840 | 0.6315 | 31.66% | 21.02% | 30.00 | 0/5 |
| 45 | `Pooled` | 5 | 1684 | 10.123 | 11.684 | 0.6848 | 25.74% | 22.97% | 30.00 | 0/5 |
| 46 | `R0` | 5 | 1684 | 11.248 | 12.760 | 0.5913 | 37.57% | 14.45% | 30.00 | 0/5 |
| 46 | `Pooled` | 5 | 1684 | 11.113 | 12.658 | 0.6013 | 35.69% | 17.85% | 30.00 | 0/5 |
| 47 | `R0` | 5 | 1684 | 10.852 | 12.433 | 0.5858 | 36.26% | 20.16% | 30.00 | 0/5 |
| 47 | `Pooled` | 5 | 1684 | 10.030 | 11.661 | 0.6909 | 24.00% | 25.04% | 30.00 | 0/5 |
| 48 | `R0` | 5 | 1684 | 11.162 | 12.791 | 0.5758 | 38.66% | 16.44% | 30.00 | 0/5 |
| 48 | `Pooled` | 5 | 1684 | 10.573 | 12.114 | 0.6355 | 32.06% | 20.67% | 30.00 | 0/5 |

## Arm Means

| Arm | Mean MAE | Mean RMSE | Mean Adult-gate AUC | Mean FPR | Adult FNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `R0` | 11.002 | 12.590 | 0.5825 | 37.52% | 17.53% |
| `Pooled` | 10.715 | 12.255 | 0.6234 | 32.89% | 19.56% |
| `Mean-age fixed baseline, age=25.17` | 10.707 | 13.472 | NA | NA | NA |
| `Median-age fixed baseline, age=19.00` | 9.894 | 14.815 | NA | NA | NA |

For MAE, the median-age predictor is the optimal fixed-age baseline on this masked-only sample set.

## Paired Comparisons

Positive mean advantage means `Pooled` is better than `R0`; for MAE, RMSE, and Adult FNR this is a reduction, while for AUC it is an increase.

| Metric | Mean Pooled advantage | Wins | Paired t p two-sided | Wilcoxon p two-sided |
| --- | ---: | ---: | ---: | ---: |
| `mae` | 0.2874 | 6/7 | 0.1264 | 0.1562 |
| `rmse` | 0.3341 | 6/7 | 0.1069 | 0.1094 |
| `auc_adult_gate` | 0.0409 | 6/7 | 0.1054 | 0.1562 |
| `adult_fnr` | -2.0325 pp | 1/7 | 0.0883 | 0.1094 |

## Aggregated Samples

Per-user aggregation was recomputed from the masked-only raw predictions for `n=1,2,3,4` using the existing repository aggregation logic. Remainders smaller than the group size are discarded.

| n | Arm | Samples/fold | MAE | RMSE | Adult-gate AUC | Mean FPR | Adult FNR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `R0` | 1684.0 | 11.002 | 12.590 | 0.5825 | 37.52% | 17.53% |
| 1 | `Pooled` | 1684.0 | 10.715 | 12.255 | 0.6234 | 32.89% | 19.56% |
| 2 | `R0` | 822.0 | 10.941 | 12.434 | 0.6060 | 35.62% | 16.06% |
| 2 | `Pooled` | 822.0 | 10.652 | 12.082 | 0.6529 | 30.49% | 17.88% |
| 3 | `R0` | 516.0 | 10.889 | 12.341 | 0.6090 | 35.45% | 15.99% |
| 3 | `Pooled` | 516.0 | 10.599 | 11.969 | 0.6597 | 29.84% | 17.80% |
| 4 | `R0` | 390.0 | 10.900 | 12.287 | 0.6237 | 34.34% | 15.34% |
| 4 | `Pooled` | 390.0 | 10.614 | 11.930 | 0.6724 | 29.04% | 17.23% |

## Output Files

- Seed metrics CSV: `runs\q5_archive_eval_local_masked_only_20260813\archive_seed_metrics.csv`
- Paired comparison CSV: `runs\q5_archive_eval_local_masked_only_20260813\archive_paired_comparison.csv`
- Aggregation metrics CSV: `runs\q5_archive_eval_local_masked_only_20260813\archive_aggregation_metrics.csv`
- MAE by age plot: `runs\q5_archive_eval_local_masked_only_20260813\archive_mae_by_age_smoothed.png`
- MAE by age CSV: `runs\q5_archive_eval_local_masked_only_20260813\archive_mae_by_age.csv`
- Real vs inferred scatter plot: `runs\q5_archive_eval_local_masked_only_20260813\archive_real_vs_inferred_scatter.png`
- Real vs inferred ensemble CSV: `runs\q5_archive_eval_local_masked_only_20260813\archive_real_vs_inferred_ensemble.csv`
