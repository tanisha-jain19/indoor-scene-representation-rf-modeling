> Historical local experiment record, not a newly executed validation.

# Controlled RF-surrogate follow-up

We tested a separate follow-up using the existing data and preserved the original
model and outputs. No new rooms or simulation-label settings were introduced.
The same 1,120 training links, 320 validation links, 320 test links and 55 auxiliary
anchor links were used. Selection used validation only. Because the original test
results were already known, this is an exploratory comparison.

The selected hybrid retains the original Extra Trees model for 13 descriptors.
It adds a separate Extra Trees gain regressor (250 trees, leaf size 1, 70% of
features per split), a separate Extra Trees dominant-fraction regressor (250
trees, leaf size 2), and an absolute-error histogram gradient boosting delay
regressor. The delay head uses 62 geometry features: the original 50, six ideal
wall-reflection excess delays, their minimum/mean/maximum, and three material
obstruction-length fractions. These features do not call Sionna.

| Test metric | Original fast model | Follow-up |
|---|---:|---:|
| Path-gain MAE (dB) | 2.664111 | 2.187291 |
| Path-gain R2 | 0.796934 | 0.762026 |
| RMS-delay MAE (ns) | 1.067580 | 1.280075 |
| RMS-delay R2 | 0.010818 | -0.290489 |
| Dominant-fraction MAE | 0.101891 | 0.090214 |
| Dominant-fraction R2 | 0.180508 | 0.323271 |

Gain MAE improved about 17.9%, and dominant-fraction MAE about 11.5%. Delay MAE
worsened about 19.9%. Gain R2 also fell: reducing average absolute error does not
guarantee reducing squared error. The three errors improved on validation, but
that did not carry over to test delay prediction.

The seven training-configuration diagnostic folds showed mean delay MAE improving
from 0.946 to 0.805 ns, yet both final test configurations had worse delay MAE.
These fixed-setting folds are not an independent nested validation estimate.
They reveal variation across configurations rather than resolving it.

Test embedding cosine also worsened: no-chair 0.620 to 0.548 and no-sofa -0.060
to -0.093. Embedding preprocessing was kept fixed. The remaining 13 predictions
were unchanged; improvements in a few descriptors do not establish reliable
80-D embeddings.

The final matched T4-host benchmark (30 queries, five predictor repeats per query)
measured Sionna at 52.20 ms, the original at 19.15 ms, and the follow-up at 60.96 ms.
The original was 2.73x faster than Sionna; the follow-up's speed ratio was 0.856x,
meaning it was about 16.8% slower than Sionna and 3.18 times slower than the original.
These are warm timings: synchronized solve plus descriptor extraction versus
CPU feature extraction plus full16 inference. Loading, compilation, training and
disk I/O are excluded. The original Sionna reference outputs reproduced within
the benchmark tolerances. This new measurement is separate from the historical
2.54x result. The earlier CPU-only old/new timing is retained separately.

Conclusion: this candidate is **not an overall upgrade**. Keep the original as
the baseline. The experiment demonstrates a gain/fraction improvement with delay,
embedding and runtime regressions; it does not solve the delay limitation or
establish new-building generalization. Neither model is a general Sionna replacement.

Validation includes baseline metric reproduction, original key-file hash checks,
saved-model reload consistency, finite/shape and split checks, train-only target
scaling, fixed original embedding scaling, feature-group ablation/permutation
analysis, per-room metrics, and standalone inference without importing Sionna.


## Saved evidence

- `followup_predictor.joblib`
- `validation_models.csv`
- `validation_heads.csv`
- `training_configuration_cv.csv`
- `delay_feature_ablation.csv`
- `validation_group_importance.csv`
- `test_metrics_all16.csv`
- `test_predictions.csv`
- `embedding_comparison.csv`
- `same_runtime_sionna_benchmark.csv`

The proposed changes were selected on validation; no subsequent test-guided model changes were made. Original work is retained as baseline. No new independent-scene generalization claim.
