> Historical local experiment record, not a newly executed validation.

# Executed RF surrogate results

The prototype was trained and evaluated in Colab using 1,815 Sionna reference queries from 11 configurations. Seven configurations supplied training data, two supplied validation data, and two supplied 320 test links. All share the same wall mesh; this does not validate transfer to unrelated buildings.

The fast predictor is a 250-tree Extra Trees regressor with minimum leaf size 2 and one CPU thread. It predicts all 16 descriptors from 50 geometry/material/TX/RX features without importing Sionna. Model loading and standalone prediction were tested in a separate process; predictions matched.

| Primary output | Fast model MAE | Lookup MAE | Fast model R² |
|---|---:|---:|---:|
| Noncoherent summed path gain | 2.664111 dB | 3.618837 dB | 0.796934 |
| RMS delay spread | 1.067580 ns | 1.143578 ns | 0.010818 |
| Dominant-path power fraction | 0.101891 | 0.106872 | 0.180508 |

On 30 matched queries across both held-out configurations, the same Tesla T4 runtime measured median Sionna time 54.016 ms and CPU full16 prediction time 21.306 ms: **2.535× warm speedup**. Prediction includes feature extraction, and simulation includes synchronized solving and descriptor extraction. This excludes imports, model/scene loading, compilation and disk writes. Fresh-process model loading took 2.108 seconds. Use a persistent loaded session for repeated queries.

Accuracy remains mixed. No-chair path-gain MAE is 0.997 dB; no-sofa MAE is 4.331 dB. Delay R² is near zero. The predicted 80-D embeddings are unreliable: reference/prediction cosine is 0.619875 for no-chair and -0.060452 for no-sofa. Do not present this as a validated embedding replacement or generally accurate Sionna replacement.

The original gradient-boosting model is preserved. Its gain MAE was 2.120932 dB, delay MAE 1.286280 ns and fraction MAE 0.095901, with no full16 speed advantage. The fast candidate was an exploratory follow-up after initial evaluation; candidate selection used validation error and latency, not test labels. Neither candidate reached the proposed 3× validation threshold, so the specified fastest-candidate fallback applied. Earlier 4.91× timing concerned only three gradient-boosting outputs.

Checks passed: analytic free-space gain/delay; doubled-ray pilot convergence; scene/material and placement checks; split separation; train-only preprocessing; finite outputs; feature reconstruction; standalone inference; primary-descriptor reproduction on 30 queries after runtime reset. These checks do not eliminate the observed accuracy limitations.

Saved on Drive under `MyDrive/RF_Surrogate_20260923`: reference paths, consolidated dataset, both models, baseline comparisons, all predictions, embeddings, figures, timing records, source, environment, final report, and a clean replay notebook. The live notebook had an autosave conflict; the independent replay notebook and execution history preserve the work.

Next research step: reserve independent room geometries, expand training diversity and difficult-link coverage, improve RF-sensitive geometry features, and diagnose descriptor/pooling/scaling errors. Repeatedly tuning against the present test rooms would invalidate their use as untouched final evaluation.
