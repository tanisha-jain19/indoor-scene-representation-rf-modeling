> Historical local experiment record, not a newly executed validation.

# One-room WRF-GS+ — completed on Colab T4

Executed and checked on 14 September 2026 in the user's WRF-GS+.ipynb notebook.

## Actual results

| Test method | MAE (lower is better) | SSIM (higher is better) |
|---|---:|---:|
| WRF-GS+ adaptation | 0.018261 | 0.862078 |
| Copy nearest training position | 0.020114 | 0.845950 |
| Average training image | 0.049067 | 0.647438 |

90 reserved test positions. Median warmed T4 prediction time: 7.491 ms (model/render only). Trained 10,000 updates in approximately 272 seconds, excluding setup; validation selected update 5,000. These scores are not accuracy percentages.

## The workflow in simple words

1. **Room:** used your existing fullroom geometry and furniture.
2. **Radio examples:** reused the 600 Sionna RF images already saved in Drive. Checked their file hash, array dimensions, finite values, split separation, and saved LOS/convergence checks.
3. **What an image means:** a directional map of relative radio power arriving at a fixed 4x4 receiver. Bright areas indicate stronger directions. The transmitter changes position, at one height, at 3.5 GHz.
4. **Learning:** gave 420 examples to the authors' Gaussian/deformation model and RF CUDA renderer. Used 90 separate examples to select the best saved model.
5. **Testing:** reloaded both model parts in a fresh Python process and predicted all 90 positions reserved for final testing.
6. **Comparison:** compared predictions against Sionna and two simple baselines. WRF-GS+ had lower average error than both. Main lobes generally match; weaker lobes and difficult examples still show errors.
7. **Saving:** checkpoints, predictions, per-example scores, graphs, source package, and explanation are saved in `MyDrive/WRFGS_fullroom`. The complete archive is `MyDrive/WRFGS_fullroom_results.zip`.

## Errors resolved
- Made subprocess output visible and stopped later commands after failures.
- Selected GCC explicitly and added the environment's build tools to PATH.
- Used GCC 12 because GCC 13 triggered a CUDA internal compiler error.
- Pinned pip 25.0.1 for the extension build.
- Removed the unsupported point-cloud `normals` argument in the adapter.
- Converted loaded transmitter coordinates from float64 to float32.

Both CUDA extensions built. Three CPU numerical tests passed. The GPU smoke test verified gradients and both optimisers; its three losses were 0.404858, 0.382021 and 0.357577. The model then completed actual training and held-out evaluation.


Scope: one simulated room, fixed receiver, normalized relative power, no new-building or measured-RF claim.
