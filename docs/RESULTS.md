# Results and evidence

## Newly replayed for publication

The CPU geometry workflow completed on 11 rooms with 8 aligned object identifiers. The two generated cosine matrices are in `results/geometry_replay/`. Checks verified dimensions, finite values, symmetry, and unit diagonals within float32 rounding tolerance. This confirms execution of the representation computation, not scientific validation of RF similarity.

The four RF-surrogate analytic tests and three WRF-GS+ math/scene-split tests all passed. The latter checks directional projection, coherent cancellation, room placement, and reproducible disjoint splits. See `VALIDATION.json` for the publication check record.

## RF surrogate baseline and follow-up

The primary table in the README comes from `results/rf_followup/main_comparison.csv`. For both the original fast model and follow-up, the three primary MAEs were recomputed from **320 saved per-query predictions** and matched the recorded table. This is independent arithmetic verification of saved results, not new model inference.

The earlier baseline run recorded approximately 2.535× warm speedup; a separate matched follow-up benchmark recorded approximately 2.73× for the same baseline and 0.856× for the follow-up. These are different timing runs and are not interchangeable. See `same_runtime_sionna_benchmark.json` and its per-query CSV for the later benchmark.

The original and follow-up both struggle with delay and embedding reliability. Follow-up path-gain MAE improves while path-gain R² decreases. The exploratory follow-up is slower than the matched Sionna reference and does not establish new-building generalization. Full local experiment narratives are preserved in `RF_SURROGATE_RECORDED.md` and `RF_FOLLOWUP_RECORDED.md`.

## WRF-GS+

`WRFGS_RECORDED.md` is a saved local completion record reporting 90 test positions, MAE 0.018261, SSIM 0.862078, and warmed model/render time 7.491 ms on a T4. The complete historical predictions/checkpoint archive was not available in the reviewed local delivery folder, so those metrics were not independently recomputed. They are record-attributed observations, with lower evidence strength than the RF follow-up per-query tables.

These scores describe normalized relative directional RF images in one room. They are not absolute power errors, real-world measurement validation, a general accuracy percentage, or a measured speedup against Sionna.

## Optical and RF-3DGS

The cleaned RF-3DGS notebook retains the original code but clears outputs. Recorded metrics and the separately curated optical experiment are documented in the [companion collection](https://github.com/tanisha-jain19/vision-and-rf-gaussian-splatting). Different tasks and splits must not be ranked by a single PSNR table.
