# Setup and reproduction

## CPU geometry replay

Use Python 3.11/3.12 and the root `requirements.txt`. Run `python scripts/geometry_similarity.py` from the root. Outputs go to `results/geometry_replay/`; use `--data` and `--out` to choose other paths. This publication was replayed with Python 3.12.7 on Windows.

Run the existing analytic tests independently:

```bash
cd experiments/rf_surrogate
python -m unittest test_experiment.py -v
```

```bash
cd experiments/wrfgs_one_room
python -m unittest test_rf_math.py -v
```

## CLIP and DINOv2 notebooks

`notebooks/originals/` contains the original Colab workflows and distinct exported variants as valid notebooks. They preserve experimental code and historical paths; they are not promised to run unmodified on a local Windows machine. Supply your own Drive layout and `Room*` image directories. The root `data/original_figures/` preserves raw filenames rather than guessing room assignments.

Install vision dependencies according to [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), and the notebook's Hugging Face Transformers calls. These download model weights. GPU/PyTorch compatibility is separate from the CPU requirements.

## RF baseline

Use Linux/Colab with a supported Sionna RT GPU environment. The historical notebook installs `sionna-rt==1.2.1` and `scikit-learn==1.6.1`, plus NumPy, pandas, matplotlib and joblib. Consult [Sionna RT](https://github.com/NVlabs/sionna-rt/tree/v1.2.1) for hardware/runtime setup.

From `experiments/rf_surrogate/`, run `python experiment.py prepare`, then `pilot`, `generate`, `train`, and `evaluate` in that order. Set `RF_SURROGATE_OUT` to a new output directory if desired. Inspect the pilot before the full generation stage. `fast_model.py` and `benchmark.py` implement the original subsequent fast-candidate selection and timing workflow.

The included scenes support regeneration. The original consolidated reference dataset and baseline trained models are not present in the local source folders copied into this repository. The historical Colab runner expects an external `RF_Surrogate_Package.zip`; the direct commands above use the repository source instead.

## Controlled follow-up

`experiments/rf_followup/followup.py --base ORIGINAL --out NEW_DIRECTORY` requires the original completed baseline run directory, including its reference dataset and trained baseline artifacts. The saved evaluation evidence in `results/rf_followup/` is sufficient to inspect results, not to retrain the follow-up from scratch.

For the saved predictor, use the historical compatible environment (scikit-learn 1.6.1, NumPy 2.1.3, SciPy 1.16.3 and joblib 1.6.0); the root CPU requirements are not a model-serialization lockfile. Only load trusted joblib files. An example from the root is:

```bash
python experiments/rf_followup/predict_followup.py --scenes experiments/rf_surrogate/scenes --out results/rf_followup --room fullroom --tx 1 1.2 -4 --rx 3 1.2 -4
```

Choose valid free-space positions for your scene. The added `--scenes` option makes the original inference helper usable with this repository layout. Model loading/inference was not rerun in the compatible historical environment during publication.

## WRF-GS+ adapter

Fetch the pinned original implementation from the repository root:

```bash
python scripts/fetch_wrfgs.py
```

The helper clones commit `e564f3184463717ae08660a8b7232c6318c73ca6` with its submodules into the ignored vendor directory. Read the upstream licenses before use. In a Google Colab Linux runtime, run `experiments/wrfgs_one_room/setup_colab.py` with the default Python. It builds an isolated Python 3.11 / PyTorch 2.6 CUDA 12.4 environment and may install GCC/build dependencies. It is not a Windows setup script.

Then change into `experiments/wrfgs_one_room/`, set `WRF_RUN_DIR` to your persistent output directory, and use `/content/wrfgs_env/bin/python run_pipeline.py --through pilot`. Review the checks before continuing with `--through data`, `--through train`, or `--through evaluate`. Source and scene assets are included; vendor code and the historical training outputs are external.

## RF-3DGS

The cleaned original notebook expects the upstream `RF-3DGS_dataset.zip` and initialization checkpoint. Obtain these from [SunLab-UGA/RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS). They are not bundled. Use a separate runtime from WRF-GS+.

## What has not been reproduced

No fresh CUDA build, CLIP/DINO model download, Sionna label generation, model training, or historical serialized-model inference was performed for this upload. Current cloud environments may require adjustments. Numerical test success is not end-to-end scientific reproduction.
