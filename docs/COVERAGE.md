# Source coverage and publication scope

The source review covered `Airtel Report` and the related `Airtel_Similarity` research workspace. `SOURCE_MANIFEST.json` maps copied inputs to published paths and records original hashes. Notebook contents were cleaned, so their original hashes describe provenance rather than the modified public file.

## Included

- Every distinct top-level research notebook and exported Colab Python variant in `Airtel Report`. Exports containing notebook-only syntax were converted to `.ipynb`; none are presented as ordinary runnable Python scripts.
- The original `image.py` computation, made portable as `scripts/inspect_rf_image.py` with an input argument instead of a private temporary path.
- RF baseline, controlled follow-up and WRF-GS+ computational scripts, numerical tests, and inference tools.
- The saved follow-up baseline source snapshot, so its historical code can be inspected independently of later source changes.
- Blender `.blend`/`.blend1` files, GLB/PLY/XML exports, coordinate CSVs, rendered room views, original figures, and approved simulation results available in the reviewed source folders.
- Saved follow-up prediction tables and model, and fresh geometry-replay outputs.

## Explicitly external or omitted

| Material | Reason / alternative |
| --- | --- |
| Personal identity, employment and administrative documents | Not part of the research code; excluded from public disclosure |
| Chat transcripts, private cloud links, temporary logs | Not needed to understand or run the project |
| Document/package builder scripts | Presentation/delivery tooling; not part of the model or simulation computation |
| Vendored RF-3DGS / WRF-GS+ repositories and compiled binaries | Linked upstream; WRF-GS+ has a pinned fetch helper |
| Third-party RF dataset ZIP, initialization checkpoint and RF render archive | Obtain from upstream; not relicensed or copied into this repository |
| Approximately 695 MB intermediate visual dataset directory and duplicate ZIP deliveries | Rendered intermediates are not bundled; source scenes and the original selected room views are included |
| Full WRF-GS+ completed training/test artifact archive | Referenced in historical notes but not present in the reviewed local delivery folder |
| RF baseline full reference dataset and original trained model | Not present in the reviewed baseline source directory; regeneration code is included |
| Duplicate pickled geometry vectors | Human-readable coordinate/vector CSVs and deterministic replay code are provided |

This is complete coverage of the identified main coding workstreams, not a byte-for-byte backup of the device or a claim that every historical experiment can be reproduced from the checkout alone. Source notebook versions and raw figure names are retained rather than silently selecting an unverified final version.
