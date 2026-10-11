# Methodology and interpretation

## Visual representation

The original Colab workflows extract CLIP ViT-B/32 and DINOv2-base features from rendered room views, then compare room representations. The historical variants are preserved so changes are inspectable; a later filename is not proof of superior accuracy. The raw view filenames are retained as supplied. Configure the expected `Room*` folders before running the visual notebooks; the mapping from every letter-prefixed image to a numbered room was not verified during publication.

## Geometric representation

Object names are aligned across configurations. Missing objects are represented by coordinates `(0, 0, 0)` to preserve the original experiment convention. Pairwise Euclidean distances are flattened and compared with cosine similarity. The Fourier method applies sine and cosine at six powers-of-two frequencies to each raw coordinate, then normalizes the encoded vector.

The new `geometry_similarity.py` extracts these computations into a local CPU entry point. It validates the required CSV columns and rejects duplicate names/nonfinite coordinates. It does not silently change the mathematical representation.

Limits: zero-filled absent objects can introduce artificial distances and confuse absence with a true origin position. Fourier similarity depends on coordinate scale/origin and periodic encoding. Similarity heatmaps alone do not establish RF equivalence, classifier accuracy, or superiority of one representation.

## RF surrogate

The baseline maps 50 geometry, material, transmitter and receiver features to 16 Sionna-derived descriptors. The reference protocol uses 3.5 GHz, line-of-sight plus specular reflections, and approximated bounding-box obstruction features. It omits diffuse scattering, diffraction, and refraction.

The historical data comprise 1,120 training links, 320 validation links, 320 test links, and 55 auxiliary anchor links across 11 configurations of a shared room family. This is held-out configuration evaluation, not unrelated-building generalization. Path gain is a noncoherent sum of path powers in dB, not received power in dBm.

The follow-up keeps 13 baseline outputs and changes gain, delay and dominant-fraction heads. Its delay features add idealized wall-reflection excess delays and material obstruction fractions. Selection uses validation data, but the study remains exploratory because prior test results were known. Improvements in MAE do not imply improvements in R² or inference speed.

## WRF-GS+

The adapter generates 600 normalized directional RF images in one room: 420 training, 90 validation and 90 test positions. The fixed 4×4 receiver and transmitter-height plane limit the scope. Training uses the upstream Gaussian/deformation model and RF CUDA renderer. Per-image normalization removes absolute received-power information, and the planar array has directional ambiguity.

## RF-3DGS

The historical notebook uses the upstream RF dataset and visual checkpoint before RF training. It is retained separately from the self-authored room-surrogate experiments; these are different datasets and targets and should not be merged into one performance claim.
