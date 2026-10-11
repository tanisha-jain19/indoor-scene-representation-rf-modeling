# Attribution and reuse

This repository organizes Tanisha Jain's internship research and subsequent related experiment work. Naming Airtel describes the internship context; it does not imply company endorsement, product status, or ownership of the cited research algorithms.

## Research software

- [OpenAI CLIP](https://github.com/openai/CLIP): pretrained visual representations.
- [Meta DINOv2](https://github.com/facebookresearch/dinov2): pretrained visual representations, loaded through Transformers in the original notebooks.
- [NVIDIA Sionna RT](https://github.com/NVlabs/sionna-rt): radio propagation simulation.
- [WRF-GS+](https://github.com/wenchaozheng/WRF-GSplus): the original Gaussian/deformation model and RF rendering implementation. The adapter references commit `e564f3184463717ae08660a8b7232c6318c73ca6`.
- [RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS): radio radiance-field research code and external dataset.
- [GRAPHDECO Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): underlying Gaussian-splatting research and related components.
- Blender, NumPy, SciPy, pandas, scikit-learn, PyTorch, matplotlib, seaborn, Pillow and Hugging Face Transformers support the workflows.

Upstream source, pretrained weights, and datasets retain their respective licenses and notices. WRF-GS+ submodules have their own license terms; its top-level license is not a replacement for those component terms. Downloaded vendor code is ignored by Git and must retain its original notices.

The public repository contains self-authored simulated assets as confirmed by the project owner. No broad MIT/Apache license is applied to this mixed research collection or to third-party simulation material. Consult the relevant author and upstream terms before reuse.

## Publication changes

Fresh documentation, source organization, a portable geometry replay, a portable image-inspection utility, and a `--scenes` inference option were added. Original notebook outputs and private runtime metadata were removed. Scientific limitations and provenance were retained; recorded results are not presented as new runs.
