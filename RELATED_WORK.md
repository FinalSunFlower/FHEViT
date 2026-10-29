# Related Work

This is a focused citation map for the current FHEViT scope: chest-radiograph
visual representations, reduced CKKS computation, and privacy leakage after a
representation is decrypted. It is not a systematic review, and it does not
establish worldwide priority or novelty.

## Data and model sources

- Wang et al., [ChestX-ray8: Hospital-scale Chest X-ray Database and
  Benchmarks](https://arxiv.org/abs/1705.02315), CVPR 2017. Cite the original
  NIH dataset paper and the NIH ChestX-ray14 terms when using the data.
- OpenGVLab, [InternVL project and model documentation](https://github.com/OpenGVLab/InternVL).
  Cite the specific InternVL/InternVL2 model-card or technical-report version
  actually used; the adapter in this repository is not the model itself.
- Qwen team, [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191),
  2024. Cite only when reporting the Qwen2-VL comparison.
- Benaissa et al., [TenSEAL: A Library for Encrypted Tensor Operations Using
  Homomorphic Encryption](https://arxiv.org/abs/2104.03152), 2021. The measured
  block here uses TenSEAL/CKKS and does not imply full-model FHE.

## Homomorphic inference

- [THE-X](https://doi.org/10.18653/v1/2022.findings-acl.277) studies
  homomorphic Transformer inference and polynomial approximations.
- [Primer](https://doi.org/10.1109/DAC56929.2023.10247719) and
  [BOLT](https://doi.org/10.1109/SP54263.2024.00130) are relevant systems work
  for encrypted Transformer inference and cost comparisons.
- [PrivCirNet](https://doi.org/10.52202/079017-3550) includes a ViT experiment;
  this rules out blanket “first HE ViT” positioning.
- [SHAFT](https://doi.org/10.14722/ndss.2025.242287) and
  [non-interactive secure Transformer inference](https://doi.org/10.14722/ndss.2025.230868)
  are relevant comparisons for interaction and trust assumptions.

These systems differ in task, security model, model size, interaction, hardware,
and output surface. A runtime or depth comparison is meaningful only when those
dimensions are reported alongside the number.

## Representation leakage and medical privacy

- [Privacy Vulnerability of Split Computing](https://arxiv.org/abs/2107.06304)
  studies reconstruction from intermediate representations.
- [ResSFL](https://doi.org/10.1109/CVPR52688.2022.00995) is an attacker-aware
  split-learning defense and a relevant reconstruction baseline family.
- [Generative Medical Image Anonymization](https://arxiv.org/abs/2501.09114)
  studies a medical-image privacy/utility tradeoff using image synthesis.
- [Medical imaging deep learning with differential privacy](https://doi.org/10.1038/s41598-021-93030-0)
  provides context for privacy/utility costs in medical imaging.

FHE protects computation while values remain encrypted. When a representation
is decrypted, reconstruction, linkage, attribute inference, and membership
attacks become separate questions; encryption alone does not answer them.

## Positioning and limits

The current empirical contribution is a bounded evaluation of PCA-prefix
utility and reconstruction leakage on a development analysis set, alongside a
small CKKS-block feasibility measurement. It is not a completed private release,
a clinical validation, a full encrypted VLM, or evidence that a useful output is
private. No “first”, “lossless”, “zero leakage”, or SOTA claim is supported.

Useful search terms: `fully homomorphic encryption vision transformer`,
`CKKS medical imaging`, `chest X-ray representation privacy`,
`split inference feature inversion`, and `medical vision-language model privacy`.
