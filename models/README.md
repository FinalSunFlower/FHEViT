# Local Model Snapshots

Model snapshots and tokenizer/processor files are not included in the public source
release. The root `.gitignore` excludes all files under `models/`; provider credentials
must stay outside this repository.

| Adapter / role | Provider model page | Expected local directory | Status |
|---|---|---|---|
| InternVL adapter | [OpenGVLab/InternVL2_5-2B](https://huggingface.co/OpenGVLab/InternVL2_5-2B) | `models/internvl2-2b/` | Adapter implementation; current research summary uses the shorter label InternVL2-2B. Verify the exact checkpoint revision used to create any cached features before claiming result reproduction. |
| Qwen comparison adapter | [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | `models/qwen2vl-2b/` | Earlier comparison, not the current primary evaluation line. |

Download only after reviewing the model card, license, gated-access terms, and the
revision required by the experiment. The adapter source documents the expected local
paths and never downloads model weights implicitly at runtime. Example:

```powershell
huggingface-cli download OpenGVLab/InternVL2_5-2B `
  --local-dir models/internvl2-2b --local-dir-use-symlinks False
```

Model files are not enough to reproduce the reported aggregates. The run also needs
the matching adapter/checkpoint artifacts, the exact dataset and cohort manifest,
cached features, and compatible CUDA/Transformers versions. See
[`../data/README.md`](../data/README.md), [`../artifacts/README.md`](../artifacts/README.md),
and [`../docs/ENVIRONMENT.md`](../docs/ENVIRONMENT.md). Do not upload snapshots, cache
files, access tokens, or local model configuration containing private paths.
