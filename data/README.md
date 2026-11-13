# Oxford-IIIT Pet responses

`oxford_pets.npz` contains the cached class responses used for the paper's OxfordPet37 experiment: **7,349 images, 5 frozen VLMs, and 37 named classes**. The file is approximately 10.3 MiB and loads with NumPy alone.

The source dataset is [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/). This package contains probability responses, evaluation labels, sample identifiers and saved splits. Original images, visual embeddings and model weights are not included.

## Contents

| Array | Shape | Description |
|---|---|---|
| `probabilities` | `(7349, 5, 37)` | Float64 candidate-normalized responses; axes are image, source, class |
| `labels` | `(7349,)` | Zero-based class IDs, used only for evaluation |
| `sample_ids` | `(7349,)` | Stable identifiers in the collected sample order |
| `class_names` | `(37,)` | Class vocabulary in the probability-column order |
| `model_ids` | `(5,)` | Model identifiers in the source-axis order |
| `model_revisions` | `(5,)` | Exact checkpoint revisions |
| `image_sha256` | `(7349,)` | Recorded SHA-256 hashes of the source image bytes |
| `seeds` | `(5,)` | Split seeds: 2026, 2027, 2028, 2029 and 2030 |
| `train_indices` | `(5, 5879)` | Saved unlabeled fitting indices, one row per seed |
| `test_indices` | `(5, 1470)` | Matching held-out indices |
| `original_splits` | `(7349,)` | Original `trainval`/`test` identifiers, retained for provenance |
| `image_paths` | `(7349,)` | Relative image paths in the collected sample order |

All arrays load with `allow_pickle=False`. Probabilities, labels, sample order, source order, class order and split indices are unchanged from the experiment inputs. `metadata.json` records the scoring prompts, model revisions, schema and checksums.

## Loading

```python
import numpy as np

with np.load("data/oxford_pets.npz", allow_pickle=False) as data:
    probabilities = data["probabilities"]
    class_names = data["class_names"]
    seed_index = np.flatnonzero(data["seeds"] == 2026).item()
    train = data["train_indices"][seed_index]
    test = data["test_indices"][seed_index]
```

For inductive evaluation, fit on `probabilities[train]`, freeze the fitted statistics, and predict `probabilities[test]`. Read `labels[test]` only when evaluating the saved predictions. Use the stored indices rather than generating a new split from the seed. The paper's repeated 80/20 evaluation differs from the original dataset's supervised `trainval`/`test` protocol.

For transductive evaluation, fit on every probability row and evaluate the resulting memberships on that same pool. This protocol uses no reference labels during fitting.

The five sources are Qwen2.5-VL-7B-Instruct, MiniCPM-V-4.5, LLaVA-v1.6-Mistral-7B, Phi-3.5-Vision-Instruct and SmolVLM2-2.2B-Instruct. Each probability vector is a softmax over candidate scores defined by mean candidate-token log likelihood. These responses are candidate-normalized scores rather than calibrated class posteriors.

## Attribution and license

This data directory is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); the full text is in `LICENSE.txt`. The Oxford-IIIT Pet website specifies the same license for the source dataset. Copyright in the source images remains with their original owners. The source images are not redistributed here. Model weights retain their respective upstream terms and are not part of this package.

Please cite the original dataset:

```bibtex
@inproceedings{parkhi2012cats,
  author = {Parkhi, Omkar M. and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, C. V.},
  title = {Cats and Dogs},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2012}
}
```
