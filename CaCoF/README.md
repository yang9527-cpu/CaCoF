# CaCoF

Code for **From Class Bias to Fuzzy Consensus: Semantic-Anchored Clustering with Vision–Language Model Predictions**.

CaCoF learns named consensus assignments and source-specific response prototypes from the class probabilities of frozen vision–language models. Fitting uses no reference labels, image embeddings or backbone updates.

## Setup

Python 3.11 or newer. The included example runs on CPU.

```bash
git clone https://github.com/yang9527-cpu/CaCoF.git
cd CaCoF
python -m pip install -r requirements.txt
```

## Run

The Oxford-IIIT Pet example is included in `data/oxford_pets.npz`.

```bash
# Fit on the saved training split and predict its held-out images.
python main.py --protocol inductive --seed 2026

# Fit and evaluate the entire unlabeled collection.
python main.py --protocol transductive

# Reproduce both protocols and the CalMean control over all five saved splits.
python main.py --protocol both --seed all --method both
```

Outputs are written to `results/`: probabilities, class predictions, sample indices, fitted models, per-run metrics, and `results.csv`. Reference labels are read only after predictions have been saved. Class indices retain the supplied vocabulary; no ground-truth label permutation is applied.

The default parameters are `rho=0.1`, `tau=64`, `gamma=-0.5`, 30 warmup steps and at most 35 profiled optimization steps. The split seed chooses stored indices; mean initialization and full-pool fitting are deterministic. `--threads` controls BLAS threads, and `--output` changes the output directory.

## Results

Oxford-IIIT Pet, five frozen sources and 37 classes. Inductive results average the five supplied 80/20 splits; transductive results describe one full unlabeled pool. ACC and macro-F1 are percentages; Brier is unscaled.

| Method | Protocol | ACC | Macro-F1 | Brier |
|---|---|---:|---:|---:|
| CalMean | Inductive | 85.27 | 84.05 | 0.5939 |
| CaCoF | Inductive | **90.33** | **90.24** | **0.1667** |
| CalMean | Transductive | 86.08 | 84.56 | 0.5945 |
| CaCoF | Transductive | **92.24** | **92.16** | **0.1305** |

For inductive seed 2026, CaCoF obtains ACC 89.7959, macro-F1 89.6318 and Brier 0.184373. Small floating-point differences can occur across numerical libraries.

## Python interface

```python
import numpy as np
from cacof import CaCoF

with np.load("data/oxford_pets.npz", allow_pickle=False) as data:
    p = data["probabilities"]
    row = np.flatnonzero(data["seeds"] == 2026).item()
    train = data["train_indices"][row]
    test = data["test_indices"][row]

model = CaCoF().fit(p[train])
q = model.predict_proba(p[test])
prediction = q.argmax(axis=1)

model.save("model.npz")
restored = CaCoF.load("model.npz")
```

Input shape is `(images, sources, classes)`. All sources must use the same class dictionary and preserve source order between fitting and prediction. `fit_predict(p)` returns the memberships optimized on the fitted collection; `predict_proba(p)` applies the conditional rule with frozen profiles and statistics. These are different inference operators and need not return the same probabilities on training images.

The solver uses an analytical prototype update, the full proportion-feedback derivative, and mirror steps with objective-based backtracking. The membership problem remains nonconvex. Marginal correction does not guarantee calibrated probabilities or robustness to missing classes.

## Data

The included response dataset contains 7,349 images, five source models, evaluation labels, named coordinates and the exact five saved splits. It is approximately 10.3 MiB. Original images, visual embeddings and model weights are not required to reproduce fusion.

See [data/README.md](data/README.md) for the schema, model sources, scoring definition, original dataset attribution and data license. The repeated 80/20 protocol differs from the original dataset's supervised train/test split.

To use another probability dataset with the same schema:

```bash
python main.py --data path/to/dataset.npz --protocol inductive --seed 2026
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests check the explicit KL objective, analytical gradient, conditional inference, frozen-state and batch invariance, input validation, and model serialization.

## License

Code is provided under the [MIT License](LICENSE). The response dataset is covered separately by [CC BY-SA 4.0](data/LICENSE.txt); source-data attribution is in [data/README.md](data/README.md).
