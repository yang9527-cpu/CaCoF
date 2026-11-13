"""Run CaCoF on cached VLM probabilities with the supplied evaluation splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from cacof import CaCoF, EPS


def load_inputs(path: Path) -> dict[str, np.ndarray]:
    """Read observations and protocol metadata without loading reference labels."""
    fields = ("probabilities", "sample_ids", "class_names", "model_ids",
              "model_revisions", "seeds", "train_indices", "test_indices")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise ValueError(f"Missing dataset fields: {', '.join(sorted(missing))}")
        data = {name: archive[name] for name in fields}
    p = data["probabilities"]
    if p.ndim != 3 or 0 in p.shape or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Expected finite, nonnegative probabilities[N,M,K].")
    if np.any(p.sum(axis=-1) <= 0):
        raise ValueError("Every source response must have positive mass.")
    n, m, k = p.shape
    if data["sample_ids"].shape != (n,) or len(np.unique(data["sample_ids"])) != n:
        raise ValueError("Expected one distinct sample ID per image.")
    for name, size in (("class_names", k), ("model_ids", m), ("model_revisions", m)):
        if data[name].shape != (size,):
            raise ValueError(f"{name} does not match the probability coordinates.")
    seeds = data["seeds"]
    if seeds.ndim != 1 or seeds.dtype.kind not in "iu" or len(np.unique(seeds)) != len(seeds):
        raise ValueError("Expected distinct integer split seeds.")
    for name in ("train_indices", "test_indices"):
        ids = data[name]
        if ids.ndim != 2 or ids.shape[0] != len(seeds) or ids.shape[1] == 0 or ids.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain one nonempty integer row per seed.")
    for train, test in zip(data["train_indices"], data["test_indices"]):
        if not np.array_equal(np.sort(np.concatenate((train, test))), np.arange(n)):
            raise ValueError("Each saved split must partition every image exactly once.")
    return data


def evaluate(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Named-class ACC/F1 in percent and unscaled multiclass Brier."""
    labels = np.asarray(labels)
    q = np.asarray(probabilities, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Predictions must be finite nonnegative probability rows.")
    n, k = q.shape
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= k)):
        raise ValueError("Evaluation labels must use the original zero-based class indices.")
    labels = labels.astype(np.int64, copy=False)
    if not np.allclose(q.sum(1), 1, rtol=0, atol=1e-9):
        raise ValueError("Prediction rows must sum to one.")
    prediction = q.argmax(1)
    counts = np.bincount(labels * k + prediction, minlength=k * k).reshape(k, k)
    denominator = counts.sum(0) + counts.sum(1)
    f1 = np.divide(2.0 * np.diag(counts), denominator,
                   out=np.zeros(k, dtype=np.float64), where=denominator != 0)
    error = q.copy()
    error[np.arange(n), labels] -= 1
    return {"ACC": float(100 * np.mean(prediction == labels)),
            "MacroF1": float(100 * np.mean(f1)),
            "Brier": float(np.mean(np.sum(error * error, axis=1)))}


def corrected_mean(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CalMean control using only the fitting pool's source-class marginals."""
    train = np.maximum(np.asarray(train, dtype=np.float64), EPS)
    train /= train.sum(axis=-1, keepdims=True)
    marginals = train.mean(axis=0)
    query = np.maximum(np.asarray(query, dtype=np.float64), EPS)
    query /= query.sum(axis=-1, keepdims=True)
    corrected = query / marginals[None, :, :]
    corrected /= corrected.sum(axis=-1, keepdims=True)
    return corrected.mean(axis=1), marginals


def run(args: argparse.Namespace) -> None:
    data = load_inputs(args.data)
    p = data["probabilities"]
    seeds = data["seeds"].tolist() if args.seed == "all" else [int(args.seed)]
    if not set(seeds) <= set(data["seeds"].tolist()):
        raise ValueError(f"Available saved seeds: {data['seeds'].tolist()}")
    protocols = ["inductive", "transductive"] if args.protocol == "both" else [args.protocol]
    methods = ["cacof", "calmean"] if args.method == "both" else [args.method]
    with args.data.open("rb") as stream:
        data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    records = []
    for protocol in protocols:
        for seed in seeds:
            position = int(np.flatnonzero(data["seeds"] == seed)[0])
            if protocol == "inductive":
                train, test = data["train_indices"][position], data["test_indices"][position]
            else:
                train = test = np.arange(len(p), dtype=np.int64)
            for method in methods:
                folder = args.output / protocol / str(seed) / method
                folder.mkdir(parents=True, exist_ok=True)
                start = perf_counter()
                if method == "cacof":
                    model = CaCoF(rho=args.rho, tau=args.tau, gamma=args.gamma).fit(p[train])
                    q = model.memberships_.copy() if protocol == "transductive" else model.predict_proba(p[test])
                    model.save(folder / "model.npz")
                    parameters = model.get_params()
                    iterations = model.n_iter_
                else:
                    q, marginals = corrected_mean(p[train], p[test])
                    np.savez_compressed(folder / "model.npz", marginals=marginals)
                    parameters, iterations = {}, 0
                seconds = perf_counter() - start
                # Persist predictions before the first access to reference labels.
                np.savez_compressed(folder / "predictions.npz", probabilities=q,
                                    predictions=q.argmax(1), train_indices=train, test_indices=test,
                                    sample_ids=data["sample_ids"][test], class_names=data["class_names"],
                                    model_ids=data["model_ids"], model_revisions=data["model_revisions"])
                with np.load(args.data, allow_pickle=False) as archive:
                    metrics = evaluate(archive["labels"][test], q)
                record = {"method": method, "protocol": protocol, "seed": seed,
                          "n_fit": len(train), "n_evaluate": len(test), **metrics,
                          "iterations": iterations, "seconds": seconds}
                detail = {**record, "parameters": parameters, "data_sha256": data_hash}
                (folder / "metrics.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
                records.append(record)
                print(f"{method:7s} {protocol:12s} seed={seed} "
                      f"ACC={metrics['ACC']:.4f} F1={metrics['MacroF1']:.4f} "
                      f"Brier={metrics['Brier']:.6f} time={seconds:.2f}s", flush=True)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved predictions, fitted state and metrics to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/oxford_pets.npz")
    parser.add_argument("--protocol", choices=("inductive", "transductive", "both"), default="inductive")
    parser.add_argument("--seed", default="2026", help="Saved split seed, or all.")
    parser.add_argument("--method", choices=("cacof", "calmean", "both"), default="cacof")
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=64.0)
    parser.add_argument("--gamma", type=float, default=-0.5)
    parser.add_argument("--threads", type=int, default=1, help="BLAS threads used during this run.")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.seed != "all":
        try:
            int(args.seed)
        except ValueError:
            parser.error("--seed must be an integer or all")
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
