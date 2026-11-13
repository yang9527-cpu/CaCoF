"""Semantic-anchored fuzzy consensus for aligned VLM probabilities.

Input arrays have shape (images, sources, named classes). Fitting uses only
unlabeled probabilities. ``fit_predict`` returns the memberships optimized on
that collection; ``predict_proba`` uses frozen fitted statistics for new images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax


EPS = 1e-12


def _normalize(values: np.ndarray, *, ndim: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != ndim or 0 in values.shape:
        raise ValueError(f"Expected a nonempty {ndim}-dimensional probability array.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Probabilities must be finite and nonnegative.")
    mass = values.sum(axis=-1, keepdims=True)
    if not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("Every probability row must have finite positive mass.")
    values = np.maximum(values, EPS)
    return values / values.sum(axis=-1, keepdims=True)


def _correct(probabilities: np.ndarray, marginals: np.ndarray) -> np.ndarray:
    corrected = probabilities / marginals[None, :, :]
    return corrected / corrected.sum(axis=-1, keepdims=True)


def _feedback(
    corrected: np.ndarray,
    proportions: np.ndarray,
    reference: np.ndarray,
    gamma: float,
) -> np.ndarray:
    responses = corrected * np.exp(gamma * np.log(proportions / reference))[None, None, :]
    return responses / responses.sum(axis=-1, keepdims=True)


def _prototypes(memberships: np.ndarray, responses: np.ndarray, anchor: float) -> np.ndarray:
    occupancy = memberships.sum(axis=0)
    counts = np.stack(
        [memberships.T @ responses[:, source, :] for source in range(responses.shape[1])]
    )
    counts += anchor * occupancy[None, :, None] * np.eye(responses.shape[-1])[None, :, :]
    return counts / counts.sum(axis=-1, keepdims=True)


def _evaluate(
    memberships: np.ndarray,
    corrected: np.ndarray,
    reference: np.ndarray,
    conditional_weight: float,
    anchor: float,
    gamma: float,
    *,
    gradient: bool = True,
) -> float | tuple[float, np.ndarray]:
    """Profiled objective and N-scaled gradient, up to row constants."""
    n_images, n_sources, _ = corrected.shape
    memberships = _normalize(memberships, ndim=2)
    proportions = memberships.mean(axis=0)
    responses = _feedback(corrected, proportions, reference, gamma)
    direct = responses.mean(axis=1)
    occupancy = memberships.sum(axis=0)
    profiles = _prototypes(memberships, responses, anchor)
    log_profiles = np.log(np.maximum(profiles, EPS))
    log_responses = np.log(np.maximum(responses, EPS))

    fidelity = (memberships * (np.log(memberships) - np.log(direct))).sum()
    profile_entropy = -(profiles * log_profiles).sum(axis=-1)
    conditional = (1 + anchor) * np.einsum("k,mk->", occupancy, profile_entropy)
    background = -(responses * log_responses).sum(axis=-1).sum()
    objective = float((fidelity + (conditional_weight / n_sources) * (conditional - background)) / n_images)
    if not gradient:
        return objective

    cross = np.stack(
        [responses[:, source] @ log_profiles[source].T for source in range(n_sources)], axis=1
    )
    diagonal = np.diagonal(log_profiles, axis1=1, axis2=2)
    local = np.log(memberships) - np.log(direct) + (conditional_weight / n_sources) * (
        -cross.sum(axis=1) - anchor * diagonal.sum(axis=0)[None, :]
    )
    weighted_profiles = np.stack(
        [memberships @ log_profiles[source] for source in range(n_sources)], axis=1
    )
    response_gradient = -(memberships / direct)[:, None, :] / n_sources + (
        conditional_weight / n_sources
    ) * (log_responses + 1 - weighted_profiles)
    row_mean = (responses * response_gradient).sum(axis=-1)
    feedback_gradient = (gamma / proportions) * (
        responses * (response_gradient - row_mean[:, :, None])
    ).sum(axis=(0, 1)) / n_images
    return objective, local + feedback_gradient[None, :]


class CaCoF:
    """Learn source-specific class-response profiles without reference labels.

    Parameters follow the paper: ``rho`` is the class-anchor weight, ``tau``
    the conditional-response weight, and ``gamma`` the proportion feedback
    exponent. ``warmup_steps`` uses the same conditional model with feedback
    disabled before the profiled optimization starts. All calculations use
    float64 and the deterministic corrected-mean initialization.

    ``fit_predict(P)`` returns probabilities for the fitted collection. Calling
    ``predict_proba(P)`` after fitting instead applies the conditional rule with
    frozen profiles and proportions, even if P contains the training images.
    These two outputs need not be equal when feedback is enabled.
    """

    def __init__(
        self,
        rho: float = 0.1,
        tau: float = 64.0,
        gamma: float = -0.5,
        warmup_steps: int = 30,
        max_iter: int = 35,
        step_size: float = 0.8,
        tol: float = 1e-9,
    ) -> None:
        parameters = np.asarray([rho, tau, gamma, step_size, tol], dtype=np.float64)
        if not np.isfinite(parameters).all():
            raise ValueError("Model parameters must be finite.")
        if not 0 <= rho < 1:
            raise ValueError("rho must satisfy 0 <= rho < 1.")
        if tau <= 0 or step_size <= 0 or tol < 0:
            raise ValueError("tau and step_size must be positive; tol must be nonnegative.")
        for name, value in (("warmup_steps", warmup_steps), ("max_iter", max_iter)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        self.rho = float(rho)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.warmup_steps = int(warmup_steps)
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.tol = float(tol)

    def get_params(self) -> dict[str, float | int]:
        return {
            "rho": self.rho,
            "tau": self.tau,
            "gamma": self.gamma,
            "warmup_steps": self.warmup_steps,
            "max_iter": self.max_iter,
            "step_size": self.step_size,
            "tol": self.tol,
        }

    def fit(self, probabilities: np.ndarray) -> "CaCoF":
        """Fit the unlabeled collection P[N,M,K], preserving class order."""
        probabilities = _normalize(probabilities, ndim=3)
        n_images, n_sources, n_classes = probabilities.shape
        marginals = probabilities.mean(axis=0)
        corrected = _correct(probabilities, marginals)
        reference = corrected.mean(axis=(0, 1))
        direct = corrected.mean(axis=1)
        memberships = _normalize(direct.copy(), ndim=2)
        anchor = self.rho / (1 - self.rho)
        conditional_weight = self.tau * (1 - self.rho)

        for _ in range(self.warmup_steps):
            profiles = _prototypes(memberships, corrected, anchor)
            score = np.zeros((n_images, n_classes), dtype=np.float64)
            for source in range(n_sources):
                log_profiles = np.log(np.maximum(profiles[source], EPS))
                score += corrected[:, source, :] @ log_profiles.T
                score += anchor * np.log(np.maximum(np.diag(profiles[source]), EPS))[None, :]
            updated = _normalize(
                softmax(np.log(np.maximum(direct, EPS)) + (conditional_weight / n_sources) * score, axis=1),
                ndim=2,
            )
            change = np.max(np.abs(updated - memberships))
            memberships = updated
            if change < 1e-7:
                break

        args = (corrected, reference, conditional_weight, anchor, self.gamma)
        objective, gradient = _evaluate(memberships, *args)
        history = [objective]
        for _ in range(self.max_iter):
            centered = gradient - gradient.mean(axis=1, keepdims=True)
            step = self.step_size
            accepted = False
            for _ in range(20):
                logits = np.log(np.maximum(memberships, EPS)) - step * centered
                candidate = _normalize(np.exp(logits - logsumexp(logits, axis=1, keepdims=True)), ndim=2)
                candidate_objective = _evaluate(candidate, *args, gradient=False)
                if candidate_objective <= objective - 1e-12:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            memberships = candidate
            previous = objective
            objective, gradient = _evaluate(memberships, *args)
            history.append(objective)
            if previous - objective < self.tol:
                break

        self.memberships_ = memberships
        self.objective_history_ = np.asarray(history, dtype=np.float64)
        self.marginals_ = marginals
        self.reference_proportions_ = reference
        self.class_proportions_ = memberships.mean(axis=0)
        self.prototypes_ = _prototypes(
            memberships, _feedback(corrected, self.class_proportions_, reference, self.gamma), anchor
        )
        self.n_sources_ = n_sources
        self.n_classes_ = n_classes
        self.n_iter_ = len(history) - 1
        return self

    def fit_predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return the fitted collection's native membership probabilities."""
        return self.fit(probabilities).memberships_.copy()

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Predict each query independently using frozen fitting statistics."""
        self._require_fitted()
        probabilities = _normalize(probabilities, ndim=3)
        if probabilities.shape[1:] != (self.n_sources_, self.n_classes_):
            raise ValueError("Prediction source count and class dictionary must match fitting.")
        corrected = _correct(probabilities, self.marginals_)
        responses = _feedback(corrected, self.class_proportions_, self.reference_proportions_, self.gamma)
        scores = np.log(np.maximum(responses.mean(axis=1), EPS))
        anchor = self.rho / (1 - self.rho)
        conditional_weight = self.tau * (1 - self.rho)
        for source in range(self.n_sources_):
            log_profiles = np.log(np.maximum(self.prototypes_[source], EPS))
            scores += (conditional_weight / self.n_sources_) * (
                responses[:, source, :] @ log_profiles.T + anchor * np.diag(log_profiles)[None, :]
            )
        return softmax(scores, axis=1)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Return zero-based class indices from frozen conditional prediction."""
        return self.predict_proba(probabilities).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        """Save parameters and fitted arrays to a portable NPZ file."""
        self._require_fitted()
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream,
                parameters=np.asarray(json.dumps(self.get_params(), sort_keys=True)),
                memberships=self.memberships_,
                objective_history=self.objective_history_,
                marginals=self.marginals_,
                reference_proportions=self.reference_proportions_,
                class_proportions=self.class_proportions_,
                prototypes=self.prototypes_,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CaCoF":
        """Load a fitted model without unpickling Python objects."""
        with np.load(path, allow_pickle=False) as arrays:
            model = cls(**json.loads(str(arrays["parameters"].item())))
            state = {
                name: np.asarray(arrays[name], dtype=np.float64).copy()
                for name in (
                    "memberships", "objective_history", "marginals", "reference_proportions",
                    "class_proportions", "prototypes",
                )
            }
        if any(not np.isfinite(value).all() for value in state.values()):
            raise ValueError("Saved model arrays must be finite.")
        profiles = state["prototypes"]
        if profiles.ndim != 3 or 0 in profiles.shape or profiles.shape[1] != profiles.shape[2]:
            raise ValueError("Saved prototypes must have shape [sources, classes, classes].")
        n_sources, n_classes, _ = profiles.shape
        if state["marginals"].shape != (n_sources, n_classes):
            raise ValueError("Saved marginals do not match the prototypes.")
        for name in ("reference_proportions", "class_proportions"):
            if state[name].shape != (n_classes,):
                raise ValueError(f"Saved {name} does not match the class dictionary.")
        memberships = state["memberships"]
        if memberships.ndim != 2 or memberships.shape[0] == 0 or memberships.shape[1] != n_classes:
            raise ValueError("Saved memberships do not match the class dictionary.")
        history = state["objective_history"]
        if history.ndim != 1 or len(history) == 0 or len(history) > model.max_iter + 1:
            raise ValueError("Saved objective history does not match the iteration budget.")
        for name, values in state.items():
            if name == "objective_history":
                continue
            if np.any(values <= 0):
                raise ValueError(f"Saved {name} must be strictly positive.")
            if not np.allclose(values.sum(axis=-1), 1, rtol=0, atol=1e-9):
                raise ValueError(f"Saved {name} must contain normalized probability rows.")
        for name, value in state.items():
            setattr(model, name + "_", value)
        model.n_sources_ = n_sources
        model.n_classes_ = n_classes
        model.n_iter_ = len(history) - 1
        return model

    def _require_fitted(self) -> None:
        if not hasattr(self, "prototypes_"):
            raise RuntimeError("Call fit() before prediction or saving.")
