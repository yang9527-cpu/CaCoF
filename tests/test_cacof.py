"""Numerical checks for the objective, derivative and frozen prediction rule."""

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import softmax

from cacof import CaCoF, _correct, _evaluate, _feedback, _normalize, _prototypes


class CaCoFTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(91)
        self.p = self.rng.dirichlet(np.ones(5), size=(29, 3))

    def test_profiled_objective_matches_explicit_kl(self):
        b = self.rng.dirichlet(np.ones(5), size=len(self.p))
        corrected = _correct(self.p, self.p.mean(axis=0))
        reference = corrected.mean(axis=(0, 1))
        rho, tau, gamma = 0.2, 7.0, -0.4
        anchor = rho / (1 - rho)
        v = _feedback(corrected, b.mean(axis=0), reference, gamma)
        c = _prototypes(b, v, anchor)
        q = v.mean(axis=1)
        direct = (b * np.log(b / q)).sum() / len(b)
        conditional = 0.0
        for i in range(len(b)):
            for m in range(v.shape[1]):
                for k in range(b.shape[1]):
                    distance = (1 - rho) * np.sum(v[i, m] * np.log(v[i, m] / c[m, k]))
                    distance -= rho * np.log(c[m, k, k])
                    conditional += b[i, k] * distance
        explicit = direct + tau * conditional / (len(b) * v.shape[1])
        profiled = _evaluate(b, corrected, reference, tau * (1 - rho), anchor, gamma, gradient=False)
        self.assertAlmostEqual(explicit, profiled, places=12)
        expected = (1 - rho) * np.stack([b.T @ v[:, m] for m in range(3)]) / b.sum(0)[None, :, None]
        expected += rho * np.eye(5)[None]
        np.testing.assert_allclose(c, expected, rtol=0, atol=1e-15)

    def test_feedback_gradient_by_simplex_directional_differences(self):
        b = self.rng.dirichlet(np.ones(5), size=len(self.p))
        corrected = _correct(self.p, self.p.mean(axis=0))
        reference = corrected.mean(axis=(0, 1))
        args = (corrected, reference, 6.4, 0.1 / 0.9, -0.5)
        _, derivative = _evaluate(b, *args)
        for _ in range(6):
            direction = self.rng.normal(size=b.shape)
            direction -= direction.mean(axis=1, keepdims=True)
            direction /= np.linalg.norm(direction)
            h = 1e-6
            numeric = (
                _evaluate(b + h * direction, *args, gradient=False)
                - _evaluate(b - h * direction, *args, gradient=False)
            ) / (2 * h)
            analytic = np.sum(derivative * direction) / len(b)
            self.assertAlmostEqual(numeric, analytic, delta=2e-8)

    def test_frozen_prediction_matches_conditional_kl_solution(self):
        model = CaCoF().fit(self.p[:21])
        query = self.p[21:]
        v = _feedback(
            _correct(_normalize(query, ndim=3), model.marginals_),
            model.class_proportions_, model.reference_proportions_, model.gamma,
        )
        q = v.mean(axis=1)
        distances = np.zeros_like(q)
        for i in range(len(query)):
            for m in range(model.n_sources_):
                for k in range(model.n_classes_):
                    c = model.prototypes_[m, k]
                    distances[i, k] += (1 - model.rho) * np.sum(v[i, m] * np.log(v[i, m] / c))
                    distances[i, k] -= model.rho * np.log(c[k])
        expected = softmax(np.log(q) - model.tau * distances / model.n_sources_, axis=1)
        np.testing.assert_allclose(model.predict_proba(query), expected, rtol=0, atol=2e-14)

    def test_frozen_state_input_and_batch_invariance(self):
        original = self.p.copy()
        model = CaCoF().fit(self.p[:21])
        attributes = ["marginals_", "reference_proportions_", "class_proportions_", "prototypes_", "memberships_"]
        before = {name: getattr(model, name).copy() for name in attributes}
        query = self.p[21:]
        batch = model.predict_proba(query)
        individual = np.concatenate([model.predict_proba(row[None]) for row in query])
        np.testing.assert_allclose(batch, individual, rtol=0, atol=3e-14)
        np.testing.assert_allclose(model.predict_proba(query[::-1]), batch[::-1], rtol=0, atol=3e-14)
        np.testing.assert_array_equal(self.p, original)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(model, name), values)
        np.testing.assert_array_equal(model.predict(query), batch.argmax(axis=1))
        np.testing.assert_allclose(batch.sum(axis=1), 1, rtol=0, atol=1e-14)
        self.assertTrue(np.all(np.diff(model.objective_history_) <= 0))
        self.assertEqual(set(inspect.signature(CaCoF.fit).parameters), {"self", "probabilities"})

    def test_native_output_and_saved_state_roundtrip(self):
        model = CaCoF()
        native = model.fit_predict(self.p)
        np.testing.assert_array_equal(native, model.memberships_)
        native[0] = 0
        self.assertGreater(model.memberships_[0].sum(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            restored = CaCoF.load(path)
        self.assertEqual(model.get_params(), restored.get_params())
        np.testing.assert_array_equal(model.memberships_, restored.memberships_)
        np.testing.assert_array_equal(model.predict_proba(self.p[:5]), restored.predict_proba(self.p[:5]))

    def test_zeros_and_empty_class_coordinate(self):
        p = self.p.copy()
        p[:, :, -1] = 0
        p[0, 0, 0] = 0
        model = CaCoF(warmup_steps=3, max_iter=2).fit(p)
        self.assertEqual(model.memberships_.shape, (len(p), 5))
        self.assertTrue(np.isfinite(model.prototypes_).all())
        self.assertTrue(np.isfinite(model.predict_proba(p[:2])).all())

    def test_invalid_inputs_and_unfitted_use(self):
        for p in [np.ones((2, 3)), np.zeros((2, 3, 4)), np.ones((0, 3, 4)), np.full((2, 3, 4), np.nan), -self.p]:
            with self.subTest(shape=p.shape), self.assertRaises(ValueError):
                CaCoF().fit(p)
        for parameters in [{"rho": 1}, {"rho": -0.1}, {"tau": 0}, {"gamma": np.inf}, {"step_size": 0}, {"tol": -1}, {"max_iter": 1.5}, {"warmup_steps": -1}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                CaCoF(**parameters)
        with self.assertRaises(RuntimeError):
            CaCoF().predict_proba(self.p)
        model = CaCoF(max_iter=0, warmup_steps=0).fit(self.p)
        with self.assertRaises(ValueError):
            model.predict_proba(np.ones((2, 2, 5)))


if __name__ == "__main__":
    unittest.main()
