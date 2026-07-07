"""Market-regime detection with a Gaussian HMM (dependency-free EM implementation).

Fitted on (return, log-volatility) observations; the decoded state sequence is
fed to the other models as a categorical feature.
"""

from __future__ import annotations

import numpy as np


class GaussianHMMRegime:
    def __init__(self, n_states: int = 3, n_iter: int = 50, seed: int = 7, tol: float = 1e-4) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.seed = seed
        self.tol = tol
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None

    # diagonal-covariance Gaussian log-density per state
    def _log_b(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.sum(diff**2 / self.vars_[None], axis=2)
            + np.sum(np.log(2 * np.pi * self.vars_), axis=1)[None, :]
        )

    def fit(self, X: np.ndarray) -> GaussianHMMRegime:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 1:
            X = X.T
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        k = self.n_states

        # init: quantile-split means, global variance
        order = np.argsort(X[:, 0])
        splits = np.array_split(order, k)
        self.means_ = np.array([X[s].mean(axis=0) for s in splits])
        self.vars_ = np.tile(X.var(axis=0) + 1e-8, (k, 1))
        self.transmat_ = np.full((k, k), 0.1 / max(k - 1, 1)) + np.eye(k) * 0.9
        self.startprob_ = np.full(k, 1.0 / k)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            log_b = self._log_b(X)
            log_alpha, ll = self._forward(log_b)
            log_beta = self._backward(log_b)

            log_gamma = log_alpha + log_beta
            log_gamma -= log_gamma.max(axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)

            # xi accumulation in log space
            xi_sum = np.zeros((k, k))
            log_A = np.log(self.transmat_ + 1e-300)
            for t in range(n - 1):
                m = log_alpha[t][:, None] + log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :]
                m -= m.max()
                e = np.exp(m)
                xi_sum += e / e.sum()

            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / w[:, None]
            for s in range(k):
                diff = X - self.means_[s]
                self.vars_[s] = (gamma[:, s][:, None] * diff**2).sum(axis=0) / w[s] + 1e-8

            if abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                break
            prev_ll = ll
            _ = rng  # deterministic; rng kept for future stochastic restarts
        return self

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        n, k = log_b.shape
        log_A = np.log(self.transmat_ + 1e-300)
        log_alpha = np.zeros((n, k))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, n):
            log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
        return log_alpha, float(_logsumexp(log_alpha[-1], axis=0))

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        n, k = log_b.shape
        log_A = np.log(self.transmat_ + 1e-300)
        log_beta = np.zeros((n, k))
        for t in range(n - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decode: most likely state sequence."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] == 1:
            X = X.T
        log_b = self._log_b(X)
        n, k = log_b.shape
        log_A = np.log(self.transmat_ + 1e-300)
        delta = np.zeros((n, k))
        psi = np.zeros((n, k), dtype=int)
        delta[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, n):
            scores = delta[t - 1][:, None] + log_A
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + log_b[t]
        states = np.zeros(n, dtype=int)
        states[-1] = delta[-1].argmax()
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1][states[t + 1]]
        return states


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def regime_observations(returns: np.ndarray, vol_window: int = 20) -> np.ndarray:
    """Stack (return, rolling log-vol) into the HMM observation matrix."""
    r = np.asarray(returns, dtype=float)
    vol = np.full_like(r, np.nan)
    for i in range(vol_window, len(r)):
        vol[i] = np.std(r[i - vol_window : i]) + 1e-12
    obs = np.column_stack([r, np.log(vol)])
    obs = obs[vol_window:]
    return obs
