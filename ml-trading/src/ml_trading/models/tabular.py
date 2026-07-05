"""Tabular baseline: LightGBM (default) / CatBoost (optional extra), tuned with Optuna
under purged walk-forward CV. This is the bar every deep model must clear."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import f1_score

from ml_trading.validation.purged_cv import PurgedWalkForward

try:
    from catboost import CatBoostClassifier
except ImportError:  # optional extra
    CatBoostClassifier = None


@dataclass
class TabularModel:
    """Multiclass (-1/0/+1 mapped to 0/1/2) gradient-boosted classifier."""

    backend: str = "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, repr=False)

    LABEL_OFFSET = 1  # maps {-1,0,1} -> {0,1,2}

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabularModel:
        y_enc = (y + self.LABEL_OFFSET).astype(int)
        if self.backend == "lightgbm":
            defaults = dict(
                objective="multiclass",
                num_class=3,
                learning_rate=0.05,
                n_estimators=300,
                num_leaves=63,
                min_child_samples=50,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.8,
                verbosity=-1,
            )
            self._model = lgb.LGBMClassifier(**{**defaults, **self.params})
        elif self.backend == "catboost":
            if CatBoostClassifier is None:
                raise RuntimeError("catboost not installed; run: uv pip install 'ml-trading[catboost]'")
            defaults = dict(loss_function="MultiClass", iterations=300, verbose=False)
            self._model = CatBoostClassifier(**{**defaults, **self.params})
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        self._model.fit(X, y_enc)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Columns ordered as [P(stop), P(timeout), P(profit)]."""
        return np.asarray(self._model.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1) - self.LABEL_OFFSET

    def feature_importances(self) -> np.ndarray:
        return np.asarray(self._model.feature_importances_, dtype=float)


def cv_score(
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    label_end_idx: np.ndarray,
    cv: PurgedWalkForward,
    backend: str = "lightgbm",
) -> float:
    """Mean macro-F1 across purged walk-forward folds."""
    scores = []
    for train_idx, test_idx in cv.split(len(y), label_end_idx):
        m = TabularModel(backend=backend, params=model_params).fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        scores.append(f1_score(y[test_idx], pred, average="macro"))
    return float(np.mean(scores)) if scores else 0.0


def tune_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    label_end_idx: np.ndarray,
    cv: PurgedWalkForward,
    n_trials: int = 50,
    seed: int = 7,
) -> optuna.Study:
    """Optuna search over LightGBM hyperparameters, scored with purged CV only."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return cv_score(params, X, y, label_end_idx, cv)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
