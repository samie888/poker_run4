"""CPU-friendly MLP binary classifier for Poker44 chunk-level bot detection.

This is a deep-learning base learner that consumes the same aligned 293-dim
chunk feature rows used by the LightGBM models, so it slots into a
:class:`poker44_ml.stacked.StackedEnsemble` (as a ``base_model``) and into the
canonical :class:`poker44_ml.inference.Poker44Model` loader unchanged.

Design notes:
* Inputs are standardized internally (mean/std fitted on the training rows);
  neural nets need this, unlike trees.
* Architecture is a configurable feed-forward stack
  ``Linear -> BatchNorm -> ReLU -> Dropout`` ending in a single logit.
* The module trains on CPU (this host has no CUDA) and is small enough that
  ~6k chunks x 293 features fit in seconds/epoch.
* ``__getstate__`` / ``__setstate__`` persist only the config, the
  standardizer stats, and the weight tensors as plain numpy, so the artifact
  pickles cleanly and always reloads on CPU.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class TorchMLPClassifier:
    """Feed-forward MLP exposing a scikit-style ``predict_proba``.

    Parameters mirror the knobs we want to sweep in the hyperparameter search:
    layer widths, dropout, learning rate, weight decay, epochs and batch size.
    """

    def __init__(
        self,
        *,
        hidden_sizes: Sequence[int] = (512, 256, 128),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 120,
        batch_size: int = 256,
        patience: int = 15,
        class_weight: bool = True,
        seed: int = 17,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.class_weight = bool(class_weight)
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        self.verbose = bool(verbose)

        self.input_dim: int | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._module: Any = None
        self.best_val_ap_: float | None = None
        self.best_epoch_: int | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        device = str(device or "cpu")
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def _build_module(self, input_dim: int):
        import torch.nn as nn

        layers: list[Any] = []
        prev = input_dim
        for width in self.hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mean) / self._std

    @staticmethod
    def _as_matrix(x: Any) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "TorchMLPClassifier":
        import torch
        from sklearn.metrics import average_precision_score

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = self._as_matrix(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.input_dim = int(X.shape[1])

        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self._std = std

        Xs = self._standardize(X).astype(np.float32)
        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = self._standardize(self._as_matrix(X_val)).astype(np.float32)
            yv = np.asarray(y_val, dtype=np.float64).reshape(-1)
            Xv_t = torch.from_numpy(Xv).to(torch.device(self.device))
            yv_t = torch.from_numpy(yv.astype(np.float32)).reshape(-1, 1).to(
                torch.device(self.device)
            )

        device = torch.device(self.device)
        module = self._build_module(self.input_dim).to(device)

        pos_weight = None
        if self.class_weight:
            n_pos = float((y == 1).sum())
            n_neg = float((y == 0).sum())
            if n_pos > 0:
                pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.n_epochs)
        )

        X_t = torch.from_numpy(Xs)
        y_t = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
        n = X_t.shape[0]
        rng = np.random.default_rng(self.seed)

        best_state = None
        # Early stopping/best-weights are selected on validation LOSS (lower is
        # better), not validation AP. On the benchmark val set the classes are
        # trivially separable, so AP pegs at ~1.0 from epoch 0; selecting on AP
        # froze the net at near-untrained weights that collapse to flat scores
        # on out-of-distribution inputs. Val loss keeps improving as the
        # net actually learns, yielding a model that spreads on live data.
        best_loss = float("inf")
        best_ap = 0.0
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(self.n_epochs):
            module.train()
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if idx.size < 2:
                    continue  # BatchNorm needs >=2 rows
                xb = X_t[idx].to(device)
                yb = y_t[idx].to(device)
                optimizer.zero_grad()
                logits = module(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            if has_val:
                module.eval()
                with torch.no_grad():
                    v_logits_t = module(Xv_t)
                    v_loss = float(loss_fn(v_logits_t, yv_t).item())
                    v_logits = v_logits_t.cpu().numpy().reshape(-1)
                v_prob = 1.0 / (1.0 + np.exp(-np.clip(v_logits, -40, 40)))
                ap = float(average_precision_score(yv, v_prob)) if len(np.unique(yv)) > 1 else 0.0
                if v_loss < best_loss - 1e-5:
                    best_loss = v_loss
                    best_ap = ap
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                    print(f"    epoch {epoch:3d}  val_loss={v_loss:.4f}  val_AP={ap:.4f}  best_loss={best_loss:.4f}")
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"    early stop at epoch {epoch} (best epoch {best_epoch})")
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
            self.best_val_ap_ = best_ap
            self.best_epoch_ = best_epoch
        module.eval()
        self._module = module.cpu()
        self.device = "cpu"
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        import torch

        if self._module is None:
            raise RuntimeError("TorchMLPClassifier is not fitted.")
        x = self._standardize(self._as_matrix(X)).astype(np.float32)
        self._module.eval()
        with torch.no_grad():
            logits = self._module(torch.from_numpy(x)).cpu().numpy().reshape(-1)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        p1 = np.clip(p1, 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self) -> dict[str, Any]:
        state_dict = None
        if self._module is not None:
            state_dict = {
                k: v.detach().cpu().numpy() for k, v in self._module.state_dict().items()
            }
        return {
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "class_weight": self.class_weight,
            "seed": self.seed,
            "verbose": self.verbose,
            "input_dim": self.input_dim,
            "mean": None if self._mean is None else np.asarray(self._mean),
            "std": None if self._std is None else np.asarray(self._std),
            "state_dict": state_dict,
            "best_val_ap_": self.best_val_ap_,
            "best_epoch_": self.best_epoch_,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.hidden_sizes = tuple(state["hidden_sizes"])
        self.dropout = float(state["dropout"])
        self.lr = float(state["lr"])
        self.weight_decay = float(state["weight_decay"])
        self.n_epochs = int(state["n_epochs"])
        self.batch_size = int(state["batch_size"])
        self.patience = int(state["patience"])
        self.class_weight = bool(state["class_weight"])
        self.seed = int(state["seed"])
        self.verbose = bool(state.get("verbose", False))
        self.device = "cpu"
        self.input_dim = state.get("input_dim")
        self._mean = state.get("mean")
        self._std = state.get("std")
        self.best_val_ap_ = state.get("best_val_ap_")
        self.best_epoch_ = state.get("best_epoch_")
        self._module = None
        state_dict = state.get("state_dict")
        if state_dict is not None and self.input_dim is not None:
            import torch

            module = self._build_module(self.input_dim)
            module.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state_dict.items()})
            module.eval()
            self._module = module.cpu()
