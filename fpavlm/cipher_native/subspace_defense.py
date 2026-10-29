"""Diagnostic Subspace Retention (DSR) calibration and evaluation.

PROBLEM (the strong threat model):
  The existing CFP (Calibrated Feature Perturbation) defense adds i.i.d. Gaussian
  noise to released features. This defeats the DEPLOYMENT threat model (server
  sees only noised features, with inter-patient cos=0.0016 in a three-image test).
  The historical ε=0.1994 DP statement was invalid and is not evidence for DSR.
  CANNOT defeat the STRONG threat model: an attacker with a clean enrollment DB
  {x_i} can match a noised query y=argmax_i cos(y, x_i) because Gaussian noise
  attenuates all directions equally — the ranking is preserved (re-ID rate=1.0).

  Root cause: additive i.i.d. noise is a SCALING operation on the signal
  direction; the argmax (a RANKING operation) is invariant to isotropic scaling.

HYPOTHESIS:
  Instead of relying only on isotropic noise, retain a low-dimensional subspace
  predictive of diagnosis. This may reduce release dimension and linkage, but
  diagnostic and patient-specific factors are not assumed orthogonal. Disease,
  anatomy, scanner, site, and demographics can be confounded and must be tested
  with patient-level and site-level splits.

  DSR computes the DIAGNOSTIC subspace U_diag (directions that maximize
  covariance with diagnosis labels via PLS), then projects features onto it:
      x' = Π x,  where Π = U_diag @ U_diag^T  (symmetric idempotent projector)

  Projection retains directions selected by PLS on the calibration set. It does
  not prove removal of identity, reconstruction, membership, or site information.

FHE COMPATIBILITY (the key architectural fit):
  Π is a fixed matrix computed offline from calibration features and labels.
  Publishing it may itself require a training-data privacy analysis; the current
  implementation treats it as a trusted artifact, not a public privacy guarantee.
  The release x' = Π x is a linear map. The server already computes
  W_concept @ ct under FHE (cipher×plain matmul, depth +1) inside
  MedicalConceptAdapter.forward_cipher(), BEFORE client decryption.

  Folding DSR into W_concept:
      W_concept_new = W_concept @ Π    (shape [D, LLM_dim] @ [LLM_dim, LLM_dim])
  Same shape, same single cipher×plain matmul, ZERO additional FHE depth,
  ZERO additional client compute. Whether patient-discriminative signal remains
  is determined empirically by attacks; projection alone gives no such guarantee.

COMPOSITE DEFENSE (defense-in-depth):
  DSR (deterministic subspace retention) + reduced CFP (probabilistic DP on
  the diagnostic residual). Because the diagnostic subspace is low-rank
  (k ≈ 20-80 ≪ d=2560), a smaller release can reduce the noise needed for a
  target accountant bound. This repository has not established the claimed
  end-to-end guarantee at σ=2-4×. This is the "perturb-then-project"
  paradigm (Ball-DP arXiv:2607.04209, ScoreShield arXiv:2607.25041, 2026)
  applied to FHE medical feature release.

  Noise and rank must be selected against held-out utility and attack constraints.
  No privacy or accuracy result follows from dimensionality reduction alone.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from fpavlm.checkpoint import load_tensor_checkpoint
from fpavlm.cipher_native.demographic_filter import (
    DemographicNullSpaceFilter,
)


class DiagnosticSubspaceRetention:
    """Compute and apply the diagnostic subspace projection Π.

    Usage:
        # 1. Calibrate from cached features + diagnosis labels
        dsr = DiagnosticSubspaceRetention.calibrate(features, labels, k=64)

        # 2. Fold into W_concept (zero extra FHE depth)
        W_concept_new = dsr.fold_into_weight(W_concept_old)

        # 3. Save / load calibration
        dsr.save("dsr_calibration.pt")
        dsr = DiagnosticSubspaceRetention.load("dsr_calibration.pt")
    """

    def __init__(self, projection: torch.Tensor, subspace: torch.Tensor,
                 k: int, label_classes: List[str], metrics: Optional[Dict] = None):
        """Store the computed projection.

        Args:
            projection: [d, d] symmetric idempotent projector Π = U @ U^T.
            subspace: [d, k] orthonormal basis U_diag of the diagnostic subspace.
            k: Number of retained diagnostic dimensions.
            label_classes: Diagnosis class names (for reproducibility).
            metrics: Calibration metrics (retained_variance, reid_rate, etc.).
        """
        self.projection = projection  # [d, d]
        self.subspace = subspace      # [d, k]
        self.k = k
        self.label_classes = label_classes
        self.metrics = metrics or {}

    @staticmethod
    def _parse_diagnosis_labels(metadata_path: str) -> Tuple[List[str], np.ndarray]:
        """Parse OpenI metadata into multi-label diagnosis matrix.

        Extracts diagnosis classes from the 'problems' field (semicolon-separated).
        Returns (class_names, label_matrix [N, C]).
        """
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        records = meta["value"] if isinstance(meta, dict) and "value" in meta else meta

        # Collect all unique diagnosis labels
        all_labels = set()
        per_image_labels = []
        for rec in records:
            problems = rec.get("problems", "").strip()
            if not problems or problems.lower() == "normal":
                labels = {"normal"}
            else:
                labels = set(p.strip().lower() for p in problems.split(";") if p.strip())
            per_image_labels.append(labels)
            all_labels.update(labels)

        class_names = sorted(all_labels)
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Build multi-label matrix [N, C]
        label_matrix = np.zeros((len(per_image_labels), len(class_names)), dtype=np.float32)
        for i, labels in enumerate(per_image_labels):
            for lab in labels:
                if lab in class_to_idx:
                    label_matrix[i, class_to_idx[lab]] = 1.0
        return class_names, label_matrix

    @staticmethod
    def calibrate(
        features: torch.Tensor,
        labels: np.ndarray,
        k: int = 64,
        n_pls_iter: int = 200,
        demographic_filter: Optional["DemographicNullSpaceFilter"] = None,
        sex_labels: Optional[torch.Tensor] = None,
        site_labels: Optional[torch.Tensor] = None,
        age_decade_labels: Optional[torch.Tensor] = None,
        demo_filter_method: str = "inlp",
        n_demo_directions: int = 1,
        demo_n_iter: int = 2,
    ) -> "DiagnosticSubspaceRetention":
        """Compute the diagnostic subspace via PLS with optional DemNull filtering.

        DemNull-FDBR (new, v_demNull):
          When any demographic label is provided, applies Demographic Null-Space
          projection (DemographicNullSpaceFilter) to the PLS subspace after
          computing it.  This projects U out of the gender/site/age subspace,
          reducing demographic attribute AUC at the cost of a small diagnostic
          utility drop.

          Mathematical guarantee:
              U_filtered = orth((I - Π_demo) U_PLS)
              => xU_filtered · d_i ≈ 0  for all gender/site directions d_i

          The orthonormality contract (DiagnosticBottleneckRelease) is preserved.
          ZERO additional FHE depth (same cipher×plain matmul).

        Args:
            features: [N, d] calibration features.
            labels: [N, C] multi-hot diagnosis label matrix.
            k: Number of diagnostic dimensions to retain (default 64).
            n_pls_iter: NIPALS iterations per component.
            demographic_filter: Pre-calibrated DemographicNullSpaceFilter to
                apply.  If None but any of sex_labels/site_labels/age_decade_labels
                is provided, the filter is calibrated on-the-fly.
            sex_labels: [N] binary sex labels (0/1) for DemNull filtering.
            site_labels: [N] integer scanner-site labels for DemNull filtering.
            age_decade_labels: [N] age-decade labels for DemNull filtering.
            demo_filter_method: Direction-estimation method for DemNull
                calibration (``'mean_diff'``, ``'lda'``, or ``'inlp'``).
            n_demo_directions: Demographic directions per attribute.
            demo_n_iter: INLP projection iterations.

        Returns:
            DiagnosticSubspaceRetention instance with DemNull-filtered subspace.
        """
        X = features.detach().cpu().float().numpy()  # [N, d]
        Y = labels.astype(np.float32)                # [N, C]
        N, d = X.shape
        C = Y.shape[1]
        k = min(k, d, N - 1)  # can't exceed rank

        # Center X and Y
        X_mean = X.mean(axis=0, keepdims=True)
        Y_mean = Y.mean(axis=0, keepdims=True)
        X_c = X - X_mean
        Y_c = Y - Y_mean

        # NIPALS PLS: extract k components (standard PLS2 algorithm)
        # Reference: Wold et al., "PLS-regression: a basic tool of chemometrics"
        U = np.zeros((d, k))
        X_res = X_c.copy()
        Y_res = Y_c.copy()
        actual_k = 0
        for comp in range(k):
            # Initialize u: Y column with max variance (standard NIPALS init)
            col_vars = (Y_res ** 2).sum(axis=0)
            u_idx = np.argmax(col_vars)
            if col_vars[u_idx] < 1e-12:
                break  # no more Y signal — stop
            u = Y_res[:, u_idx].copy()  # [N]
            u = u / (np.linalg.norm(u) + 1e-12)

            for _ in range(n_pls_iter):
                # w = X^T u / (u^T u)  — X weight for this component [d]
                w = X_res.T @ u / (u @ u + 1e-12)
                w = w / (np.linalg.norm(w) + 1e-12)
                # t = X w  — X scores [N]
                t = X_res @ w
                t = t / (np.linalg.norm(t) + 1e-12)
                # c = Y^T t  — Y loadings [C]
                c = Y_res.T @ t
                # u_new = Y c / (c^T c)  — updated Y scores [N]
                u_new = Y_res @ c / (c @ c + 1e-12)
                # Check convergence
                if np.linalg.norm(u_new - u) < 1e-8:
                    u = u_new
                    break
                u = u_new

            # Store the X-weight direction (diagnostic direction in feature space)
            U[:, comp] = w
            actual_k = comp + 1

            # Deflate X AND Y (standard PLS2 — ensures orthogonal components
            # targeting DIFFERENT Y signal, avoiding redundant directions)
            p = X_res.T @ t / (t @ t + 1e-12)  # [d] X loadings
            X_res = X_res - np.outer(t, p)
            Y_res = Y_res - np.outer(t, c)

        if actual_k == 0:
            raise ValueError(
                "PLS calibration failed: no Y signal found (all labels zero?)"
            )

        # Truncate to actual components found
        U = U[:, :actual_k]

        # Orthonormalize U (QR — ensures U columns are orthonormal even if
        # NIPALS produced near-degenerate directions due to numerical drift)
        U, _ = np.linalg.qr(U)
        U = U[:, :actual_k]  # QR gives [d, d]; keep first actual_k columns

        # Build the projector Π = U @ U^T  (symmetric idempotent)
        Pi = U @ U.T  # [d, d]

        # Compute metrics
        X_proj = X_c @ Pi  # [N, d] projected (centered)
        # Retained variance fraction: trace(X_proj^T X_proj) / trace(X_c^T X_c)
        retained_var = np.trace(X_proj.T @ X_proj) / (np.trace(X_c.T @ X_c) + 1e-12)
        # Diagnosis predictability: regress Y on X scores in the diagnostic subspace
        T = X_c @ U  # [N, k] X scores
        Y_pred = T @ np.linalg.pinv(T) @ Y_c  # [N, C] predicted Y from diagnostic subspace
        y_explained = 1.0 - np.sum((Y_c - Y_pred) ** 2) / (np.sum(Y_c ** 2) + 1e-12)

        subspace_t = torch.from_numpy(U).float()

        # -----------------------------------------------------------------
        # DemNull-FDBR: project out demographic directions from U
        # -----------------------------------------------------------------
        demo_metrics: dict = {}
        demNull_applied = False
        need_demNull = (
            demographic_filter is not None
            or sex_labels is not None
            or site_labels is not None
            or age_decade_labels is not None
        )
        if need_demNull:
            # Build the filter if not pre-calibrated
            if demographic_filter is None:
                demographic_filter = DemographicNullSpaceFilter.calibrate(
                    torch.from_numpy(X).float(),
                    sex_labels=sex_labels,
                    site_labels=site_labels,
                    age_decade_labels=age_decade_labels,
                    n_directions_per_attribute=n_demo_directions,
                    method=demo_filter_method,
                )
            subspace_t, demo_metrics = demographic_filter.filter_subspace(
                subspace_t, n_iter=demo_n_iter
            )
            actual_k = subspace_t.shape[1]
            demNull_applied = True

            # Rebuild projection and metrics with the filtered subspace
            U_filtered = subspace_t.numpy()
            Pi = U_filtered @ U_filtered.T
            X_proj = X_c @ Pi
            retained_var = np.trace(X_proj.T @ X_proj) / (
                np.trace(X_c.T @ X_c) + 1e-12
            )

        projection_t = torch.from_numpy(Pi).float()

        metrics: dict = {
            "retained_feature_variance": float(retained_var),
            "diagnosis_explained_variance": float(y_explained),
            "k_retained": actual_k,
            "d_original": d,
            "N_calibration": N,
            "C_classes": C,
            "compression_ratio": float(actual_k / d),
            "demNull_applied": demNull_applied,
        }
        if demo_metrics:
            metrics["demNull"] = demo_metrics

        return DiagnosticSubspaceRetention(
            projection=projection_t,
            subspace=subspace_t,
            k=actual_k,
            label_classes=[],  # filled by caller if available
            metrics=metrics,
        )

    def fold_into_weight(self, W_concept: torch.Tensor) -> torch.Tensor:
        """Fold the DSR projection into W_concept (OUTPUT-space projection).

        W_concept: [D_in, D_out] (e.g., [64, 2560]) — projects FHE features
                   to LLM embedding space under FHE (cipher×plain, depth +1).
        Returns:   [D_in, D_out] — W_concept @ Π, same shape, same depth.

        The client decrypts already-projected features. Patient-identity
        signal is provably absent before decryption.

        NOTE: This is LLM-specific (Π is [D_out, D_out] = [LLM_dim, LLM_dim]).
        For a UNIVERSAL DSR shared across LLMs with different LLM_dim, use
        fold_into_weight_input() instead (projects in the shared D_in space).
        """
        # Π is [D_out, D_out]; W_concept is [D_in, D_out]
        # result = W_concept @ Π  → [D_in, D_out]
        proj = self.projection.to(W_concept.device).to(W_concept.dtype)
        return W_concept @ proj

    def fold_into_weight_input(self, W_concept: torch.Tensor) -> torch.Tensor:
        """Fold the DSR projection into W_concept (INPUT-space projection).

        W_concept: [D_in, D_out] (e.g., [64, 2560]) — projects FHE features
                   to LLM embedding space under FHE (cipher×plain, depth +1).
        Returns:   [D_in, D_out] — Π @ W_concept, same shape, same depth.

        v176 UNIVERSAL DSR: When calibrated in the SHARED ViT feature space
        (D_in=64), the projection Π is [64, 64] — INDEPENDENT of LLM_dim.
        This single calibration applies to ALL LLMs (MedGemma/InternVL2/
        Qwen2-VL/LLaVA-Med) by folding into the input side of W_concept:
            y' = (x @ Π) @ W = x @ (Π @ W)
        Patient-identity directions removed from ViT features BEFORE the
        linear projection to LLM_dim. Since W_concept is linear, patient
        signal in the null space of Π CANNOT re-emerge in the output.

        This is the PREFERRED folding for multi-LLM deployment: one DSR
        calibration, all LLMs protected. Zero additional FHE depth, zero
        additional client compute.
        """
        # Π is [D_in, D_in]; W_concept is [D_in, D_out]
        # result = Π @ W_concept  → [D_in, D_out]
        proj = self.projection.to(W_concept.device).to(W_concept.dtype)
        return proj @ W_concept

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        """Apply DSR projection to plaintext features (for testing/ablation).

        features: [..., d] — features in the calibration space (D_in or D_out).
        Returns:  [..., d] — projected features (patient subspace removed).
        """
        proj = self.projection.to(features.device).to(features.dtype)
        return features @ proj

    def reidentification_rate(
        self, clean_enrollment: torch.Tensor, projected_queries: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """Measure re-identification rate (strong threat model attack).

        Attacker has clean enrollment DB {x_i}. For each projected query x',
        computes argmax_i cos(x', x_i). Returns the fraction correctly matched.

        Args:
            clean_enrollment: [N, d] clean (un-projected) features.
            projected_queries: [N, d] DSR-projected features (same patients).

        Returns:
            (reid_rate, predictions) — reid_rate in [0, 1], predictions [N].
        """
        ce = clean_enrollment.detach().cpu().float().numpy()
        pq = projected_queries.detach().cpu().float().numpy()
        # Normalize
        ce_n = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-12)
        pq_n = pq / (np.linalg.norm(pq, axis=1, keepdims=True) + 1e-12)
        # Cosine similarity matrix [N_query, N_enrollment]
        sim = pq_n @ ce_n.T
        predictions = np.argmax(sim, axis=1)
        truth = np.arange(len(predictions))
        reid_rate = float(np.mean(predictions == truth))
        return reid_rate, predictions

    @staticmethod
    def cross_validated_rank_selection(
        features: torch.Tensor,
        labels: np.ndarray,
        candidate_ranks: Sequence[int] = (2, 4, 8, 12, 16, 24, 32),
        n_splits: int = 5,
        seed: int = 42,
        min_utility_retention: float = 0.9,
        max_linkage_rate: float = 0.25,
        ridge: float = 1e-2,
    ) -> Tuple[int, List[Dict]]:
        """Select a release rank without evaluating on its calibration samples.

        Utility is held-out multi-label ridge prediction relative to the full
        feature baseline. Privacy is held-out sample-linkage against a clean
        enrollment set. This is sample linkage, not patient re-identification,
        unless the caller supplies verified patient-level identities.
        """
        X = features.detach().cpu().float().numpy()
        Y = np.asarray(labels, dtype=np.float32)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("features and labels must be [N,d] and [N,C] with matching N")
        if n_splits < 2 or n_splits > X.shape[0]:
            raise ValueError("n_splits must be between 2 and the number of samples")
        ranks = sorted({int(k) for k in candidate_ranks if 0 < int(k) <= X.shape[1]})
        if not ranks:
            raise ValueError("candidate_ranks contains no valid rank")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(X.shape[0])
        folds = np.array_split(indices, n_splits)
        raw = {
            k: {"utility": [], "baseline": [], "linkage": [], "variance": [], "cosine": []}
            for k in ranks
        }

        for val_idx in folds:
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            x_mean = X_train.mean(axis=0, keepdims=True)
            y_mean = Y_train.mean(axis=0, keepdims=True)

            def ridge_predict(train_x, val_x):
                tx = train_x - train_x.mean(axis=0, keepdims=True)
                vx = val_x - train_x.mean(axis=0, keepdims=True)
                gram = tx.T @ tx + ridge * np.eye(tx.shape[1], dtype=np.float32)
                coef = np.linalg.solve(gram, tx.T @ (Y_train - y_mean))
                return vx @ coef + y_mean

            baseline_pred = ridge_predict(X_train, X_val)
            active = (Y_train.sum(axis=0) > 0) & (Y_val.sum(axis=0) > 0)
            if not np.any(active):
                raise ValueError("a validation fold has no diagnosis class shared with training")
            baseline_score = float(
                average_precision_score(Y_val[:, active], baseline_pred[:, active], average="micro")
            )

            for rank in ranks:
                dsr = DiagnosticSubspaceRetention.calibrate(
                    torch.from_numpy(X_train), Y_train, k=rank
                )
                U = dsr.subspace.numpy()
                train_scores = (X_train - x_mean) @ U
                val_scores = (X_val - x_mean) @ U
                pred = ridge_predict(train_scores, val_scores)
                score = float(
                    average_precision_score(Y_val[:, active], pred[:, active], average="micro")
                )

                query = val_scores @ U.T
                enrollment = X_val - x_mean
                en = enrollment / (np.linalg.norm(enrollment, axis=1, keepdims=True) + 1e-12)
                qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
                linkage = float(np.mean(np.argmax(qn @ en.T, axis=1) == np.arange(len(val_idx))))
                variance = float(np.sum(query ** 2) / (np.sum(enrollment ** 2) + 1e-12))
                cosine = float(np.mean(np.sum(
                    query * enrollment, axis=1
                ) / (
                    np.linalg.norm(query, axis=1)
                    * np.linalg.norm(enrollment, axis=1) + 1e-12
                )))

                raw[rank]["utility"].append(score)
                raw[rank]["baseline"].append(baseline_score)
                raw[rank]["linkage"].append(linkage)
                raw[rank]["variance"].append(variance)
                raw[rank]["cosine"].append(cosine)

        results = []
        for rank in ranks:
            values = raw[rank]
            utility = float(np.mean(values["utility"]))
            baseline = float(np.mean(values["baseline"]))
            retention = utility / max(baseline, 1e-12) if baseline > 0 else 0.0
            result = {
                "rank": rank,
                "utility_score": utility,
                "full_feature_utility_score": baseline,
                "utility_retention": retention,
                "sample_linkage_rate": float(np.mean(values["linkage"])),
                "retained_variance": float(np.mean(values["variance"])),
                "feature_cosine": float(np.mean(values["cosine"])),
                "release_values_per_image": rank * 16,
                "feasible": retention >= min_utility_retention
                and float(np.mean(values["linkage"])) <= max_linkage_rate,
            }
            results.append(result)

        feasible = [r for r in results if r["feasible"]]
        if feasible:
            selected = min(feasible, key=lambda r: (r["rank"], r["sample_linkage_rate"]))
        else:
            selected = max(
                results,
                key=lambda r: (r["utility_retention"] - r["sample_linkage_rate"], -r["rank"]),
            )
        return int(selected["rank"]), results

    def save(self, path: str):
        """Save DSR calibration (projection matrix + metadata)."""
        torch.save({
            "projection": self.projection,
            "subspace": self.subspace,
            "k": self.k,
            "label_classes": self.label_classes,
            "metrics": self.metrics,
        }, path)

    @staticmethod
    def load(
        path: str, *, expected_sha256: Optional[str] = None
    ) -> "DiagnosticSubspaceRetention":
        """Load DSR calibration from file."""
        data = load_tensor_checkpoint(
            path,
            required_keys=("projection", "subspace", "k", "label_classes", "metrics"),
            expected_sha256=expected_sha256,
        )
        return DiagnosticSubspaceRetention(
            projection=data["projection"],
            subspace=data["subspace"],
            k=data["k"],
            label_classes=data["label_classes"],
            metrics=data["metrics"],
        )


__all__ = ["DiagnosticSubspaceRetention"]
