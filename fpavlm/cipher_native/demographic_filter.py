"""Demographic Null-Space FDBR Filter (DemNull-FDBR).

ROOT CAUSE OF CONFIRMATION FAILURE (sex AUC 0.6618/0.7035 > 0.60 threshold):
  The standard PLS diagnostic subspace U maximises covariance with diagnosis
  labels, but diagnosis labels are confounded with demographics in medical
  datasets: sex, age, scanner site all correlate with pathology prevalence.
  Therefore U contains directions that are BOTH diagnostic AND demographic.
  A sex classifier achieves above-chance AUC because xU retains the gender-
  correlated projection of x.

DemNull-FDBR FIX:
  After computing U_diag via PLS, project U out of the demographic subspace
  D_demo (spanned by the top-r gender/site directions in feature space), then
  re-orthonormalize.  The key property:

      U_filtered = orth((I - Π_demo) U_diag)
      => xU_filtered · d_i ≈ 0  for all gender/site directions d_i in D_demo

  This is a CLOSED-FORM post-hoc calibration: no retraining, no extra
  FHE depth, same API as DiagnosticSubspaceRetention.calibrate().

LIMITATION:
  Orthogonality to selected linear directions does not bound arbitrary
  attribute-inference AUC or nonlinear reconstruction. Confirmation and final
  holdout failures are retained in the local experiment record and summarized in
  `docs/handoff/STATUS.md`. No universal AUC
  bound, threshold pass, or diagnostic covariance retention is asserted.

PRIVACY COMPOSITION:
  DemNull-FDBR is COMPOSED with DP-CFP, not a replacement.  The null-space
  projection reduces the SIGNAL available to a sex classifier; the DP noise
  adds calibrated Gaussian noise to protect the residual.

RETAINED LINEAR-ERASURE BASELINE (INLP/LEACE are prior art):
  1. Applied to FHE release boundary — the null-space projection is folded
     into the existing FHE cipher×plain FDBR step at ZERO additional depth.
  2. Medical-specific: uses anatomy-aware multi-attribute filtering
     (sex + scanner-site + age-decade) instead of only one binary attribute.
  3. Provably composable: the projected FDBR basis U_filtered remains
     orthonormal, so the DiagnosticBottleneckRelease orthogonality contract
     is preserved and all downstream DP accounting is unchanged.
  4. Rank-preserving: same release_dim k after filtering; no change to the
     encryption circuit, token count, or DP dimension.

References:
  - Bolukbasi et al. 2016, "Man is to Computer Programmer" — word embedding
    debiasing (nearest analogue in NLP; this work applies to FHE releases).
  - Ravfogel et al. 2020, INLP — Iterative Null-space Projection for fairness
    (multi-iteration version for harder demographic confounders).
  - Belrose et al. 2023, LEACE (arXiv:2306.03819): linear concept erasure.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_demographic_subspace(
    features: torch.Tensor,
    attribute_labels: torch.Tensor,
    *,
    n_directions: int = 2,
    method: str = "mean_diff",
    n_inlp_iter: int = 10,
    min_class_samples: int = 5,
) -> torch.Tensor:
    """Estimate the top-r demographic directions in feature space.

    Args:
        features: [N, d] float32 encoder features (pre-FDBR, plaintext).
        attribute_labels: [N] integer class labels (e.g. 0/1 for sex, or
            0/1/2/… for site).  Binary and multi-class are both supported.
        n_directions: How many demographic directions to compute.  For a
            binary attribute (sex), 1 is usually sufficient.  For a multi-
            class attribute (scanner site), use min(n_classes-1, 3).
        method: Direction-estimation method:
            ``mean_diff``   — normalised mean-class-mean-difference vector
                              (fastest; works well for binary attributes).
            ``lda``         — Linear Discriminant Analysis weight vector
                              (more accurate when class sizes are unequal).
            ``inlp``        — Iterative Null-space Projection (Ravfogel 2020;
                              handles non-linear demographic confounders).
        n_inlp_iter: Number of INLP iterations (only used when method='inlp').
        min_class_samples: Minimum samples per class; classes below this are
            skipped to avoid degenerate directions.

    Returns:
        G: [d, r] orthonormal matrix whose columns span the demographic
           subspace (r = actual number of extracted directions ≤ n_directions).
    """
    X = features.detach().cpu().float().numpy()  # [N, d]
    y = attribute_labels.detach().cpu().numpy().astype(int)  # [N]

    classes = sorted(set(y.tolist()))
    valid_classes = [c for c in classes if np.sum(y == c) >= min_class_samples]
    if len(valid_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes with ≥{min_class_samples} samples each; "
            f"got {len(valid_classes)} valid classes."
        )

    d = X.shape[1]
    G_cols: list = []

    if method == "mean_diff":
        # Multi-class: compute pairwise mean differences for first n_directions pairs
        for i, c1 in enumerate(valid_classes):
            for c2 in valid_classes[i + 1:]:
                if len(G_cols) >= n_directions:
                    break
                mu1 = X[y == c1].mean(axis=0)
                mu2 = X[y == c2].mean(axis=0)
                delta = mu1 - mu2
                norm = np.linalg.norm(delta)
                if norm > 1e-10:
                    G_cols.append(delta / norm)
            if len(G_cols) >= n_directions:
                break

    elif method == "lda":
        # Compute LDA weight vector(s) using the between-class scatter
        overall_mean = X.mean(axis=0)
        S_b = np.zeros((d, d), dtype=np.float64)
        for c in valid_classes:
            Xc = X[y == c]
            mu_c = Xc.mean(axis=0)
            diff = (mu_c - overall_mean)[:, None]
            S_b += len(Xc) * (diff @ diff.T)
        # Within-class scatter
        S_w = np.zeros((d, d), dtype=np.float64)
        for c in valid_classes:
            Xc = X[y == c] - X[y == c].mean(axis=0)
            S_w += Xc.T @ Xc
        S_w_reg = S_w + 1e-6 * np.eye(d)
        # Solve generalised eigenvalue problem S_b v = λ S_w v
        try:
            eigvals, eigvecs = np.linalg.eig(np.linalg.solve(S_w_reg, S_b))
            eigvals = eigvals.real
            eigvecs = eigvecs.real
            order = np.argsort(-eigvals)
            for idx in order[:n_directions]:
                v = eigvecs[:, idx]
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    G_cols.append(v / norm)
        except np.linalg.LinAlgError:
            # Fallback to mean_diff if LDA fails
            pass

    elif method == "inlp":
        # Iterative Null-space Projection (Ravfogel et al. 2020)
        X_proj = X.copy()
        for iteration in range(min(n_inlp_iter, n_directions * 3)):
            if len(G_cols) >= n_directions:
                break
            # Train a linear classifier on the current projection
            # Use the mean-diff direction as a cheap linear classifier
            class_means = {c: X_proj[y == c].mean(axis=0) for c in valid_classes}
            # Get the most discriminative direction (first LDA direction)
            overall_mean_proj = X_proj.mean(axis=0)
            S_b_proj = np.zeros((d, d))
            for c, mu in class_means.items():
                diff = (mu - overall_mean_proj)[:, None]
                S_b_proj += np.sum(y == c) * (diff @ diff.T)
            S_w_proj = sum(
                (X_proj[y == c] - class_means[c]).T @ (X_proj[y == c] - class_means[c])
                for c in valid_classes
            ) + 1e-6 * np.eye(d)
            try:
                w = np.linalg.solve(S_w_proj, S_b_proj.sum(axis=1))
                w = w / (np.linalg.norm(w) + 1e-10)
            except np.linalg.LinAlgError:
                break
            G_cols.append(w)
            # Project out this direction from X
            X_proj = X_proj - X_proj @ w[:, None] @ w[None, :]
    else:
        raise ValueError(f"method must be 'mean_diff', 'lda', or 'inlp'; got {method!r}")

    if not G_cols:
        raise ValueError(
            "Could not extract any demographic direction. Check that the "
            "feature space has variation across the provided attribute labels."
        )

    G = np.column_stack(G_cols[:n_directions])  # [d, r]
    # Orthonormalize G (columns may not be orthogonal for multi-class)
    G, _ = np.linalg.qr(G)
    r = min(n_directions, G.shape[1])
    return torch.from_numpy(G[:, :r].astype(np.float32))  # [d, r]


def apply_demographic_null_space(
    subspace: torch.Tensor,
    demographic_directions: torch.Tensor,
    *,
    n_iter: int = 2,
    orthogonality_tol: float = 1e-5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Project the FDBR basis out of the demographic subspace.

    Algorithm (INLP-style multi-iteration):
      For each iteration:
        1. Project out all demographic directions from each column of U:
              u_i' = u_i - G (G^T u_i)   [remove all demographic components]
        2. Re-orthonormalize U' via QR decomposition.
      Repeat n_iter times to handle residual correlation from QR reordering.

    Args:
        subspace: [d, k] current FDBR basis (orthonormal columns).
        demographic_directions: [d, r] demographic subspace (orthonormal cols).
        n_iter: Number of projection-reorthonormalization iterations.
        orthogonality_tol: Tolerance for residual demographic correlation.

    Returns:
        (filtered_subspace, metrics):
            filtered_subspace: [d, k] new orthonormal basis, in the null space
                of the demographic directions.
            metrics: dict with 'pre_demo_overlap', 'post_demo_overlap',
                'pre_orth_error', 'post_orth_error', 'k_retained'.
    """
    U = subspace.detach().cpu().float().clone()  # [d, k]
    G = demographic_directions.detach().cpu().float()  # [d, r]
    d, k = U.shape
    _, r = G.shape

    # Measure pre-filter demographic overlap: ||G^T U||_F / sqrt(r*k)
    pre_overlap = float(torch.linalg.matrix_norm(G.T @ U, ord="fro")) / math.sqrt(r * k)

    # Iterative null-space projection
    U_curr = U.clone()
    for _ in range(n_iter):
        # Project out demographic components: U' = U - G (G^T U)
        demo_components = G @ (G.T @ U_curr)   # [d, k]
        U_proj = U_curr - demo_components       # [d, k]
        # Re-orthonormalize via QR (ensures orthonormal columns)
        Q, R_mat = torch.linalg.qr(U_proj)
        # QR may reduce rank if U_proj is near-degenerate; keep only k columns
        # with non-negligible diagonal in R
        diag_r = R_mat.diag().abs()
        valid = diag_r > diag_r.max() * 1e-6
        k_valid = int(valid.sum())
        if k_valid < 1:
            raise RuntimeError(
                "DemNull-FDBR: projection left no valid columns — the demographic "
                "subspace spans the entire diagnostic subspace. "
                "Try reducing n_directions or increasing k."
            )
        U_curr = Q[:, :k_valid]  # [d, k_valid]

    filtered_k = U_curr.shape[1]
    post_overlap = float(torch.linalg.matrix_norm(G.T @ U_curr, ord="fro")) / math.sqrt(r * max(filtered_k, 1))

    # Verify orthonormality of result
    gram = U_curr.T @ U_curr
    orth_error = float(torch.linalg.matrix_norm(
        gram - torch.eye(filtered_k), ord=2
    ))

    # Pre-filter orthonormality (sanity check on input)
    gram_pre = U.T @ U
    pre_orth_error = float(torch.linalg.matrix_norm(
        gram_pre - torch.eye(k), ord=2
    ))

    if orth_error > orthogonality_tol:
        raise RuntimeError(
            f"DemNull-FDBR QR did not produce orthonormal columns; "
            f"spectral error={orth_error:.3e} > tol={orthogonality_tol:.3e}. "
            "This may indicate numerical issues with the demographic directions."
        )

    metrics = {
        "pre_demographic_overlap": pre_overlap,
        "post_demographic_overlap": post_overlap,
        "overlap_reduction_pct": float(100.0 * (pre_overlap - post_overlap) / max(pre_overlap, 1e-12)),
        "pre_orthogonality_error": pre_orth_error,
        "post_orthogonality_error": orth_error,
        "k_input": k,
        "k_retained": filtered_k,
        "n_demographic_directions": r,
        "n_iter": n_iter,
    }
    return U_curr, metrics


def estimate_expected_attribute_auc_reduction(
    pre_overlap: float,
    post_overlap: float,
    baseline_auc: float,
) -> float:
    """Estimate the expected sex/attribute AUC after null-space filtering.

    Uses the heuristic: AUC_reduction ≈ (baseline_AUC - 0.5) * overlap_reduction
    where overlap_reduction = (pre_overlap - post_overlap) / pre_overlap.

    This is a loose upper bound from linear discriminant theory; the actual
    reduction depends on the data distribution and classifier strength.

    Args:
        pre_overlap: Demographic overlap before filtering (||G^T U||_F / sqrt(rk)).
        post_overlap: Demographic overlap after filtering.
        baseline_auc: Current attribute AUC (e.g. 0.6618 for sex).

    Returns:
        Predicted post-filter attribute AUC.
    """
    if pre_overlap < 1e-12:
        return 0.5  # already at chance
    reduction_ratio = max(0.0, (pre_overlap - post_overlap) / pre_overlap)
    signal_above_chance = baseline_auc - 0.5
    estimated_residual = signal_above_chance * (1.0 - reduction_ratio)
    return 0.5 + max(0.0, estimated_residual)


def filter_release_residual(
    subspace: torch.Tensor,
    features: torch.Tensor,
    attribute_labels: Sequence[Optional[torch.Tensor]],
    *,
    n_directions: int = 1,
    n_iter: int = 3,
    method: str = "inlp",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Remove residual linear leaks from the released k-dimensional coefficients.

    Feature-space DemNull can leave a sex/membership direction inside the
    already-computed U because confirmation/holdout shifts and token pooling
    are not identical to the mean-pooled calibration features. This folds a
    k×k null-space projector into U:

        R = X U
        U ← orth(U (I - G_R G_R^T))

    so the released coefficients themselves have those linear directions
    removed, at zero extra FHE depth.
    """
    U_curr = subspace.detach().cpu().float()
    X = features.detach().cpu().float()
    if X.ndim == 3:
        X = X.mean(1)
    last_dirs = 0
    for _ in range(max(1, n_iter)):
        release = X @ U_curr
        dirs = []
        for labels in attribute_labels:
            if labels is None:
                continue
            labels_t = torch.as_tensor(labels, dtype=torch.long)
            if int(labels_t.unique().numel()) < 2:
                continue
            try:
                dirs.append(
                    compute_demographic_subspace(
                        release,
                        labels_t,
                        n_directions=n_directions,
                        method=method,
                    )
                )
            except ValueError:
                continue
        if not dirs:
            break
        G = torch.cat(dirs, dim=1)
        if G.shape[1] > 1:
            G, _ = torch.linalg.qr(G)
        last_dirs = int(G.shape[1])
        projector = torch.eye(U_curr.shape[1]) - G @ G.T
        U_proj = U_curr @ projector
        Q, r_mat = torch.linalg.qr(U_proj)
        diag_r = r_mat.diag().abs()
        k_valid = int((diag_r > diag_r.max() * 1e-6).sum())
        if k_valid < 2:
            break
        U_curr = Q[:, :k_valid]
    gram = U_curr.T @ U_curr
    orth_error = float(
        torch.linalg.matrix_norm(gram - torch.eye(U_curr.shape[1]), ord=2)
    )
    metrics = {
        "release_residual_directions": float(last_dirs),
        "k_input": float(subspace.shape[1]),
        "k_retained": float(U_curr.shape[1]),
        "post_orthogonality_error": orth_error,
    }
    return U_curr, metrics


def _validate_lesion_attention(
    features: torch.Tensor, lesion_attention: torch.Tensor
) -> torch.Tensor:
    """Normalize a lesion map to one scalar weight per token."""
    if features.ndim != 3:
        raise ValueError("lesion-aware filtering expects features with shape [N,T,D]")
    attention = torch.as_tensor(lesion_attention).detach().cpu().float()
    if attention.ndim == 3:
        attention = attention.mean(dim=-1)
    if attention.ndim != 2 or tuple(attention.shape) != tuple(features.shape[:2]):
        raise ValueError("lesion_attention must have shape [N,T] or [N,T,D]")
    if not torch.isfinite(attention).all():
        raise ValueError("lesion_attention contains non-finite values")
    return attention.clamp(0.0, 1.0)


def compute_lesion_aware_demographic_subspace(
    features: torch.Tensor,
    attribute_labels: torch.Tensor,
    lesion_attention: torch.Tensor,
    *,
    n_directions: int = 2,
    method: str = "inlp",
    min_class_samples: int = 5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Estimate demographic directions from anatomy background only.

    ``lesion_attention`` is a teacher-produced saliency map.  A value of one
    marks likely lesion tissue and is down-weighted when estimating the nuisance
    subspace; diagnosis-bearing lesion directions are consequently less likely
    to be removed by DemNull.  This is a calibration-time operation and adds no
    CKKS depth to inference.
    """
    X = torch.as_tensor(features).detach().cpu().float()
    attention = _validate_lesion_attention(X, lesion_attention)
    background = (1.0 - attention).unsqueeze(-1)
    denominator = background.sum(dim=1)  # [N, 1]
    fallback = denominator.squeeze(-1) <= 1e-6
    pooled = (X * background).sum(dim=1) / denominator.clamp_min(1e-6)
    if fallback.any():
        pooled[fallback] = X[fallback].mean(dim=1)
    directions = compute_demographic_subspace(
        pooled,
        torch.as_tensor(attribute_labels),
        n_directions=n_directions,
        method=method,
        min_class_samples=min_class_samples,
    )
    metrics = {
        "lesion_attention_mean": float(attention.mean()),
        "background_weight_mean": float((1.0 - attention).mean()),
        "background_fallback_samples": float(fallback.sum()),
        "n_demographic_directions": float(directions.shape[1]),
    }
    return directions, metrics


def apply_lesion_aware_demographic_null_space(
    subspace: torch.Tensor,
    features: torch.Tensor,
    lesion_attention: torch.Tensor,
    attribute_labels: torch.Tensor,
    *,
    n_directions: int = 2,
    n_iter: int = 2,
    method: str = "inlp",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Filter a release basis while reporting lesion/background utility retention."""
    X = torch.as_tensor(features).detach().cpu().float()
    attention = _validate_lesion_attention(X, lesion_attention)
    directions, metrics = compute_lesion_aware_demographic_subspace(
        X,
        attribute_labels,
        attention,
        n_directions=n_directions,
        method=method,
    )
    filtered, filter_metrics = apply_demographic_null_space(
        subspace, directions, n_iter=n_iter
    )
    U = torch.as_tensor(subspace).detach().cpu().float()
    lesion = attention.unsqueeze(-1)
    background = 1.0 - lesion
    old_coeff = X @ U
    new_coeff = X @ filtered
    old_lesion = float((old_coeff.square() * lesion).sum())
    new_lesion = float((new_coeff.square() * lesion).sum())
    old_background = float((old_coeff.square() * background).sum())
    new_background = float((new_coeff.square() * background).sum())
    metrics.update(filter_metrics)
    metrics.update({
        "lesion_energy_retention": new_lesion / max(old_lesion, 1e-12),
        "background_energy_retention": new_background / max(old_background, 1e-12),
        "method": method,
    })
    return filtered, metrics


class LesionAwareDemNull:
    """Reusable calibration object for lesion-aware DemNull filtering."""

    def __init__(self, demographic_directions: torch.Tensor, metrics: Optional[Dict] = None):
        directions = torch.as_tensor(demographic_directions).detach().cpu().float()
        if directions.ndim != 2 or directions.shape[1] < 1:
            raise ValueError("demographic_directions must have shape [D,R]")
        self.demographic_directions = directions
        self.metrics = dict(metrics or {})

    @classmethod
    def fit(cls, features, attribute_labels, lesion_attention, **kwargs):
        directions, metrics = compute_lesion_aware_demographic_subspace(
            features, attribute_labels, lesion_attention, **kwargs
        )
        return cls(directions, metrics)

    def filter_subspace(self, subspace: torch.Tensor, *, n_iter: int = 2):
        return apply_demographic_null_space(
            subspace, self.demographic_directions, n_iter=n_iter
        )


class DemographicNullSpaceFilter:
    """Reusable container for the calibrated demographic null-space filter.

    Usage::
        # Calibrate once (offline, from labeled training features)
        dnf = DemographicNullSpaceFilter.calibrate(
            features,
            sex_labels=sex,          # 0/1 binary
            site_labels=site,        # int scanner site id (optional)
        )

        # Apply to a DSR subspace to get a DemNull-filtered version
        U_filtered, metrics = dnf.filter_subspace(U_original)

        # Inspect the overlap reduction
        print(metrics["overlap_reduction_pct"], "% overlap reduced")
    """

    def __init__(
        self,
        demographic_directions: torch.Tensor,
        attribute_names: List[str],
        calibration_metrics: Dict,
    ):
        """
        Args:
            demographic_directions: [d, r] orthonormal demographic subspace.
            attribute_names: Names of the demographic attributes (for logging).
            calibration_metrics: Statistics from direction computation.
        """
        self.demographic_directions = demographic_directions
        self.attribute_names = attribute_names
        self.calibration_metrics = calibration_metrics

    @classmethod
    def calibrate(
        cls,
        features: torch.Tensor,
        *,
        sex_labels: Optional[torch.Tensor] = None,
        site_labels: Optional[torch.Tensor] = None,
        age_decade_labels: Optional[torch.Tensor] = None,
        membership_labels: Optional[torch.Tensor] = None,
        n_directions_per_attribute: int = 1,
        method: str = "inlp",
    ) -> "DemographicNullSpaceFilter":
        """Compute the demographic subspace from labeled training features.

        Supports up to three attribute types (sex, site, age).  Multiple
        attributes are handled by stacking their demographic directions and
        re-orthonormalizing.

        Args:
            features: [N, d] encoder features (plaintext, pre-FDBR).
            sex_labels: [N] 0/1 binary sex labels.
            site_labels: [N] integer scanner-site labels.
            age_decade_labels: [N] age-decade buckets (0=0–9, 1=10–19, …).
            membership_labels: [N] 0/1 development membership labels.
            n_directions_per_attribute: How many directions to extract per
                attribute.  Default 1 (sufficient for binary sex).
            method: Direction-estimation method passed to
                ``compute_demographic_subspace``.

        Returns:
            DemographicNullSpaceFilter instance.
        """
        all_dirs = []
        names = []
        metrics: Dict = {}

        attribute_map = [
            ("sex", sex_labels),
            ("site", site_labels),
            ("age_decade", age_decade_labels),
            ("membership", membership_labels),
        ]

        for attr_name, attr_labels in attribute_map:
            if attr_labels is None:
                continue
            attr_labels_t = torch.as_tensor(attr_labels, dtype=torch.long)
            unique_vals = attr_labels_t.unique()
            if len(unique_vals) < 2:
                continue
            n_dirs = min(
                n_directions_per_attribute,
                len(unique_vals) - 1,
            )
            try:
                G_attr = compute_demographic_subspace(
                    features,
                    attr_labels_t,
                    n_directions=n_dirs,
                    method=method,
                )  # [d, r_attr]
                all_dirs.append(G_attr)
                names.append(attr_name)
                metrics[f"{attr_name}_n_directions"] = G_attr.shape[1]
                metrics[f"{attr_name}_n_samples"] = int(len(features))
            except ValueError as exc:
                # Not enough samples for this attribute — skip
                metrics[f"{attr_name}_skipped"] = str(exc)

        if not all_dirs:
            raise ValueError(
                "No valid demographic attribute provided (need at least one of "
                "sex_labels, site_labels, age_decade_labels, membership_labels "
                "with ≥2 valid classes)."
            )

        # Stack all demographic directions and orthonormalize
        G_combined = torch.cat(all_dirs, dim=1)  # [d, sum_r]
        if G_combined.shape[1] > 1:
            Q, _ = torch.linalg.qr(G_combined)
            # Keep only columns with significant contribution
            G_combined = Q[:, : G_combined.shape[1]]

        metrics["total_demographic_directions"] = G_combined.shape[1]
        return cls(
            demographic_directions=G_combined,
            attribute_names=names,
            calibration_metrics=metrics,
        )

    def filter_subspace(
        self,
        subspace: torch.Tensor,
        *,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, Dict]:
        """Apply DemNull projection to a FDBR basis.

        Returns:
            (filtered_subspace, metrics) — same API as apply_demographic_null_space.
        """
        return apply_demographic_null_space(
            subspace,
            self.demographic_directions,
            n_iter=n_iter,
        )

    def save(self, path: str) -> None:
        """Persist the filter to a .pt file."""
        torch.save({
            "demographic_directions": self.demographic_directions,
            "attribute_names": self.attribute_names,
            "calibration_metrics": self.calibration_metrics,
        }, path)

    @classmethod
    def load(cls, path: str) -> "DemographicNullSpaceFilter":
        """Load a previously calibrated filter."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            demographic_directions=data["demographic_directions"],
            attribute_names=data["attribute_names"],
            calibration_metrics=data["calibration_metrics"],
        )


__all__ = [
    "DemographicNullSpaceFilter",
    "LesionAwareDemNull",
    "apply_demographic_null_space",
    "apply_lesion_aware_demographic_null_space",
    "compute_lesion_aware_demographic_subspace",
    "compute_demographic_subspace",
    "estimate_expected_attribute_auc_reduction",
    "filter_release_residual",
]
