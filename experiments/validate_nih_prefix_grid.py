"""Score the locked prefix grid on the inspected NIH patients.

EVALUATION_PROTOCOL.md sections 2.24 and 2.25. Section 2.24 is not rewritten.
The pass rule is the PA-only paired difference on the 506 analysis patients.
The reserved confirmation patients are not scored.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diagnose_release_sufficiency import _fingerprint  # noqa: E402
from experiments.fit_nih_oracle_bridge import DEVELOPMENT, _native, _records  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    NEGATIVE_LIMIT,
    ORDER_SALT,
    POSITIVE_LIMIT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm import DiagnosticPrefixProjector, PrefixSpec  # noqa: E402
from fpavlm.evaluation import paired_auc_noninferiority, roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
FIXED_RANKS = (4, 8, 16, 32, 64)
FOLDS = 5
MARGIN = 0.05
PROBE = ROOT / "data" / "nih_chestxray14"
CONFIRMATION = ROOT / "data" / "nih_chestxray14_confirmation"
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
STORED_AUC = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
CHUNK = 65536


def fold_id(patient_id):
    """Patient fold from the existing order salt. The assignment is not tuned."""
    digest = hashlib.sha256(f"{ORDER_SALT}{patient_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def select_cell(cells):
    """Smallest rank, then the supervised basis, then the 72-patient bridge."""
    passing = [cell for cell in cells if cell.get("passes") and cell.get("basis") in {"pca", "pls"}]
    if not passing:
        return None

    def order(cell):
        basis = 0 if cell["basis"] == "pls" else 1
        bridge = 0 if cell["bridge"] == "development_72" else 1
        return (int(cell["rank_order"]), basis, bridge)

    return min(passing, key=order)


def pls_coefficients(kernel, labels, rank):
    """Univariate centered PLS coefficient matrices in the row span of X."""
    gram = np.asarray(kernel, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    y = y - y.mean()
    size = gram.shape[0]
    change = np.eye(size)
    responses = y.reshape(-1, 1).copy()
    left, right = [], []
    for _ in range(rank):
        score = responses[:, 0]
        current = change @ gram @ change.T
        quad = float(score @ current @ score)
        if quad <= 1e-18:
            break
        scale = np.sqrt(quad)
        latent = current @ score / scale
        left.append(change.T @ score / scale)
        right.append(change.T @ latent / float(latent @ latent))
        loading = float(latent @ score) / float(latent @ latent)
        responses = responses - np.outer(latent, [loading])
        change = change - np.outer(latent, latent) @ change / float(latent @ latent)
    if not left:
        raise ValueError("PLS found no component")
    return np.column_stack(left), np.column_stack(right)


def pls_mapping(train_kernel, coefficients, loadings, rank):
    """Map new rows through one prefix. The rotation is fit on the training kernel."""
    used = min(rank, coefficients.shape[1])
    left, right = coefficients[:, :used], loadings[:, :used]
    rotation = np.linalg.pinv(right.T @ train_kernel @ left)
    return left @ rotation


def apply_pls(kernel, mapping):
    return kernel @ mapping


def bridge_tokens(train_code, train_tokens, test_code):
    """Ridge-1 prefix. Sigma is 0. Targets stay chunked so the fit does not copy them."""
    features = train_code.double()
    feature_mean = features.mean(0)
    centered = features - feature_mean
    gram = centered.T @ centered + torch.eye(features.shape[1], dtype=torch.float64)
    coef = torch.linalg.solve(gram, centered.T)
    flat = train_tokens.reshape(len(train_tokens), -1)
    predicted = torch.empty(len(test_code), flat.shape[1], dtype=torch.float32)
    for start in range(0, flat.shape[1], 4096):
        end = min(start + 4096, flat.shape[1])
        block = flat[:, start:end].double()
        target_mean = block.mean(0)
        weight64 = coef @ (block - target_mean)
        weight = weight64.float()
        bias = (target_mean - feature_mean.double() @ weight64).float()
        predicted[:, start:end] = test_code.float() @ weight + bias
    return predicted.reshape(len(test_code), train_tokens.shape[-2], train_tokens.shape[-1])


def pca_svd_scores(train_tokens, test_tokens, rank):
    """Development PCA. Same centering and SVD path as the sealed oracle."""
    train = train_tokens.reshape(len(train_tokens), -1).double()
    test = test_tokens.reshape(len(test_tokens), -1).double()
    mean = train.mean(0)
    basis = torch.linalg.svd(train - mean, full_matrices=False).Vh.T[:, :rank]
    return ((train - mean) @ basis).float(), ((test - mean) @ basis).float()


def _chunks(width):
    for start in range(0, width, CHUNK):
        yield start, min(start + CHUNK, width)


def gram_and_mean(flat):
    mean = torch.zeros(flat.shape[1], dtype=torch.float64)
    count = len(flat)
    for start, end in _chunks(flat.shape[1]):
        mean[start:end] = flat[:, start:end].double().mean(0)
    gram = torch.zeros(count, count, dtype=torch.float64)
    for start, end in _chunks(flat.shape[1]):
        block = flat[:, start:end].double()
        block -= mean[start:end]
        gram += block @ block.T
    return mean, gram


def pca_gram_scores(train_flat, test_flat, mean, gram, rank):
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    chosen = [int(index) for index in order.tolist() if float(values[index]) > 1e-8][:rank]
    if not chosen:
        raise ValueError("PCA found no positive component")
    picked = torch.tensor(chosen)
    scale = values[picked].sqrt()
    coeff = vectors[:, picked] / scale
    train_scores = (gram @ coeff).float()
    test_scores = torch.zeros(len(test_flat), len(chosen), dtype=torch.float64)
    for start, end in _chunks(train_flat.shape[1]):
        block = train_flat[:, start:end].double() - mean[start:end]
        right = block.T @ coeff
        test_block = test_flat[:, start:end].double() - mean[start:end]
        test_scores += test_block @ right
    return train_scores, test_scores.float(), len(chosen)


def _load_split(path, expected):
    records = _records(path, expected)
    return [{**row, "root": path} for row in records]


def _log_odds_rows(llm, predicted, positive_id, negative_id):
    scores = []
    for offset in range(len(predicted)):
        scores.append(_log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id))
    return scores


def _auc_block(labels, native, release, patients):
    return paired_auc_noninferiority(
        labels, native, release, patients, margin=MARGIN, bootstrap_samples=2000, seed=42,
    )


def _versus_view(labels, view_scores, release, patients):
    raw = _auc_block(labels, view_scores, release, patients)
    return {
        "auc_view": raw["auc_native"],
        "auc_release": raw["auc_release"],
        "difference": raw["difference"],
        "one_sided_95_lower": raw["one_sided_95_lower"],
        "valid_draws": raw["valid_draws"],
    }


def _cell(basis, bridge, rank_name, rank_order, labels, views, native, release, patients, view_release):
    pa = [offset for offset, view in enumerate(views) if view == "PA"]
    full = _auc_block(labels, native, release, patients)
    pa_only = _auc_block(
        [labels[offset] for offset in pa],
        [native[offset] for offset in pa],
        [release[offset] for offset in pa],
        [patients[offset] for offset in pa],
    )
    return {
        "basis": basis,
        "bridge": bridge,
        "rank": rank_name,
        "rank_order": rank_order,
        "full_set": full,
        "pa_only": pa_only,
        "versus_view_full_set": _versus_view(labels, view_release, release, patients),
        "passes": bool(pa_only["noninferior"]),
        "scores": release,
    }


def _view_row(bridge, labels, native, release, patients):
    full = _auc_block(labels, native, release, patients)
    return {
        "basis": "view",
        "bridge": bridge,
        "rank": "ap_indicator",
        "full_set": full,
        "pa_only_auc": None,
        "passes": False,
        "scores": release,
    }


def _codes_for_ranks(train_codes, test_codes, ranks):
    return {rank: (train_codes[:, :rank], test_codes[:, :rank]) for rank in ranks}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _load_split(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    probe = _load_split(PROBE, POSITIVE_LIMIT + NEGATIVE_LIMIT)
    previous = _load_split(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    sealed = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))
    if not sealed.get("pool_complete") or sealed.get("auc_computed") is not True:
        raise SystemExit("the 506 analysis images are not the scored section 2.22 set")
    analysis = [{**row, "root": SEALED} for row in sealed["records"]]
    if len(analysis) != 506 or any(row.get("download_error") for row in analysis):
        raise SystemExit("the analysis set is not complete")
    inspected = development + analysis + probe + previous
    identities = [str(row["patient_id"]) for row in inspected]
    if len(identities) != 677 or len(set(identities)) != 677:
        raise SystemExit("the inspected pool is not 677 distinct patients")
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed") or holdout.get("negative_sample_drawn"):
        raise SystemExit("the reserved confirmation pool was already opened")
    reserved = {str(row["patient_id"]) for row in holdout["records"]}
    if reserved & set(identities) or len(reserved) != 598:
        raise SystemExit("the reserved confirmation patients are not intact")
    folds = [fold_id(patient) for patient in identities]
    labels_all = [int(row["label_pneumonia"]) for row in inspected]
    if any(row.get("view_position") not in {"PA", "AP"} for row in inspected):
        raise SystemExit("an inspected image has a view outside PA and AP")
    for fold in range(FOLDS):
        train = [label for label, group in zip(labels_all, folds) if group != fold]
        if len(set(train)) < 2 or not any(group == fold for group in folds):
            raise SystemExit("a patient fold does not support the locked cross-fit")
    llm = load_backend(BACKEND)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    tokens = torch.cat([
        _native(llm, row["root"], [row], BACKEND) for row in inspected
    ]).float()
    dev_index = list(range(len(development)))
    eval_index = list(range(len(development), len(development) + len(analysis)))
    eval_rows = [inspected[offset] for offset in eval_index]
    eval_labels = [int(row["label_pneumonia"]) for row in eval_rows]
    eval_views = [row["view_position"] for row in eval_rows]
    eval_ids = [str(row["patient_id"]) for row in eval_rows]
    if any(view not in {"PA", "AP"} for view in eval_views):
        raise SystemExit("an analysis image has a view outside PA and AP")
    native_scores = _log_odds_rows(llm, tokens[eval_index], positive_id, negative_id)
    pa_offsets = [offset for offset, view in enumerate(eval_views) if view == "PA"]
    pa_auc = roc_auc(
        [eval_labels[offset] for offset in pa_offsets],
        [native_scores[offset] for offset in pa_offsets],
    )
    stored_pa = json.loads(STRATA.read_text(encoding="utf-8"))["pa_only"]["native"]["auc"]
    if abs(pa_auc - stored_pa) > 1e-6:
        raise SystemExit("the recomputed PA-only native AUC disagrees with section 2.23")
    stored_payload = json.loads(STORED_AUC.read_text(encoding="utf-8"))
    stored_ranks = {row["rank"]: row["auc"]["auc_release"] for row in stored_payload["ranks"]}
    stored_native = stored_payload["ranks"][0]["reports"]
    if [row["patient_id"] for row in stored_native] != eval_ids:
        raise SystemExit("the analysis patient order does not match section 2.22")
    native_gap = max(abs(left - right["native_log_odds"]) for left, right in zip(native_scores, stored_native))
    print(json.dumps({"native_log_odds_max_abs_gap": native_gap, "pa_only_native_auc": pa_auc}), flush=True)
    dev_tokens = tokens[dev_index]
    eval_tokens = tokens[eval_index]
    dev_labels = [int(row["label_pneumonia"]) for row in development]
    dev_view = torch.tensor(
        [[1.0 if row["view_position"] == "AP" else 0.0] for row in development], dtype=torch.float32,
    )
    eval_view = torch.tensor(
        [[1.0 if view == "AP" else 0.0] for view in eval_views], dtype=torch.float32,
    )
    cells, view_rows = [], []
    view_72 = _log_odds_rows(
        llm, bridge_tokens(dev_view, dev_tokens, eval_view), positive_id, negative_id,
    )
    view_rows.append(_view_row("development_72", eval_labels, native_scores, view_72, eval_ids))
    flat_dev = dev_tokens.reshape(len(dev_tokens), -1)
    flat_eval = eval_tokens.reshape(len(eval_tokens), -1)
    dev_mean, dev_gram = gram_and_mean(flat_dev)
    dev_kernel = dev_gram.numpy()
    pca_train, pca_test = pca_svd_scores(dev_tokens, eval_tokens, len(development) - 1)
    pls_left, pls_right = pls_coefficients(dev_kernel, dev_labels, len(development) - 1)
    eval_kernel = torch.zeros(len(eval_tokens), len(dev_tokens), dtype=torch.float64)
    for start, end in _chunks(flat_dev.shape[1]):
        block = flat_dev[:, start:end].double() - dev_mean[start:end]
        eval_kernel += (flat_eval[:, start:end].double() - dev_mean[start:end]) @ block.T
    ranks_72 = [*FIXED_RANKS, len(development) - 1]
    for rank in ranks_72:
        name = "n_train_minus_1" if rank == len(development) - 1 else str(rank)
        pca_release = _log_odds_rows(
            llm,
            bridge_tokens(pca_train[:, :rank], dev_tokens, pca_test[:, :rank]),
            positive_id, negative_id,
        )
        if rank in stored_ranks:
            release_auc = roc_auc(eval_labels, pca_release)
            code = pca_train[:, :rank].reshape(len(dev_tokens), rank, 1)
            test_code = pca_test[:, :rank].reshape(len(eval_tokens), rank, 1)
            digest = hashlib.sha256(
                "\n".join(str(row["patient_id"]) for row in development).encode("utf-8")
            ).hexdigest()
            spec = PrefixSpec(
                BACKEND, (rank, 1), int(dev_tokens.shape[-2]), int(dev_tokens.shape[-1]), digest,
                _fingerprint({"release": "nih_sealed_pca_oracle", "rank": rank, "section": "2.22"}),
            )
            projected = DiagnosticPrefixProjector(spec).fit(code, dev_tokens, sigma=0.)(test_code)
            projected_scores = _log_odds_rows(llm, projected, positive_id, negative_id)
            projected_auc = roc_auc(eval_labels, projected_scores)
            print(json.dumps({
                "rank": rank,
                "bridge_auc": release_auc,
                "projector_auc": projected_auc,
                "stored_auc": stored_ranks[rank],
            }), flush=True)
            if abs(projected_auc - stored_ranks[rank]) > 1e-4:
                raise SystemExit("the development PCA bridge does not reproduce section 2.22")
            pca_release = projected_scores
        cells.append(_cell(
            "pca", "development_72", name, rank, eval_labels, eval_views,
            native_scores, pca_release, eval_ids, view_72,
        ))
        mapping = pls_mapping(dev_kernel, pls_left, pls_right, rank)
        train_pls = torch.tensor(apply_pls(dev_kernel, mapping), dtype=torch.float32)
        test_pls = torch.tensor(apply_pls(eval_kernel.numpy(), mapping), dtype=torch.float32)
        pls_release = _log_odds_rows(
            llm, bridge_tokens(train_pls, dev_tokens, test_pls), positive_id, negative_id,
        )
        cells.append(_cell(
            "pls", "development_72", name, rank, eval_labels, eval_views,
            native_scores, pls_release, eval_ids, view_72,
        ))
        print(json.dumps({
            "bridge": "development_72", "rank": name,
            "pca_passes": cells[-2]["passes"], "pls_passes": cells[-1]["passes"],
        }), flush=True)
    cross_scores = {
        ("pca", rank): [None] * len(eval_index) for rank in FIXED_RANKS
    }
    cross_scores.update({("pls", rank): [None] * len(eval_index) for rank in FIXED_RANKS})
    cross_scores[("pca", "n_train_minus_1")] = [None] * len(eval_index)
    cross_scores[("pls", "n_train_minus_1")] = [None] * len(eval_index)
    cross_view = [None] * len(eval_index)
    full_ranks = []
    eval_position = {offset: position for position, offset in enumerate(eval_index)}
    flat_all = tokens.reshape(len(tokens), -1)
    for fold in range(FOLDS):
        train_idx = [offset for offset, group in enumerate(folds) if group != fold]
        test_idx = [offset for offset, group in enumerate(folds) if group == fold and offset in eval_position]
        train_flat = flat_all[train_idx]
        test_flat = flat_all[test_idx]
        mean, gram = gram_and_mean(train_flat)
        max_rank = len(train_idx) - 1
        full_ranks.append(max_rank)
        pca_train_codes, pca_test_codes, pca_width = pca_gram_scores(
            train_flat, test_flat, mean, gram, max_rank,
        )
        train_y = [labels_all[offset] for offset in train_idx]
        left, right = pls_coefficients(gram.numpy(), train_y, max_rank)
        test_kernel = torch.zeros(len(test_idx), len(train_idx), dtype=torch.float64)
        for start, end in _chunks(train_flat.shape[1]):
            block = train_flat[:, start:end].double() - mean[start:end]
            test_kernel += (test_flat[:, start:end].double() - mean[start:end]) @ block.T
        train_view = torch.tensor(
            [[1.0 if inspected[offset]["view_position"] == "AP" else 0.0] for offset in train_idx],
            dtype=torch.float32,
        )
        test_view = torch.tensor(
            [[1.0 if inspected[offset]["view_position"] == "AP" else 0.0] for offset in test_idx],
            dtype=torch.float32,
        )
        view_pred = _log_odds_rows(
            llm, bridge_tokens(train_view, tokens[train_idx], test_view), positive_id, negative_id,
        )
        for local, offset in enumerate(test_idx):
            cross_view[eval_position[offset]] = view_pred[local]
        rank_plan = [(rank, rank, min(rank, pca_width)) for rank in FIXED_RANKS]
        rank_plan.append(("n_train_minus_1", max_rank, pca_width))
        for key, rank_order, width in rank_plan:
            pca_release = _log_odds_rows(
                llm,
                bridge_tokens(
                    pca_train_codes[:, :width], tokens[train_idx], pca_test_codes[:, :width],
                ),
                positive_id, negative_id,
            )
            pls_width = min(rank_order, left.shape[1])
            mapping = pls_mapping(gram.numpy(), left, right, pls_width)
            train_pls = torch.tensor(apply_pls(gram.numpy(), mapping), dtype=torch.float32)
            test_pls = torch.tensor(apply_pls(test_kernel.numpy(), mapping), dtype=torch.float32)
            pls_release = _log_odds_rows(
                llm, bridge_tokens(train_pls, tokens[train_idx], test_pls), positive_id, negative_id,
            )
            for local, offset in enumerate(test_idx):
                cross_scores[("pca", key)][eval_position[offset]] = pca_release[local]
                cross_scores[("pls", key)][eval_position[offset]] = pls_release[local]
        del train_flat, test_flat, gram, mean, pca_train_codes, pca_test_codes, test_kernel
        gc.collect()
        print(json.dumps({
            "fold": fold, "train_patients": len(train_idx), "scored_patients": len(test_idx),
        }), flush=True)
    if any(score is None for score in cross_view):
        raise SystemExit("cross-fitting left an analysis patient unscored")
    view_rows.append(_view_row("crossfit_5", eval_labels, native_scores, cross_view, eval_ids))
    for key in list(cross_scores):
        if any(score is None for score in cross_scores[key]):
            raise SystemExit("cross-fitting left an analysis patient unscored")
        basis, rank_key = key
        rank_order = min(full_ranks) if rank_key == "n_train_minus_1" else int(rank_key)
        cells.append(_cell(
            basis, "crossfit_5", str(rank_key), rank_order, eval_labels, eval_views,
            native_scores, cross_scores[key], eval_ids, cross_view,
        ))
    chosen = select_cell(cells)
    result = {
        "schema": "nih_prefix_grid_v1",
        "protocol_section": "2.25",
        "section_2_24_unchanged": True,
        "backend": BACKEND,
        "primary_endpoint": "pa_only",
        "margin": MARGIN,
        "analysis_patients": 506,
        "inspected_patients": 677,
        "development_patients": 72,
        "folds": FOLDS,
        "fold_rule": "sha256(FHEViT-nih14-v1:{patient_id}) mod 5",
        "pa_only_native_auc": pa_auc,
        "reserved_confirmation_used": False,
        "confirmation_scored": False,
        "cells": [{key: value for key, value in cell.items() if key != "scores"} for cell in cells],
        "view_only": [{key: value for key, value in row.items() if key != "scores"} for row in view_rows],
        "passing_ranks": [
            {"basis": cell["basis"], "bridge": cell["bridge"], "rank": cell["rank"]}
            for cell in cells if cell["passes"]
        ],
        "selected_cell": None if chosen is None else {
            "basis": chosen["basis"], "bridge": chosen["bridge"], "rank": chosen["rank"],
        },
        "grid_failed": chosen is None,
        "finite_state_symbol_tested": False,
        "predictor_retarget_started": False,
        "multilayer_perceptron_started": False,
        "medgemma_ran": False,
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "The pass rule is the 506-patient PA subset. AP remains exploratory.",
            "The 72-patient bridge is fit on the replication split only.",
            "Five-fold scores for the 506 are out of fold on the 677 inspected patients.",
            "The view-only prefix is scored on the full 506, not as a PA-only AUC.",
            "A passing cell is not scored on the reserved confirmation patients.",
        ],
    }
    destination = ROOT / "experiments" / "results" / "nih_prefix_grid_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "grid_failed": result["grid_failed"],
        "selected_cell": result["selected_cell"],
        "passing": result["passing_ranks"],
    }), flush=True)


if __name__ == "__main__":
    main()
