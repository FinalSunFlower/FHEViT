import hashlib
import io
import zipfile
from pathlib import Path

from experiments.nih_pneumonia import (
    NEGATIVE_LIMIT,
    POSITIVE_LIMIT,
    PROMPT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
    patient_order,
    rows_from_csv,
    select_probe,
)
from experiments.zip_member import extract_member, read_central_directory
from fpavlm.evaluation import roc_auc

ROOT = Path(__file__).resolve().parents[1]


def _row(patient, image, finding, follow="0", age="50", view="PA"):
    return {
        "patient_id": patient,
        "image_index": image,
        "finding_labels": finding,
        "follow_up": follow,
        "patient_age": age,
        "patient_gender": "F",
        "view_position": view,
    }


def test_a_constant_shift_does_not_change_auc():
    labels = [1, 1, 0, 0, 1]
    scores = [0.2, -0.1, 0.0, 0.4, 0.0]
    assert roc_auc(labels, scores) == roc_auc(labels, [score - 3.5 for score in scores])


def test_mixed_no_finding_and_pneumonia_is_excluded():
    rows = [
        _row("1", "a.png", "No Finding", "0"),
        _row("1", "b.png", "Pneumonia", "1"),
    ]
    positives, negatives, enough = select_probe(rows, positive_limit=1, negative_limit=1)
    assert positives == []
    assert negatives == []
    assert enough is False


def test_a_later_image_is_not_promoted_when_the_earliest_age_fails():
    rows = [
        _row("1", "early.png", "Pneumonia|Infiltration", "0", age="17"),
        _row("1", "later.png", "Pneumonia", "1", age="40"),
        _row("2", "n.png", "No Finding"),
    ]
    positives, negatives, enough = select_probe(rows, positive_limit=1, negative_limit=1)
    assert positives == []
    assert negatives[0]["patient_id"] == "2"
    assert enough is False


def test_comorbid_pneumonia_stays_and_the_earliest_film_is_kept():
    rows = [
        _row("9", "second.png", "Pneumonia", "2", age="60"),
        _row("9", "first.png", "Infiltration|Pneumonia", "1", age="40"),
        _row("9", "other.png", "Effusion", "0", age="39"),
    ]
    positives, _, enough = select_probe(rows, positive_limit=1, negative_limit=0)
    assert enough is True
    assert positives == [{
        "patient_id": "9",
        "image_index": "first.png",
        "finding_labels": "Infiltration|Pneumonia",
        "follow_up": 1,
        "patient_age": 40,
        "patient_gender": "F",
        "view_position": "PA",
        "label_pneumonia": 1,
    }]


def test_hash_order_not_numeric_order_fills_the_probe():
    rows = [
        _row("2", "p2.png", "Pneumonia"),
        _row("10", "p10.png", "Pneumonia"),
        _row("3", "n3.png", "No Finding"),
        _row("4", "n4.png", "No Finding"),
    ]
    positives, negatives, enough = select_probe(rows, positive_limit=1, negative_limit=1)
    assert enough is True
    pneumonia_ids = ["2", "10"]
    expected = min(pneumonia_ids, key=patient_order)
    assert positives[0]["patient_id"] == expected
    assert expected != min(pneumonia_ids, key=int)
    assert negatives[0]["patient_id"] == min(["3", "4"], key=patient_order)


def test_csv_header_with_a_bracketed_comma_keeps_the_label_columns():
    text = (
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,"
        "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]\n"
        "00000002_000.png,No Finding,0,2,80,M,PA,2500,2048,0.171,0.171\n"
    )
    rows = rows_from_csv(text)
    assert rows == [{
        "image_index": "00000002_000.png",
        "finding_labels": "No Finding",
        "follow_up": "0",
        "patient_id": "2",
        "patient_age": "80",
        "patient_gender": "M",
        "view_position": "PA",
    }]


def test_a_sealed_patient_drops_out_and_the_next_hash_takes_the_seat():
    rows = [_row(str(patient), f"{patient}.png", "Pneumonia") for patient in range(1, 5)]
    first, _, enough = select_probe(rows, positive_limit=1, negative_limit=0)
    assert enough is True
    sealed = first[0]["patient_id"]
    second, _, enough = select_probe(rows, positive_limit=1, negative_limit=0, exclude_ids={sealed})
    rest = [str(patient) for patient in range(1, 5) if str(patient) != sealed]
    assert enough is True
    assert second[0]["patient_id"] == min(rest, key=patient_order)
    assert second[0]["patient_id"] != sealed


def test_balanced_redraws_keep_a_perfect_ranking_and_the_seed():
    from experiments.qwen_baseline_stability import balanced_redraws

    labels = [1] * 72 + [0] * 72
    perfect = [1.0] * 72 + [0.0] * 72
    summary = balanced_redraws(labels, perfect)
    assert summary["draws"] == 200
    assert summary["percentile_2_5"] == 1.0
    assert summary["both_above_0_5"] is True
    mixed = [0.2] * 36 + [0.0] * 36 + [0.2] * 36 + [0.0] * 36
    assert balanced_redraws(labels, mixed)["aucs"] == balanced_redraws(labels, mixed)["aucs"]


def _slice(age, female, view, comorbid, group):
    rows = []
    for index in range(72):
        pneumonia = index < 36
        label = "Pneumonia" if pneumonia and not comorbid else (
            "Infiltration|Pneumonia" if pneumonia else "No Finding")
        rows.append({
            "patient_id": f"{group}-{index}",
            "patient_age": age,
            "patient_gender": "F" if female and index % 2 == 0 else "M",
            "view_position": view,
            "label_pneumonia": 1 if pneumonia else 0,
            "finding_labels": label,
        })
    return rows


def test_a_large_age_gap_is_flagged_and_a_matched_split_is_not():
    from experiments.nih_split_balance import THRESHOLD, assess

    shifted = assess(
        _slice(80, True, "PA", False, "dev"), _slice(30, True, "PA", False, "conf"), draws=200, seed=42)
    assert shifted["p_values"]["age_mean_difference"] < THRESHOLD
    assert "age_mean_difference" in shifted["flagged"]
    matched = assess(
        _slice(50, True, "PA", False, "dev"), _slice(50, True, "PA", False, "conf"), draws=200, seed=42)
    assert matched["imbalanced"] is False
    assert matched["observed"]["subtype_l1"] == 0.0


def test_rbf_alone_calls_the_linear_probe_insufficient():
    from experiments.validate_nih_pooling_capacity import capacity_flags

    assert capacity_flags(False, True, True)["linear_capacity_insufficient"] is False
    assert capacity_flags(True, False, False)["linear_capacity_insufficient"] is True
    assert capacity_flags(False, False, True)["pooling_loss_supported"] is True
    assert capacity_flags(True, True, False)["pooling_loss_supported"] is False


def test_protocol_locks_pooling_capacity_and_the_qwen_redraw():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    kernel = protocol.split("## 2.13", 1)[1].split("## 2.14", 1)[0]
    attention = protocol.split("## 2.14", 1)[1].split("## 2.15", 1)[0]
    redraw = protocol.split("## 2.15", 1)[1].split("## 3.", 1)[0]
    assert "RBF kernel ridge" in kernel
    assert "degree-2 polynomial features on 16 principal components" in kernel
    assert "only when the RBF prefix beats the linear prefix" in kernel
    assert "A multilayer" in kernel
    assert "perceptron is not started" in kernel
    assert "softmax" in attention
    assert "100 epochs" in attention
    assert "A regional release is not started" in attention
    assert "does not overwrite the section 2.7" in redraw
    assert "Development release scores are" in redraw
    assert "not computed" in redraw


def test_one_noninferior_rank_keeps_the_coordinate_path_open():
    from experiments.validate_nih_oracle_auc import coordinate_path_remains_open

    assert coordinate_path_remains_open([False, False, False]) is False
    assert coordinate_path_remains_open([False, True, False]) is True


def test_protocol_locks_oracle_auc_alignment():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.16", 1)[1].split("## 2.17", 1)[0]
    assert "Ranks are 4, 16, and 64" in section
    assert "AUC(oracle) - AUC(native)" in section
    assert "`internvl2-2b`" in section
    assert "Qwen is not fit" in section
    assert "only when at least one rank is non-inferior" in section
    assert "Clinical-label training is not started" in section


def test_curve_peak_keeps_the_smaller_rank_on_a_tie():
    from experiments.validate_nih_oracle_width_neighbors import curve_peak

    points = [(4, 0.1), (8, 0.4), (16, 0.4), (32, 0.2), (64, 0.3)]
    assert curve_peak(points) == 8
    assert curve_peak([(4, 0.1), (8, 0.2), (16, 0.9), (32, 0.3), (64, 0.4)]) == 16
    try:
        curve_peak([(4, 0.1), (8, 0.2)])
    except ValueError:
        return
    raise AssertionError("a partial curve must not name a peak")


def test_protocol_locks_oracle_width_neighbors():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.17", 1)[1].split("## 2.18", 1)[0]
    assert "adds ranks 8 and 32" in section
    assert "not refit and are not rescored" in section
    assert "AUC(oracle) - AUC(native)" in section
    assert "A tie keeps the" in section
    assert "smaller rank" in section
    assert "does not retarget the predictor" in section
    assert "A privacy-channel redesign is not" in section
    assert "started" in section


def test_protocol_locks_self_prediction_and_split_balance():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    pooled = protocol.split("## 2.10", 1)[1].split("## 2.11", 1)[0]
    balance = protocol.split("## 2.11", 1)[1].split("## 2.12", 1)[0]
    later = protocol.split("## 2.12", 1)[1].split("## 3.", 1)[0]
    assert "mean of the 256 frozen native visual tokens" in pooled
    assert "unpooled token matrix is not the predictor" in pooled
    assert "section 2.8 patient lists" in pooled
    assert "A trainable adapter is not started" in pooled
    assert "The PCA target is not" in pooled
    assert "redefined in this section" in pooled
    assert "age, sex, view, and the subtype omnibus" in balance
    assert "`0.0125`" in balance
    assert "A stratified split is not" in balance
    assert "started in this section" in balance
    assert "not made from that one split" in later
    assert "That redraw is not run" in later


def test_protocol_locks_predictor_capacity_and_qwen_stability():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    capacity = protocol.split("## 2.8", 1)[1].split("## 2.9", 1)[0]
    stability = protocol.split("## 2.9", 1)[1].split("## 3.", 1)[0]
    assert "densenet121-res224-all" in capacity
    assert "leave-one-patient-out" in capacity
    assert "4, 8, 12, 16, 20, 36, and 72" in capacity
    assert "`internvl2-2b`" in capacity
    assert "A new finite-state alphabet is not started" in capacity
    assert "Two hundred draws, seed 42" in stability
    assert "36 pneumonia patients and 36 `No Finding` patients" in stability
    assert "both lie strictly above 0.5" in stability


def test_protocol_locks_same_sample_release_comparison():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.7", 1)[1].split("## 3.", 1)[0]
    assert "the next 36 pneumonia" in section
    assert "the next 36 `No Finding` patients" in section
    assert "AUC(release) - AUC(native)" in section
    assert "greater than" in section
    assert "`-0.05`" in section
    assert "`internvl2-2b` is fit first" in section
    assert "MedGemma is not fit" in section


def test_protocol_locks_the_replication_slice():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.6", 1)[1].split("## 3.", 1)[0]
    assert "next 36 pneumonia patients and the next 36" in section
    assert f"`{PROMPT}`" in section
    assert 'sha256("FHEViT-nih14-v1:{patient_id}")' in section
    assert REPLICATION_POSITIVE == 36
    assert REPLICATION_NEGATIVE == 36


def test_protocol_locks_the_prompt_and_the_probe_counts():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.5", 1)[1].split("## 3.", 1)[0]
    assert f"`{PROMPT}`" in section
    assert "first 14 pneumonia patients and the first 13" in section
    assert POSITIVE_LIMIT == 14
    assert NEGATIVE_LIMIT == 13
    assert "Constant shift" in protocol
    assert hashlib.sha256(b"FHEViT-nih14-v1:1").hexdigest() == patient_order("1")


def test_a_zip_member_is_read_from_byte_ranges():
    buffer = io.BytesIO()
    payload = b"\x89PNG\r\n\x1a\nhello"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("images/a.png", payload)
        archive.writestr("b.txt", b"skip")
    data = buffer.getvalue()

    def fetch(start, end):
        return data[start:end + 1]

    members = read_central_directory(fetch, len(data))
    assert extract_member(fetch, members["images/a.png"]) == payload


def test_curve_pairs_keep_the_smaller_rank_first_and_count_zero():
    from experiments.validate_nih_curve_intervals import includes_zero, rank_pairs

    pairs = rank_pairs()
    assert len(pairs) == 10
    assert pairs[0] == (4, 8)
    assert (8, 4) not in pairs
    assert includes_zero(-0.01, 0.02) is True
    assert includes_zero(0.0, 0.02) is True
    assert includes_zero(0.01, 0.02) is False


def test_sample_size_is_the_smallest_even_n_at_the_locked_power():
    from experiments.nih_sample_size import planned_even_n, planning_se, power_noninferior, se_at_n

    assert planning_se([0.04, 0.05, 0.06, 0.07, 0.08]) == 0.06
    size = planned_even_n(0.055, 0.0)
    assert size % 2 == 0
    assert power_noninferior(se_at_n(0.055, size), 0.0) >= 0.80
    assert power_noninferior(se_at_n(0.055, size - 2), 0.0) < 0.80


def test_attribute_calls_use_the_locked_bounds():
    from experiments.validate_nih_attribute_inference import above_chance, beats_training_mean

    assert above_chance(0.51) is True
    assert above_chance(0.5) is False
    assert beats_training_mean(0.01) is True
    assert beats_training_mean(0.0) is False


def test_protocol_locks_curve_intervals_power_and_attribute_inference():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    intervals = protocol.split("## 2.18", 1)[1].split("## 2.19", 1)[0]
    power = protocol.split("## 2.19", 1)[1].split("## 2.20", 1)[0]
    attack = protocol.split("## 2.20", 1)[1].split("## 2.21", 1)[0]
    assert "Those patients are a selection set" in intervals
    assert "AUC(smaller) - AUC(larger)" in intervals
    assert "does not select a rank" in intervals
    assert "median of the five" in power
    assert "`0` and `+0.03`" in power
    assert "Images are not downloaded" in power
    assert "ranks 4, 8," in attack
    assert "and 16" in attack
    assert "greater" in attack
    assert "than `0.5`" in attack
    assert "A reconstruction attack is not started" in attack


def test_image_level_keeps_a_mixed_patient_as_positive_and_does_not_promote():
    from experiments.nih_pneumonia import classify_image_level, classify_patient

    mixed = [
        _row("1", "a.png", "No Finding", "0"),
        _row("1", "b.png", "Pneumonia", "1", age="40"),
    ]
    assert classify_patient(mixed) is None
    label, row = classify_image_level(mixed)
    assert label == 1 and row["image_index"] == "b.png"
    blocked = [
        _row("1", "early.png", "Pneumonia", "0", age="17"),
        _row("1", "later.png", "Pneumonia", "1", age="40"),
    ]
    assert classify_image_level(blocked) is None
    negative = [_row("2", "n.png", "No Finding"), _row("2", "n2.png", "No Finding", "1")]
    assert classify_image_level(negative)[0] == 0


def test_shortcut_and_sample_gate_use_the_locked_conjunctions():
    from experiments.nih_image_level_census import gate_open
    from experiments.validate_nih_view_shortcut import shortcut_supported, view_score

    assert view_score("AP") == 1 and view_score("PA") == 0
    assert shortcut_supported(True, True) is True
    assert shortcut_supported(True, False) is False
    assert shortcut_supported(False, True) is False
    assert gate_open(253, 253) is True
    assert gate_open(252, 10000) is False
    assert gate_open(10000, 252) is False


def test_protocol_locks_the_view_shortcut_and_the_image_level_pool():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    shortcut = protocol.split("## 2.21", 1)[1].split("## 2.22", 1)[0]
    revised = protocol.split("## 2.22", 1)[1].split("## 2.23", 1)[0]
    assert "1 for AP and 0 for PA" in shortcut
    assert "that interval lies above zero" in shortcut
    assert "greater than `0.5`" in shortcut
    assert "component correlations are not computed" in shortcut
    assert "Direction nulling is not started" in shortcut
    assert "not an AUC" in revised
    assert "are not relabeled" in revised
    assert "253 patients in each" in revised
    assert "primary rank is the" in revised
    assert "smallest, rank 4" in revised
    assert "Rank 8 is the only secondary" in revised
    assert "is not widened" in revised


def test_comorbidity_call_requires_bare_pneumonia_to_miss_chance():
    from experiments.validate_nih_sealed_strata import comorbidity_carries, positive_stratum

    assert positive_stratum("Pneumonia") == "bare"
    assert positive_stratum("Effusion|Pneumonia") == "comorbid"
    assert comorbidity_carries(True, False) is True
    assert comorbidity_carries(True, True) is False
    assert comorbidity_carries(False, False) is False


def test_protocol_locks_scored_set_strata_and_the_prefix_grid():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    strata = protocol.split("## 2.23", 1)[1].split("## 2.24", 1)[0]
    grid = protocol.split("## 2.24", 1)[1].split("## 2.25", 1)[0]
    assert "no longer a confirmation set" in strata
    assert "two per" in strata
    assert "exactly `Pneumonia`" in strata
    assert "Comorbidity carries the native AUC" in strata
    assert "0.4 finite-state symbol is not tested again" in strata
    assert "unsupervised PCA and a supervised PLS" in grid
    assert "4," in grid
    assert "8, 16, 32, 64" in grid
    assert "smallest rank, then the supervised basis, then the 72-patient" in grid
    assert "If no cell passes" in grid


def test_pa_pool_does_not_promote_a_later_pa_image():
    from experiments.nih_pa_only_power import include_in_pa_pool, planning_se, required_positives

    assert include_in_pa_pool("PA") is True
    assert include_in_pa_pool("AP") is False
    assert planning_se(0.02, 0.03) == 0.03
    assert required_positives(0.04, 129, 0.0) >= 129


def test_protocol_locks_the_pa_only_endpoint_and_the_reconstruction_ceiling():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    endpoint = protocol.split("## 2.25", 1)[1].split("## 2.26", 1)[0]
    attack = protocol.split("## 2.26", 1)[1].split("## 2.27", 1)[0]
    assert "Section 2.24 is unchanged" in endpoint
    assert "The primary endpoint is PA-only" in endpoint
    assert "no later film" in endpoint
    assert "promoted" in endpoint
    assert "greater than `-0.05`" in endpoint
    assert "0.708" in endpoint
    assert "does not score the grid" in endpoint
    assert "cannot reach 80% power" in endpoint
    assert "256 native tokens, their" in attack
    assert "ranks 4, 8, and 16" in attack
    assert "PSNR, SSIM, and" in attack
    assert "LPIPS" in attack


def test_protocol_locks_the_unlabeled_curve_and_the_strong_attacker():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    curve = protocol.split("## 2.27", 1)[1].split("## 2.28", 1)[0]
    attack = protocol.split("## 2.28", 1)[1].split("## 2.29", 1)[0]
    assert "do not use the" in curve
    assert "pneumonia label" in curve
    assert "not promoted" in curve
    assert "first 8000" in curve
    assert "64, 128, 256" in curve
    assert "greater than `-0.05`" in curve
    assert "than or equal to `0`" in curve
    assert "at least 0.8" in curve
    assert "`-0.02`" in curve
    assert "before any new cell is scored" in curve
    assert "1 by 1" in attack
    assert "kernel 4 and padding 1" in attack
    assert "learning rate `0.001`" in attack
    assert "LPIPS is not a decision metric" in attack
    assert "view-conditional mean" in attack
    assert "epsilon-3" in attack


def test_unlabeled_film_is_not_promoted_and_the_power_reading_is_normal():
    import statistics

    from experiments.nih_full_rank_power import se_from_one_sided_bound
    from experiments.nih_unlabeled_pool import choose_unlabeled_film

    early = _row("1", "a.png", "Cardiomegaly", follow="0", age="10", view="AP")
    later = _row("1", "b.png", "Cardiomegaly", follow="1", age="40", view="PA")
    assert choose_unlabeled_film([early, later]) is None
    kept = choose_unlabeled_film([
        _row("2", "c.png", "Pneumonia", follow="0", age="50", view="AP"),
        _row("2", "d.png", "No Finding", follow="1", age="50", view="PA"),
    ])
    assert kept["image_index"] == "c.png"
    se = se_from_one_sided_bound(-0.020, -0.054)
    assert abs(se - (0.034 / statistics.NormalDist().inv_cdf(0.95))) < 1e-12


def test_downloaded_prefix_does_not_invent_a_3000_cell():
    import torch

    from experiments.validate_nih_prefix_grid import bridge_tokens
    from experiments.validate_nih_unlabeled_bridge import (
        bridge_from_rows,
        rank_plan,
        scheduled_prefixes,
    )

    assert scheduled_prefixes(3000) == (500, 2000)
    assert rank_plan(500) == (64, 128, 256, 499)
    assert rank_plan(2000) == (64, 128, 256, 1999)
    torch.manual_seed(0)
    tokens = torch.randn(6, 2, 4)
    code = torch.randn(6, 2)
    reference = bridge_tokens(code[:4], tokens[:4], code[4:]).reshape(2, -1)
    produced = bridge_from_rows(
        code[:4].numpy(), tokens[:4].reshape(4, 8).numpy(), code[4:].numpy(),
    )
    assert torch.allclose(reference, produced, atol=1e-5)


def test_protocol_keeps_the_confirmation_closed_until_the_release_is_fixed():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    deviation = protocol.split("## 2.29", 1)[1].split("## 2.30", 1)[0]
    redraw = protocol.split("## 2.30", 1)[1].split("## 2.31", 1)[0]
    frontier = protocol.split("## 2.31", 1)[1].split("## 2.32", 1)[0]
    students = protocol.split("## 2.32", 1)[1].split("## 2.33", 1)[0]
    assert "not a learning curve" in deviation
    assert "Do not extrapolate" in deviation
    assert "not an FHE result" in deviation
    assert "stays closed" in deviation
    assert "rank 1999" in redraw
    assert "at least 3 of the 5" in redraw
    assert "512, 1024, and 1999" in frontier
    assert "`0`, `0.25`, `0.5`, and `1`" in frontier
    assert "view-conditional mean" in frontier
    assert "No CKKS run belongs to this section" in students
    assert "`L` of 2, 4, and 8" in students
    assert "medical-polynomial-vit" in students
    assert "[1, 1, 1]" in students
    assert "If no student passes, no circuit is started" in students

    import numpy as np

    from experiments.validate_nih_rank1999_redraws import training_draws

    draws = training_draws(3000, 2000, 5, 42)
    assert len(draws) == 5
    assert all(len(set(draw.tolist())) == 2000 for draw in draws)
    assert len({draw.tobytes() for draw in draws}) == 5
    again = training_draws(3000, 2000, 5, 42)
    assert all(np.array_equal(left, right) for left, right in zip(draws, again))


def test_prefix_grid_tie_break_and_pls_span():
    import numpy as np
    import torch
    from sklearn.cross_decomposition import PLSRegression

    from experiments.validate_nih_prefix_grid import (
        apply_pls,
        bridge_tokens,
        fold_id,
        gram_and_mean,
        pca_gram_scores,
        pca_svd_scores,
        pls_coefficients,
        pls_mapping,
        select_cell,
    )
    from fpavlm import DiagnosticPrefixProjector, PrefixSpec

    assert fold_id("18133") == fold_id("18133")
    assert 0 <= fold_id("18133") < 5
    cells = [
        {"passes": True, "basis": "pca", "bridge": "development_72", "rank": "8", "rank_order": 8},
        {"passes": True, "basis": "pls", "bridge": "crossfit_5", "rank": "4", "rank_order": 4},
        {"passes": True, "basis": "pca", "bridge": "development_72", "rank": "4", "rank_order": 4},
        {"passes": False, "basis": "pls", "bridge": "development_72", "rank": "4", "rank_order": 4},
        {"passes": True, "basis": "view", "bridge": "development_72", "rank": "ap_indicator", "rank_order": 1},
    ]
    chosen = select_cell(cells)
    assert chosen["basis"] == "pls" and chosen["bridge"] == "crossfit_5" and chosen["rank"] == "4"
    same_rank = [
        {"passes": True, "basis": "pls", "bridge": "crossfit_5", "rank": "4", "rank_order": 4},
        {"passes": True, "basis": "pls", "bridge": "development_72", "rank": "4", "rank_order": 4},
    ]
    assert select_cell(same_rank)["bridge"] == "development_72"
    assert select_cell([{"passes": False, "basis": "pca", "bridge": "development_72", "rank_order": 4}]) is None
    wide = [
        {"passes": True, "basis": "pca", "bridge": "development_72", "rank": "n_train_minus_1", "rank_order": 71},
        {"passes": True, "basis": "pca", "bridge": "crossfit_5", "rank": "n_train_minus_1", "rank_order": 500},
    ]
    assert select_cell(wide)["rank_order"] == 71

    rng = np.random.default_rng(0)
    features = rng.normal(size=(30, 12))
    labels = features[:, 0] + 0.05 * rng.normal(size=30)
    train, held = features[:20], features[20:]
    mean = train.mean(0)
    centered = train - mean
    kernel = centered @ centered.T
    left, right = pls_coefficients(kernel, labels[:20], 4)
    mapping = pls_mapping(kernel, left, right, 4)
    fitted = PLSRegression(n_components=4, scale=False).fit(train, labels[:20])
    for mine, reference in (
        (apply_pls(kernel, mapping), fitted.transform(train)),
        (apply_pls((held - mean) @ centered.T, mapping), fitted.transform(held)),
    ):
        predicted = mine @ np.linalg.lstsq(mine, reference, rcond=None)[0]
        assert np.linalg.norm(predicted - reference) / np.linalg.norm(reference) < 1e-6
    short = pls_mapping(kernel, left, right, 2)
    reference = PLSRegression(n_components=2, scale=False).fit(train, labels[:20]).transform(train)
    predicted = apply_pls(kernel, short) @ np.linalg.lstsq(apply_pls(kernel, short), reference, rcond=None)[0]
    assert np.linalg.norm(predicted - reference) / np.linalg.norm(reference) < 1e-6

    torch.manual_seed(0)
    native = torch.randn(8, 2, 3)
    code = torch.randn(8, 2)
    produced = bridge_tokens(code[:5], native[:5], code[5:])
    spec = PrefixSpec("internvl2-2b", (2, 1), 2, 3, "ab" * 32, "cd" * 32)
    projector = DiagnosticPrefixProjector(spec).fit(code[:5].reshape(5, 2, 1), native[:5], sigma=0.)
    assert torch.allclose(produced, projector(code[5:].reshape(3, 2, 1)), atol=1e-5)
    svd_train, svd_test = pca_svd_scores(native[:5], native[5:], 4)
    flat_train = native[:5].reshape(5, -1)
    flat_test = native[5:].reshape(3, -1)
    column_mean, gram = gram_and_mean(flat_train)
    gram_train, gram_test, width = pca_gram_scores(flat_train, flat_test, column_mean, gram, 4)
    assert width == 4
    for left_code, right_code in ((svd_train, gram_train), (svd_test, gram_test)):
        sign = torch.sign((left_code * right_code).sum(0))
        sign[sign == 0] = 1
        assert torch.allclose(left_code * sign, right_code, atol=1e-4)


def test_frontier_revision_uses_the_refit_and_four_of_five_draws():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    revision = protocol.split("## 2.33", 1)[1].split("## 2.34", 1)[0]
    assert "2026-10-10" in revision
    assert "1 + n * v" in revision
    assert "at least 4 of the 5" in revision
    assert "not differential privacy" in revision
    assert "256 native tokens" in revision
    assert "greater than `0`" in revision
    assert "not an independent point" in revision
    assert "`-0.025`" in revision
    assert "softmax, LayerNorm, or GELU" in revision
    assert "does not start a CKKS circuit" in revision
    assert "depth budget 6" in revision

    from fpavlm.cipher_native.med_fhe_vit_block import MedFHEBlock

    block = MedFHEBlock(
        dim=64, num_tokens=4, activation_type="polyact", poly_act_degree=1,
        poly_norm_degree=2, enable_token_mix=True, use_cadn=False, num_regions=1,
    )
    assert block.multiplicative_depth() == 6


def test_frontier_revision_keeps_one_draw_set_and_the_depth_gate():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    revision = protocol.split("## 2.34", 1)[1].split("## 2.35", 1)[0]
    assert "same five" in revision
    assert "`1e-6`" in revision
    assert "divisor `n`" in revision
    assert "32, 64, and 128" in revision
    assert "4, 8, and 16" in revision
    assert "does not bootstrap" in revision
    assert "output depth `5`" in revision
    assert "depth budget `6`" in revision
    assert "expected result" in revision

    import torch

    from experiments.measure_poly_block_grid import log_slope
    from experiments.validate_nih_coordinate_frontier import feature_gram

    assert abs(log_slope((1, 10, 100), (1, 100, 10000)) - 2) < 1e-12
    torch.manual_seed(0)
    code = torch.randn(5, 2, dtype=torch.float64)
    centered = code - code.mean(0)
    sigma = 0.5
    variance = torch.full((2,), sigma ** 2)
    reference = centered.T @ centered + (1 + 5 * sigma ** 2) * torch.eye(2)
    assert torch.allclose(feature_gram(centered, variance), reference)
    zeros = feature_gram(centered, torch.zeros(2))
    assert torch.allclose(zeros, centered.T @ centered + torch.eye(2))


def test_student_depth_drops_the_layer_stacks_before_auc():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.35", 1)[1].split("## 2.36", 1)[0]
    assert "at most `5`" in section
    assert "are not scored" in section
    assert "bilinear 64 by 64" in section
    assert "depth `1`" in section
    assert "depth `2`" in section
    assert "not circuit candidates" in section
    assert "no feasible student" in section
    assert "not a pure FHE claim" in section

    from experiments.nih_student_depth import student_plans

    plans = {plan["name"]: plan for plan in student_plans()}
    assert plans["internvit_layers_2"]["depth"] == 10
    assert plans["internvit_layers_4"]["score_auc"] is False
    assert plans["internvit_layers_8"]["score_auc"] is False
    assert plans["pixel_linear"]["depth"] == 1
    assert plans["pixel_square"]["depth"] == 2
    assert plans["pixel_linear"]["circuit_candidate"] is True
    assert plans["layer1_linear"]["circuit_candidate"] is False
    assert plans["layer1_linear"]["score_auc"] is True


def test_passing_release_attack_and_later_sections_are_locked():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    attack = protocol.split("## 2.36", 1)[1].split("## 2.37", 1)[0]
    noise = protocol.split("## 2.37", 1)[1].split("## 2.38", 1)[0]
    pixels = protocol.split("## 2.38", 1)[1].split("## 2.39", 1)[0]
    tower = protocol.split("## 2.39", 1)[1].split("## 3.", 1)[0]
    assert "rank 1024 and rank 1999" in attack
    assert "view-conditional mean" in attack
    assert "greater than `0`" in attack
    assert "`0.05` and `0.1`" in noise
    assert "at least 4 of the 5" in noise
    assert "64 by 64" in pixels
    assert "still run" in pixels
    assert "not a circuit candidate" in tower
    assert "73 bootstraps" in tower
    assert "multiplicative depth is `172`" in tower
    assert "0.1365" in tower

    import torch

    from experiments.validate_nih_passing_release_attack import Decoder

    tokens = Decoder("tokens", 2048)
    assert tuple(tokens(torch.zeros(2, 256, 2048)).shape) == (2, 1, 64, 64)
    coordinates = Decoder("coordinates", 1999)
    assert tuple(coordinates(torch.zeros(2, 1999)).shape) == (2, 1, 64, 64)

    from experiments.validate_nih_fine_noise import _budget
    from experiments.validate_nih_pixel_students import coordinate_map

    summaries = [
        {"rank": 1024, "scale": 0.05, "passes": True},
        {"rank": 1024, "scale": 0.1, "passes": False},
        {"rank": 1999, "scale": 0.05, "passes": False},
        {"rank": 1999, "scale": 0.1, "passes": True},
    ]
    assert _budget(1024, True, summaries) == 0.05
    assert _budget(1999, True, summaries) == 0.0
    assert _budget(1024, False, summaries) is None
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    outcome = torch.tensor([[1.0], [2.0], [3.0]])
    source_mean, outcome_mean, weight = coordinate_map(source, outcome)
    centered = source.double() - source_mean
    gram = centered.T @ centered + torch.eye(2, dtype=torch.float64)
    expected = torch.linalg.solve(gram, centered.T @ (outcome.double() - outcome_mean))
    assert torch.allclose(weight, expected)

    from experiments.validate_nih_poly_tower import BOOTSTRAPS, MULTIPLICATIVE_DEPTH, gelu_polynomial, rsqrt_polynomial

    assert BOOTSTRAPS == 73
    assert MULTIPLICATIVE_DEPTH == 172
    assert abs(float(gelu_polynomial(torch.tensor(0.0))) - 0.1365) < 1e-8
    assert abs(float(rsqrt_polynomial(torch.tensor(1.0))) - 1.0) < 1e-8
