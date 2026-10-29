from pathlib import Path

from experiments.openi_negatives import (
    ALLOWED_IMPRESSIONS,
    is_clean_normal,
    normalize_impression,
    select_negative_pool,
)

ROOT = Path(__file__).resolve().parents[1]


def test_redacted_normal_chest_matches_the_allow_list():
    assert normalize_impression("Normal chest x-XXXX.") == "normal chest xray"
    assert is_clean_normal({
        "problems": "normal", "mesh": "normal", "impression": "Normal chest x-XXXX.",
    })


def test_another_diagnosis_tag_blocks_a_normal_impression():
    record = {
        "problems": "Cardiomegaly",
        "mesh": "Cardiomegaly/borderline",
        "impression": "No acute pulmonary findings.",
    }
    assert not is_clean_normal(record)


def test_an_extra_impression_sentence_is_not_on_the_allow_list():
    record = {
        "problems": "normal",
        "mesh": "normal",
        "impression": "No acute cardiopulmonary abnormality. Pleural effusion.",
    }
    assert not is_clean_normal(record)


def test_pool_keeps_uid_order_and_drops_used_reports():
    records = [
        {"uid": "3", "problems": "normal", "mesh": "normal",
         "impression": "No acute cardiopulmonary abnormality.", "image_url": "c.png"},
        {"uid": "1", "problems": "normal", "mesh": "normal",
         "impression": "Normal chest.", "image_url": "a.png"},
        {"uid": "2", "problems": "normal", "mesh": "normal",
         "impression": "Normal study.", "image_url": "b.png"},
    ]
    chosen, enough = select_negative_pool(records, used_uids={"2"})
    assert [row["uid"] for row in chosen] == ["1", "3"]
    assert enough is False


def test_protocol_lists_every_allowed_impression():
    protocol = (ROOT / "EVALUATION_PROTOCOL.md").read_text(encoding="utf-8")
    section = protocol.split("## 2.2", 1)[1].split("## 3.", 1)[0]
    missing = [phrase for phrase in ALLOWED_IMPRESSIONS if f"`{phrase}`" not in section]
    assert missing == []
