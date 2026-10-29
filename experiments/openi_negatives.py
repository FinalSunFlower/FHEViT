"""Locked Open-I negative-study rule.

The allow-list matches EVALUATION_PROTOCOL.md section 2.2. It is applied before any
log-odds on the selected studies and is not widened after the eligibility count.
"""
from __future__ import annotations

import re

_REDACTED_XRAY = re.compile(r"x-x{2,}")
_REDACTED_WORD = re.compile(r"\bx{2,}\b")
_NON_WORD = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")

ALLOWED_IMPRESSIONS = frozenset({
    "normal chest",
    "normal chest xray",
    "normal chest radiograph",
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary abnormalities",
    "no acute cardiopulmonary finding",
    "no acute cardiopulmonary findings",
    "no acute cardiopulmonary disease",
    "no acute cardiopulmonary process",
    "no acute cardiopulmonary processes",
    "no evidence of acute cardiopulmonary disease",
    "no evidence of acute cardiopulmonary process",
    "no evidence of acute cardiopulmonary abnormality",
    "no evidence of acute cardiopulmonary finding",
    "no evidence of acute cardiopulmonary findings",
    "no acute intrathoracic process",
    "no acute intrathoracic abnormality",
    "no acute intrathoracic finding",
    "no acute intrathoracic findings",
    "no acute pulmonary finding",
    "no acute pulmonary findings",
    "no acute pulmonary disease",
    "no acute pulmonary abnormality",
    "no acute disease",
    "negative for acute cardiopulmonary abnormality",
    "negative for acute cardiopulmonary disease",
    "lungs are clear",
    "clear lungs",
    "normal study",
})
POOL_LIMIT = 20
POOL_MINIMUM = 10


def normalize_impression(text):
    lowered = (text or "").lower()
    lowered = _REDACTED_XRAY.sub("xray", lowered)
    lowered = _REDACTED_WORD.sub(" ", lowered)
    lowered = _NON_WORD.sub(" ", lowered)
    return _SPACE.sub(" ", lowered).strip()


def is_clean_normal(record):
    problems = (record.get("problems") or "").strip().lower()
    mesh = (record.get("mesh") or "").strip().lower()
    if problems != "normal" or mesh != "normal":
        return False
    return normalize_impression(record.get("impression")) in ALLOWED_IMPRESSIONS


def select_negative_pool(records, used_uids):
    """First 20 eligible uids. Fewer than 10 is a shortfall, not a wider rule."""
    used = {str(uid) for uid in used_uids}
    eligible = [
        record for record in records
        if str(record.get("uid")) not in used and is_clean_normal(record) and record.get("image_url")
    ]
    eligible.sort(key=lambda record: int(record["uid"]))
    if len(eligible) < POOL_MINIMUM:
        return eligible, False
    return eligible[:POOL_LIMIT], True
