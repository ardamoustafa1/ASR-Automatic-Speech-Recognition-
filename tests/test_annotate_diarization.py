"""Tests for the human-annotation CSV -> RTTM converter used to build real
diarization reference data (scripts/annotate_diarization.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.annotate_diarization import csv_to_rttm  # noqa: E402


def test_csv_to_rttm_basic(tmp_path):
    csv_path = tmp_path / "call_001.csv"
    csv_path.write_text(
        "start_sec,end_sec,speaker\n0.0,3.2,agent\n3.4,7.9,customer\n8.1,8.6,agent\n",
        encoding="utf-8",
    )

    lines = csv_to_rttm(csv_path, uri="call_001")
    assert len(lines) == 3
    assert lines[0] == "SPEAKER call_001 1 0.000 3.200 <NA> <NA> agent <NA> <NA>"
    assert lines[1] == "SPEAKER call_001 1 3.400 4.500 <NA> <NA> customer <NA> <NA>"
    assert lines[2] == "SPEAKER call_001 1 8.100 0.500 <NA> <NA> agent <NA> <NA>"


def test_csv_to_rttm_sorts_out_of_order_rows(tmp_path):
    csv_path = tmp_path / "call_002.csv"
    csv_path.write_text(
        "start_sec,end_sec,speaker\n5.0,6.0,customer\n0.0,2.0,agent\n",
        encoding="utf-8",
    )
    lines = csv_to_rttm(csv_path, uri="call_002")
    assert "0.000" in lines[0]
    assert "5.000" in lines[1]


def test_csv_to_rttm_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("start,end,who\n0.0,1.0,agent\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header columns"):
        csv_to_rttm(csv_path, uri="bad")


def test_csv_to_rttm_rejects_end_before_start(tmp_path):
    csv_path = tmp_path / "bad2.csv"
    csv_path.write_text("start_sec,end_sec,speaker\n5.0,3.0,agent\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be after"):
        csv_to_rttm(csv_path, uri="bad2")


def test_csv_to_rttm_rejects_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("start_sec,end_sec,speaker\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no annotation rows"):
        csv_to_rttm(csv_path, uri="empty")
