# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the evaluate module and the hardness knob.

The evaluate metrics are checked on a hand-built fixture with known
precision/recall/F1. The hardness knob is checked through config presets and a
generator smoke test that confirms decoy legitimate cycles are produced and
fraud amounts are jittered.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile

import pytest

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.evaluate import (
    FraudRing,
    discover_account_universe,
    evaluate,
    evaluate_dataset,
    load_flagged_accounts,
    load_fraud_rings,
    main,
    run_cli,
)
from gen_fraud_graph.typologies import FraudRingGenerator


@pytest.fixture()
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="gen_fraud_graph_eval_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Metric computation on a known fixture
# ---------------------------------------------------------------------------


def test_evaluate_known_fixture():
    """Hand-built rings and flags give exactly computable metrics.

    Two rings: R1 = {a1, a2, a3} (fully flagged), R2 = {b1, b2, b3, b4}
    (missed). One spurious flag x1 (a false positive).

    Account level: tp=3, fp=1, fn=4 -> precision 0.75, recall 3/7, f1 6/11.
    Ring level (threshold 1.0): 1 of 2 rings detected; ring_fp = acc_fp = 1
    -> precision 0.5, recall 0.5, f1 0.5.
    """
    rings = [
        FraudRing("R1", frozenset({"a1", "a2", "a3"})),
        FraudRing("R2", frozenset({"b1", "b2", "b3", "b4"})),
    ]
    flagged = {"a1", "a2", "a3", "x1"}

    m = evaluate(rings, flagged)

    assert m["account"]["true_positives"] == 3
    assert m["account"]["false_positives"] == 1
    assert m["account"]["false_negatives"] == 4
    assert m["account"]["precision"] == pytest.approx(0.75)
    assert m["account"]["recall"] == pytest.approx(3 / 7)
    assert m["account"]["f1"] == pytest.approx(6 / 11)

    assert m["ring"]["detected"] == 1
    assert m["ring"]["total"] == 2
    assert m["ring"]["precision"] == pytest.approx(0.5)
    assert m["ring"]["recall"] == pytest.approx(0.5)
    assert m["ring"]["f1"] == pytest.approx(0.5)


def test_evaluate_partial_ring_threshold():
    """A ring threshold below 1.0 counts partially flagged rings as detected."""
    rings = [FraudRing("R1", frozenset({"a1", "a2", "a3", "a4"}))]
    flagged = {"a1", "a2"}  # half the ring

    strict = evaluate(rings, flagged, ring_threshold=1.0)
    assert strict["ring"]["detected"] == 0

    lenient = evaluate(rings, flagged, ring_threshold=0.5)
    assert lenient["ring"]["detected"] == 1


def test_evaluate_perfect_detection():
    """Flagging exactly the fraud accounts gives precision=recall=f1=1.0."""
    rings = [FraudRing("R1", frozenset({"a1", "a2", "a3"}))]
    m = evaluate(rings, {"a1", "a2", "a3"})
    assert m["account"]["precision"] == pytest.approx(1.0)
    assert m["account"]["recall"] == pytest.approx(1.0)
    assert m["account"]["f1"] == pytest.approx(1.0)
    assert m["ring"]["detected"] == 1


def test_evaluate_true_negatives_with_universe():
    """A supplied account universe yields a true-negative count."""
    rings = [FraudRing("R1", frozenset({"a1", "a2"}))]
    universe = {"a1", "a2", "c1", "c2", "c3"}  # 3 legit accounts
    m = evaluate(rings, {"a1", "a2"}, account_universe=universe)
    assert m["confusion"]["true_negatives"] == 3


def test_evaluate_rejects_bad_threshold():
    """A ring threshold outside (0, 1] is rejected."""
    with pytest.raises(ValueError):
        evaluate([], set(), ring_threshold=0.0)
    with pytest.raises(ValueError):
        evaluate([], set(), ring_threshold=1.5)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def test_load_fraud_rings_roundtrip(tmp_dir):
    """A fraud_cases.csv is parsed back into FraudRing objects."""
    path = os.path.join(tmp_dir, "fraud_cases.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pattern_id", "start_acc_id", "pattern_type", "depth", "involved_accounts"])
        w.writerow(["pat_0", "acc_1", "cycle", "3", "acc_1|acc_2|acc_3"])
    rings = load_fraud_rings(path)
    assert len(rings) == 1
    assert rings[0].pattern_id == "pat_0"
    assert rings[0].accounts == frozenset({"acc_1", "acc_2", "acc_3"})


def test_load_flagged_accounts_plain_and_csv(tmp_dir):
    """Flagged accounts load from a plain list and from a CSV header column."""
    plain = os.path.join(tmp_dir, "flagged.txt")
    with open(plain, "w") as fh:
        fh.write("acc_1\nacc_2\n\nacc_3\n")
    assert load_flagged_accounts(plain) == {"acc_1", "acc_2", "acc_3"}

    csv_path = os.path.join(tmp_dir, "flagged.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account_id"])
        w.writerow(["acc_9"])
        w.writerow(["acc_8"])
    assert load_flagged_accounts(csv_path) == {"acc_9", "acc_8"}


def test_load_flagged_accounts_edge_csv(tmp_dir):
    """A CSV with src_id/dst_id columns flags both endpoints of each edge."""
    csv_path = os.path.join(tmp_dir, "edges.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["src_id", "dst_id", "amount"])
        w.writerow(["acc_1", "acc_2", "9999.00"])
        w.writerow(["acc_2", "acc_3", "9999.00"])
        w.writerow(["", "acc_4", "0"])  # blank src is ignored, dst still flagged
    assert load_flagged_accounts(csv_path) == {"acc_1", "acc_2", "acc_3", "acc_4"}


def test_load_flagged_accounts_csv_without_header(tmp_dir):
    """A comma-bearing file with no recognised id column reads the first column."""
    csv_path = os.path.join(tmp_dir, "unknown.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["foo", "bar"])  # unrecognised header -> treated as first-column data
        w.writerow(["acc_7", "ignored"])
        w.writerow(["", "skip"])  # blank first cell is skipped
    assert load_flagged_accounts(csv_path) == {"foo", "acc_7"}


def test_discover_account_universe(tmp_dir):
    """discover_account_universe reads account ids from <data>/accounts/*.csv."""
    assert discover_account_universe(tmp_dir) is None  # no accounts dir yet

    acc_dir = os.path.join(tmp_dir, "accounts")
    os.makedirs(acc_dir)
    with open(os.path.join(acc_dir, "accounts_0.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account_id", "name"])
        w.writerow(["acc_1", "Alice"])
        w.writerow(["acc_2", "Bob"])
    # A non-CSV file in the directory is ignored.
    with open(os.path.join(acc_dir, "README.txt"), "w") as fh:
        fh.write("not a csv")

    assert discover_account_universe(tmp_dir) == {"acc_1", "acc_2"}


def test_evaluate_ignores_empty_ring():
    """A ring with no involved accounts is skipped at the ring level."""
    rings = [
        FraudRing("R1", frozenset({"a1", "a2"})),
        FraudRing("R_empty", frozenset()),
    ]
    m = evaluate(rings, {"a1", "a2"})
    assert m["ring"]["detected"] == 1
    assert m["ring"]["total"] == 2


def test_evaluate_dataset_missing_fraud_cases(tmp_dir):
    """evaluate_dataset raises a clear error when fraud_cases.csv is absent."""
    flagged = os.path.join(tmp_dir, "flagged.txt")
    with open(flagged, "w") as fh:
        fh.write("acc_1\n")
    with pytest.raises(FileNotFoundError, match="fraud_cases.csv"):
        evaluate_dataset(tmp_dir, flagged)


def _build_dataset(root):
    """Write a minimal dataset (fraud/ + accounts/) and a flagged file.

    Ground truth: one ring {acc_1, acc_2}. Universe adds one legit acc_3.
    Flagged: acc_1, acc_2 (perfect detection, one true negative).
    """
    fraud_dir = os.path.join(root, "fraud")
    os.makedirs(fraud_dir)
    with open(os.path.join(fraud_dir, "fraud_cases.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pattern_id", "start_acc_id", "pattern_type", "depth", "involved_accounts"])
        w.writerow(["pat_0", "acc_1", "cycle", "2", "acc_1|acc_2"])
    acc_dir = os.path.join(root, "accounts")
    os.makedirs(acc_dir)
    with open(os.path.join(acc_dir, "accounts_0.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["account_id"])
        for aid in ("acc_1", "acc_2", "acc_3"):
            w.writerow([aid])
    flagged = os.path.join(root, "flagged.txt")
    with open(flagged, "w") as fh:
        fh.write("acc_1\nacc_2\n")
    return flagged


def test_run_cli_human_summary(tmp_dir, capsys):
    """run_cli prints a human-readable summary and returns exit code 0."""
    flagged = _build_dataset(tmp_dir)
    rc = run_cli(["--data", tmp_dir, "--flagged", flagged])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gen_fraud_graph evaluate" in out
    assert "Account level:" in out
    assert "Ring level" in out
    assert "tn=1" in out  # acc_3 is the single true negative


def test_run_cli_json_output(tmp_dir, capsys):
    """run_cli --json prints valid JSON metrics."""
    flagged = _build_dataset(tmp_dir)
    rc = run_cli(["--data", tmp_dir, "--flagged", flagged, "--json", "--ring-threshold", "0.5"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["account"]["f1"] == pytest.approx(1.0)
    assert payload["ring_threshold"] == pytest.approx(0.5)
    assert payload["confusion"]["true_negatives"] == 1


def test_main_entrypoint(tmp_dir, capsys, monkeypatch):
    """main() parses sys.argv and exits 0 on success."""
    flagged = _build_dataset(tmp_dir)
    monkeypatch.setattr(
        sys, "argv", ["gen-fraud-graph-evaluate", "--data", tmp_dir, "--flagged", flagged]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    assert "gen_fraud_graph evaluate" in capsys.readouterr().out


def test_evaluate_dataset_end_to_end(tmp_dir):
    """evaluate_dataset reads fraud_cases.csv and a flagged file from disk."""
    fraud_dir = os.path.join(tmp_dir, "fraud")
    os.makedirs(fraud_dir)
    with open(os.path.join(fraud_dir, "fraud_cases.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pattern_id", "start_acc_id", "pattern_type", "depth", "involved_accounts"])
        w.writerow(["pat_0", "acc_1", "cycle", "2", "acc_1|acc_2"])
    flagged = os.path.join(tmp_dir, "flagged.txt")
    with open(flagged, "w") as fh:
        fh.write("acc_1\nacc_2\n")
    m = evaluate_dataset(tmp_dir, flagged)
    assert m["account"]["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Hardness knob
# ---------------------------------------------------------------------------


def test_hardness_presets_wire_through_config():
    """Hardness presets populate the derived jitter/overlap/decoy fields."""
    low = Config(scale_factor=0.0001, embedding_provider="fake", hardness="low")
    assert (low.amount_jitter, low.ring_overlap, low.decoy_ratio) == (0.0, 0.0, 0.0)

    high = Config(scale_factor=0.0001, embedding_provider="fake", hardness="high")
    assert high.amount_jitter > 0.0
    assert high.ring_overlap > 0.0
    assert high.decoy_ratio > 0.0


def test_invalid_hardness_rejected():
    """An unknown hardness preset is rejected."""
    with pytest.raises(ValueError):
        Config(scale_factor=0.0001, embedding_provider="fake", hardness="extreme")


def test_generator_high_hardness_emits_decoys(tmp_dir):
    """At high hardness the ring generator writes decoy legitimate cycles and
    jitters fraud amounts so the sentinel is no longer a giveaway."""
    gen = FraudRingGenerator(
        num_rings=8,
        depth_range=(4, 5),
        amount_jitter=0.5,
        ring_overlap=0.5,
        decoy_ratio=1.0,
    )
    embedder = EmbeddingGenerator(provider="fake")
    gen.generate(
        max_account_id=10_000,
        start_tx_id=0,
        embedder=embedder,
        output_dir=tmp_dir,
        fmt="csv",
    )

    decoy_path = os.path.join(tmp_dir, "transactions", "transactions_decoy.csv")
    assert os.path.exists(decoy_path), "expected decoy legitimate cycles to be written"

    # Fraud amounts should not all equal the sentinel once jitter is on.
    fraud_tx = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
    with open(fraud_tx, newline="") as fh:
        reader = csv.DictReader(fh)
        amounts = {row["amount"] for row in reader if row.get("amount")}
    assert len(amounts) > 1, "jitter should spread fraud amounts across many values"


def test_low_hardness_is_backward_compatible(tmp_dir):
    """At low hardness no decoys are written and the sentinel amount is fixed."""
    gen = FraudRingGenerator(num_rings=5, depth_range=(4, 5))
    embedder = EmbeddingGenerator(provider="fake")
    gen.generate(
        max_account_id=10_000,
        start_tx_id=0,
        embedder=embedder,
        output_dir=tmp_dir,
        fmt="csv",
    )
    decoy_path = os.path.join(tmp_dir, "transactions", "transactions_decoy.csv")
    assert not os.path.exists(decoy_path)

    fraud_tx = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
    with open(fraud_tx, newline="") as fh:
        reader = csv.DictReader(fh)
        amounts = {row["amount"] for row in reader if row.get("amount")}
    assert len(amounts) == 1, "low hardness keeps a single sentinel fraud amount"
