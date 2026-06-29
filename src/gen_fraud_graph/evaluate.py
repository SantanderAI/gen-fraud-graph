# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Evaluate fraud-detector output against generated ground truth.

The generator's stated purpose is to benchmark fraud detectors, but it ships no
scoring.  This module closes that gap: given a generated dataset directory and a
set of detector-flagged account ids (or flagged edges), it computes precision,
recall and F1 at two granularities:

* **Account level**: does the detector flag the individual accounts that take
  part in a fraud ring?
* **Ring level**: is a whole fraud ring considered detected?  A ring counts as
  detected when at least a configurable fraction (``ring_threshold``, default
  ``1.0`` meaning all) of its involved accounts are flagged.

Ground truth is read from ``<data>/fraud/fraud_cases.csv`` whose schema is
``pattern_id,start_acc_id,pattern_type,depth,involved_accounts`` with
``involved_accounts`` a pipe-separated list of account ids.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FraudRing:
    """A single ground-truth fraud ring."""

    pattern_id: str
    accounts: frozenset[str]


# ---------------------------------------------------------------------------
# Loading ground truth and detector output
# ---------------------------------------------------------------------------


def load_fraud_rings(fraud_cases_path: str) -> list[FraudRing]:
    """Load fraud rings from a ``fraud_cases.csv`` file.

    Args:
        fraud_cases_path: Path to ``fraud_cases.csv``.

    Returns:
        A list of :class:`FraudRing`.
    """
    rings: list[FraudRing] = []
    with open(fraud_cases_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            involved = row.get("involved_accounts", "")
            accounts = frozenset(a for a in involved.split("|") if a)
            rings.append(FraudRing(pattern_id=row["pattern_id"], accounts=accounts))
    return rings


def load_flagged_accounts(flagged_path: str) -> set[str]:
    """Load detector-flagged account ids from *flagged_path*.

    The file may be:

    * a plain text file with one account id per line, or
    * a CSV whose first column (or an ``account_id`` / ``src_id`` / ``dst_id``
      column) holds account ids.  For edge rows (``src_id`` and ``dst_id``)
      both endpoints are treated as flagged.

    Blank lines and a leading header row named like an id column are ignored.

    Returns:
        The set of flagged account ids.
    """
    flagged: set[str] = set()
    with open(flagged_path, newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        has_comma = "," in sample
        if has_comma:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            id_cols = [c for c in fieldnames if c in ("account_id", "src_id", "dst_id", "id")]
            if id_cols:
                for row in reader:
                    for col in id_cols:
                        val = (row.get(col) or "").strip()
                        if val:
                            flagged.add(val)
                return flagged
            # No recognised header: fall back to first-column reading.
            fh.seek(0)
            plain = csv.reader(fh)
            for cells in plain:
                if cells and cells[0].strip():
                    flagged.add(cells[0].strip())
            return flagged

        for line in fh:
            val = line.strip()
            if val and val.lower() not in ("account_id", "id"):
                flagged.add(val)
    return flagged


def discover_account_universe(data_dir: str) -> set[str] | None:
    """Return the full set of account ids under ``<data_dir>/accounts``.

    Used to populate true negatives in the confusion summary.  Returns *None*
    when no accounts directory is present (true negatives are then unknown).
    """
    acc_dir = os.path.join(data_dir, "accounts")
    if not os.path.isdir(acc_dir):
        return None
    universe: set[str] = set()
    for name in os.listdir(acc_dir):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(acc_dir, name), newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                aid = row.get("account_id") or row.get("~id")
                if aid:
                    universe.add(aid)
    return universe


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` from a tp/fp/fn count."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(
    rings: Iterable[FraudRing],
    flagged_accounts: set[str],
    *,
    ring_threshold: float = 1.0,
    account_universe: set[str] | None = None,
) -> dict:
    """Score *flagged_accounts* against ground-truth *rings*.

    Args:
        rings: Ground-truth fraud rings.
        flagged_accounts: Account ids flagged by the detector.
        ring_threshold: Fraction in ``(0, 1]`` of a ring's accounts that must be
            flagged for the ring to count as detected.  ``1.0`` (default)
            requires every involved account to be flagged.
        account_universe: Optional full set of account ids, used to compute true
            negatives.  When *None*, ``true_negatives`` is reported as *None*.

    Returns:
        A metrics dict with ``account`` and ``ring`` sub-dicts (each carrying
        precision/recall/f1 and tp/fp/fn counts) plus a top-level
        ``confusion`` summary and the ``ring_threshold`` used.
    """
    if not 0.0 < ring_threshold <= 1.0:
        raise ValueError(f"ring_threshold must be in (0, 1], got {ring_threshold}")

    rings = list(rings)
    fraud_accounts: set[str] = set()
    for ring in rings:
        fraud_accounts |= ring.accounts

    # Account level.
    acc_tp = len(fraud_accounts & flagged_accounts)
    acc_fp = len(flagged_accounts - fraud_accounts)
    acc_fn = len(fraud_accounts - flagged_accounts)
    acc_p, acc_r, acc_f1 = _prf(acc_tp, acc_fp, acc_fn)

    # Ring level: a ring is detected when >= ring_threshold of its accounts are flagged.
    detected = 0
    for ring in rings:
        if not ring.accounts:
            continue
        hit = len(ring.accounts & flagged_accounts) / len(ring.accounts)
        if hit >= ring_threshold:
            detected += 1
    ring_tp = detected
    ring_fn = len(rings) - detected
    # A flagged account that belongs to no fraud ring is a ring-level false
    # positive at the account-of-interest granularity; we report it as the count
    # of flagged non-fraud accounts so a precision can be derived.
    ring_fp = acc_fp
    ring_p, ring_r, ring_f1 = _prf(ring_tp, ring_fp, ring_fn)

    true_negatives: int | None
    if account_universe is not None:
        legit = account_universe - fraud_accounts
        true_negatives = len(legit - flagged_accounts)
    else:
        true_negatives = None

    return {
        "account": {
            "precision": acc_p,
            "recall": acc_r,
            "f1": acc_f1,
            "true_positives": acc_tp,
            "false_positives": acc_fp,
            "false_negatives": acc_fn,
        },
        "ring": {
            "precision": ring_p,
            "recall": ring_r,
            "f1": ring_f1,
            "detected": ring_tp,
            "missed": ring_fn,
            "total": len(rings),
            "threshold": ring_threshold,
        },
        "confusion": {
            "fraud_accounts": len(fraud_accounts),
            "flagged_accounts": len(flagged_accounts),
            "true_positives": acc_tp,
            "false_positives": acc_fp,
            "false_negatives": acc_fn,
            "true_negatives": true_negatives,
        },
        "ring_threshold": ring_threshold,
    }


def evaluate_dataset(
    data_dir: str,
    flagged_path: str,
    *,
    ring_threshold: float = 1.0,
) -> dict:
    """Evaluate a detector's output against a generated dataset directory.

    Args:
        data_dir: Root output directory (contains ``fraud/`` and optionally
            ``accounts/``).
        flagged_path: File listing detector-flagged account ids (or edges).
        ring_threshold: Fraction of a ring's accounts required for detection.

    Returns:
        The metrics dict from :func:`evaluate`.
    """
    fraud_cases = os.path.join(data_dir, "fraud", "fraud_cases.csv")
    if not os.path.exists(fraud_cases):
        raise FileNotFoundError(
            f"{fraud_cases} not found. Run gen-fraud-graph to produce a dataset first."
        )
    rings = load_fraud_rings(fraud_cases)
    flagged = load_flagged_accounts(flagged_path)
    universe = discover_account_universe(data_dir)
    return evaluate(
        rings,
        flagged,
        ring_threshold=ring_threshold,
        account_universe=universe,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(argv: list[str] | None = None) -> int:
    """Run the ``evaluate`` subcommand.  Returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="gen-fraud-graph evaluate",
        description="Score detector-flagged accounts against generated ground truth.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Generated dataset directory (contains fraud/ and optionally accounts/).",
    )
    parser.add_argument(
        "--flagged",
        type=str,
        required=True,
        help="File of detector-flagged account ids (one per line, or CSV).",
    )
    parser.add_argument(
        "--ring-threshold",
        type=float,
        default=1.0,
        help="Fraction of a ring's accounts that must be flagged for it to count "
        "as detected. Default: 1.0 (all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print metrics as JSON instead of a human-readable summary.",
    )
    args = parser.parse_args(argv)

    metrics = evaluate_dataset(
        args.data,
        args.flagged,
        ring_threshold=args.ring_threshold,
    )

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        _print_summary(metrics)
    return 0


def _print_summary(metrics: dict) -> None:
    """Print a compact human-readable metrics summary."""
    acc = metrics["account"]
    ring = metrics["ring"]
    conf = metrics["confusion"]
    print("=" * 50)
    print("gen_fraud_graph evaluate")
    print("=" * 50)
    print("Account level:")
    print(f"  precision : {acc['precision']:.4f}")
    print(f"  recall    : {acc['recall']:.4f}")
    print(f"  f1        : {acc['f1']:.4f}")
    print(
        f"  tp/fp/fn  : {acc['true_positives']}/{acc['false_positives']}/"
        f"{acc['false_negatives']}"
    )
    print(f"Ring level (threshold {ring['threshold']}):")
    print(f"  precision : {ring['precision']:.4f}")
    print(f"  recall    : {ring['recall']:.4f}")
    print(f"  f1        : {ring['f1']:.4f}")
    print(f"  detected  : {ring['detected']}/{ring['total']}")
    tn = conf["true_negatives"]
    print(
        f"Confusion : tp={conf['true_positives']} fp={conf['false_positives']} "
        f"fn={conf['false_negatives']} tn={tn if tn is not None else 'unknown'}"
    )
    print("=" * 50)


def main() -> None:
    """Standalone entry point: ``python -m gen_fraud_graph.evaluate``."""
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
