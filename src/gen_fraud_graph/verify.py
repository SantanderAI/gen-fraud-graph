# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Verify that generated fraud patterns actually exist in the transaction data."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict


def verify_fraud_patterns(
    fraud_cases_path: str,
    transactions_dir: str,
) -> bool:
    """Check that every fraud-case cycle is backed by real transaction edges.

    Args:
        fraud_cases_path: Path to ``fraud_cases.csv``.
        transactions_dir: Directory containing ``transactions_fraud.csv``
            (or the fraud subdirectory).

    Returns:
        ``True`` if all patterns are valid, ``False`` otherwise.
    """
    # Build edge set from fraud transactions
    fraud_tx_path = os.path.join(os.path.dirname(fraud_cases_path), "transactions_fraud.csv")
    if not os.path.exists(fraud_tx_path):
        print(f"ERROR: {fraud_tx_path} not found", file=sys.stderr)
        return False

    print("Loading fraud transaction edges...")
    edges: dict[str, set[str]] = defaultdict(set)
    with open(fraud_tx_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src = row.get("src_id") or row.get("~from", "")
            dst = row.get("dst_id") or row.get("~to", "")
            if src and dst:
                edges[src].add(dst)

    print("Verifying fraud cases...")
    all_valid = True
    with open(fraud_cases_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pattern_id = row["pattern_id"]
            accounts = row["involved_accounts"].split("|")
            depth = int(row["depth"])

            # Check that the cycle edges exist
            for k in range(depth):
                src = accounts[k]
                dst = accounts[(k + 1) % depth]
                if dst not in edges.get(src, set()):
                    print(f"  FAIL: {pattern_id} — missing edge {src} -> {dst}")
                    all_valid = False
                    break
            else:
                continue

    if all_valid:
        print("All fraud patterns verified successfully.")
    else:
        print("Some fraud patterns failed verification.", file=sys.stderr)

    return all_valid


def main() -> None:
    """CLI entry point for verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify generated fraud patterns.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root output directory (contains fraud/ subdirectory).",
    )
    args = parser.parse_args()

    cases = os.path.join(args.data_dir, "fraud", "fraud_cases.csv")
    if not os.path.exists(cases):
        print(f"ERROR: {cases} not found. Run gen-fraud-graph first.", file=sys.stderr)
        sys.exit(1)

    ok = verify_fraud_patterns(cases, args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
