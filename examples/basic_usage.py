# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Basic usage example for gen_fraud_graph."""

from gen_fraud_graph import Config, FraudGraphGenerator


def main():
    # Small-scale generation with fake embeddings (fast, no extra deps)
    config = Config(
        scale_factor=0.001,  # ~10K accounts, ~90K transactions
        num_fraud_rings=50,  # 50 fraud ring patterns
        embedding_provider="fake",  # random vectors (no model needed)
        workers=2,  # 2 parallel processes
        output_dir="./example_output",
    )

    generator = FraudGraphGenerator(config)
    generator.run()

    # Output structure:
    #   example_output/
    #   ├── accounts/
    #   │   └── accounts_0_0.csv       (~10K rows)
    #   ├── transactions/
    #   │   └── transactions_0_0.csv   (~90K rows)
    #   └── fraud/
    #       ├── transactions_fraud.csv  (fraud edges)
    #       └── fraud_cases.csv         (fraud ring metadata)


if __name__ == "__main__":
    main()
