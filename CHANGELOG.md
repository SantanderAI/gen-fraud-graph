# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-07

### Added
- Core 3-phase generation pipeline: accounts → transactions → fraud rings
- `Config` dataclass with scale factor, embedding provider, output format, workers
- `FraudGraphGenerator` orchestrator with parallel `ProcessPoolExecutor` workers
- `EmbeddingGenerator` with three backends: `fake` (random), `local` (SentenceTransformers), `openai`
- `FraudRingGenerator` — cyclic money-laundering patterns with configurable depth (4–7 hops)
- CSV and AWS Neptune bulk-load output formats
- Resume support for interrupted generation (incremental file append)
- ZIP compression option for output files
- `gen-fraud-graph` CLI with `--scale`, `--workers`, `--provider`, `--format` flags
- Python API: `from gen_fraud_graph import Config, FraudGraphGenerator`
- `verify` module to validate fraud patterns against generated transaction edges
- Full test suite covering config, embeddings, exporters, typologies, and end-to-end pipeline
- GitHub Actions workflows (all third-party actions pinned to SHA digests):
  - `ci.yml` — ruff + black + mypy + pytest matrix (3.10/3.11/3.12) with Codecov
  - `codeql.yml` — CodeQL SAST (push, PR, weekly cron)
  - `dep-scan.yml` — `pip-audit` (push, PR, daily cron)
  - `license-check.yml` — dependency-license allowlist + SPDX header verification
  - `pattern-check.yml` — internal-pattern scan with allowlist
  - `cla.yml` — CLA Assistant Lite
  - `stale.yml` — stale issues/PRs automation
  - `release.yml` — PyPI publish via OIDC trusted publishing on GitHub Release
- `.github/dependabot.yml` — weekly Python and GitHub Actions updates
- Issue templates (bug, feature) and PR template
- Apache 2.0 LICENSE + NOTICE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CODEOWNERS

[Unreleased]: https://github.com/SantanderAI/gen-fraud-graph/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SantanderAI/gen-fraud-graph/releases/tag/v0.1.0
