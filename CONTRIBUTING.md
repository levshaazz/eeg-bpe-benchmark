# Contributing to EEG-BPE

Thank you for your interest in contributing to this project.

## How to contribute

1. **Fork** the repository and create a feature branch from `main`.
2. **Install** the development environment:
   ```bash
   pip install -e ".[all]"
   ```
3. **Run** the quick smoke test to verify your setup:
   ```bash
   python -m eeg_bpe.run_all --quick
   ```
4. **Make** your changes — keep commits focused and descriptive.
5. **Submit** a pull request with a clear description of what you changed and why.

## Reporting issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if applicable)
- Expected vs. actual behaviour
- Your Python version, OS, and GPU (if relevant)

## Code style

- Follow existing code conventions in the repository.
- Use type hints where practical.
- Keep imports inside functions for heavy dependencies (torch, pandas) to avoid slow startup.

## Adding experiments

New experiments should follow the pattern in `eeg_bpe/exp*.py`:
- Accept `datasets`, `vocab_sizes`, `seeds` parameters
- Save results to `results/logs/` as CSV
- Add a runner entry in `eeg_bpe/run_all.py`
- Include resume logic (skip already-computed rows)

## Adding datasets

- Add a download script in `datasets/`
- Add a loader function in `eeg_bpe/data_loading.py`
- Register the dataset in `eeg_bpe/config.py`
