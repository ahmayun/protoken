# ProToken: Token-Level Attribution for Federated Large Language Models

This repository is the artifact for the MLSys 2026 paper with the same name.

## Setup (fresh environment)

This project uses Flower simulation (`flwr[simulation]`), which depends on **Ray**. Ray is **not available on Python 3.13** in our current dependency set, so you must use **Python 3.12**.

### Using `uv` (recommended)

```bash
./setup.sh
```

### Using `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```