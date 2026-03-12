# ProToken: Token-Level Attribution for Federated Large Language Models

This repository is the artifact for the [MLSys 2026 paper](https://arxiv.org/pdf/2601.19672) with the same name.

## 1: Pre-requisites

### 1A: Software Requirements

- A linux distribution (preferably Ubuntu)
- `git`, `uv`

### 1B: Hardware Requirements

- 1 x NVidia A100 GPU (or better)
- 512G RAM (or better)

## 2: Setup

### 2A: Install `uv`

You may skip this step if you already have `uv` installed.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> Alternatively you may follow [instructions on the Astral website](https://docs.astral.sh/uv/getting-started/installation/)

### 2B: Installing Python libraries

### Using `uv` (recommended)

> NOTE: The following assumes that uv has been installed at `$HOME/.local/bin/uv`, which is the default location. If it has been installed elsewhere, please use the correct path.

```bash
$HOME/.local/bin/uv python install 3.12
$HOME/.local/bin/uv venv --python 3.12
$HOME/.local/bin/uv sync
```

## 3: Reproducing Results

The `reproduce.sh` script wraps the codebase to reproduce the paper's figures. The paper evaluates ProToken across 16 configurations (Model/Dataset combinations) with 6 clients over 10 rounds. Since running all of them would be prohibitively time-consuming, the commands below cover a single representative configuration i.e. `SmolLM` on the `coding` dataset for 10 rounds on 6 clients.

The expected time to run all the commands in this section is 2-3 hours.

### RQ1: How accurately does ProToken attribute token-level provenance? (Fig 2 & 3)

```bash
PATH=$PATH:~/.local/bin ./reproduce.sh --model smollm --dataset coding --rq1
```

**Reference output (main accuracy and client contribution distributions):**

RQ1 (a) — Main accuracy

![See reference-outputs/rq1-a.png](reference-outputs/rq1-a.png)

*Important features:* Overall the ProToken attribution accuracy line should be higher than the baselines.

RQ1 (b) — Client contribution distributions

![See reference-outputs/rq1-b.png](reference-outputs/rq1-b.png)

*Important features:* The mean contribution accuracy of the box plots for responsible clients should be higher (closer to 1) than Non-responsible clients.

---

### RQ2: How does gradient weighting (relevance filtering) affect attribution? (Fig 4)

```bash
PATH=$PATH:~/.local/bin ./reproduce.sh --model smollm --dataset coding --rq2
```

![See reference-outputs/rq2.png](reference-outputs/rq2.png)

RQ2 — Gradient enable/disable

*Important features:* The "gradients enabled" bar should be higher than the "gradients disabled" bar.

---

### RQ3: What is the computational overhead vs. layer count? (Fig 5 — Tractability)

```bash
PATH=$PATH:~/.local/bin ./reproduce.sh --model smollm --dataset coding --rq3
```

![See reference-outputs/rq3.png](reference-outputs/rq3.png)

RQ3 — Computational overhead

*Important features:* The average provenance time should increase as number of layers increases.

---

### RQ4: How does ProToken scale with more clients? (Fig 6 & 7)

```bash
PATH=$PATH:~/.local/bin ./reproduce.sh --model smollm --dataset coding --rq4
```

RQ4 (a) — Scalability

![See reference-outputs/rq4-a.png](reference-outputs/rq4-a.png)

*Important features:* Same as for RQ1 (a)

RQ4 (b) — Scalability

![See reference-outputs/rq4-b.png](reference-outputs/rq4-b.png)

*Important features:*  Same as for RQ1 (b)