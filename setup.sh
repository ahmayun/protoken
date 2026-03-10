curl -LsSf https://astral.sh/uv/install.sh | sh

# Flower simulation depends on Ray, which currently does not support Python 3.13.
# Force a Python 3.12 virtualenv so `flwr[simulation]` can install Ray.
$HOME/.local/bin/uv python install 3.12
$HOME/.local/bin/uv venv --python 3.12
$HOME/.local/bin/uv sync

