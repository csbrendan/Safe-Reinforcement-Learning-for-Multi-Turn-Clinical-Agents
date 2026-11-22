#!/usr/bin/env bash
# Runs ON the Azure GPU VM (via `vm.sh setup`). Idempotent: safe to re-run.
# Builds the SkyRL env, downloads MedGemma 27B, and runs the CPU tests plus a GPU check. Trains nothing.
set -euo pipefail
SKYRL_COMMIT="${SKYRL_COMMIT:-6279604}"   # SkyRL main as read on 2026-11-19 (skyrl_env.py was written against it)
MODEL="${MODEL:-google/medgemma-27b-text-it}"
REPO="$HOME/rl-gains-in-clinical-tasks"
SKY="$HOME/SkyRL"

echo "== GPU and driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if (( drv < 580 )); then
  echo "Driver $drv is too old for SkyRL's torch 2.14 / CUDA 13 wheels (need >= 580). Stop and update the driver." >&2
  exit 1
fi

echo "== uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "== SkyRL at $SKYRL_COMMIT"
if [[ ! -d "$SKY/.git" ]]; then git clone -q https://github.com/NovaSky-AI/SkyRL.git "$SKY"; fi
git -C "$SKY" fetch -q origin && git -C "$SKY" checkout -q "$SKYRL_COMMIT"
(cd "$SKY" && uv sync --extra fsdp)
PY="$SKY/.venv/bin/python"
uv pip install --python "$PY" -q pytest

echo "== repo and SD-Bench clone"
test -d "$HOME/v2_bench/sdbench/sdbench" || { echo "SD-Bench clone missing: run vm.sh push first" >&2; exit 1; }
test -s "$HOME/.secrets/azure.env" || { echo "~/.secrets/azure.env missing: run vm.sh push first" >&2; exit 1; }

echo "== model weights ($MODEL, ~54 GB)"
"$PY" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" \
  "$MODEL" "$HOME/models/$(basename "$MODEL")" >/dev/null
echo "downloaded to $HOME/models/$(basename "$MODEL")"

echo "== CPU tests (report only: a few may need packages that only the Spark TRL env has)"
(cd "$REPO" && PYTHONPATH="$REPO" "$PY" -m pytest -q -p no:cacheprovider tests/test_v2_sdbench*.py | tail -5) || \
  echo "WARNING: some CPU tests failed in the SkyRL env; read the output above before the pilot"

echo "== GPU check"
cd "$REPO" && PYTHONPATH="$REPO" "$PY" - <<'EOF'
import torch, vllm, skyrl_gym
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
print("vllm", vllm.__version__)
from tasks.v2_sdbench.skyrl_env import SDBenchEnv  # real skyrl_gym base class here
print("SDBenchEnv base:", SDBenchEnv.__mro__[1].__module__)
EOF
echo "Setup done. Next (needs Brendan's go-ahead): the pilot, V2/skyrl/run_pilot_27b.sh with MODEL=$HOME/models/$(basename "$MODEL")"
