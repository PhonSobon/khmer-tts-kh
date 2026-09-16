#!/usr/bin/env bash
# Resumable training loop for Colab's T4.
#
# Colab sessions die (OOM, idle disconnect, 12/24h cap, runtime restart)
# and WIPE local disk. So "resume from local checkpoint" isn't enough —
# on every (re)start this script:
#   1. checks the Hub repo for the most recent checkpoint-* folder
#   2. downloads it into output_dir if found
#   3. launches training with --resume_from_checkpoint pointing at it
#   4. if the training process exits for ANY reason before finishing,
#      waits a few seconds and loops back to step 1
#
# Usage (from a Colab cell, after cloning finetune-hf-vits next to this
# script and after huggingface-cli login):
#   !bash run_training_colab.sh
#
# Stop it by finishing training normally (the script exits 0) or by
# interrupting the Colab cell.

set -uo pipefail

CONFIG_FILE="${CONFIG_FILE:-./training_config.json}"
FINETUNE_REPO_DIR="${FINETUNE_REPO_DIR:-./finetune-hf-vits}"
MAX_RESTARTS="${MAX_RESTARTS:-200}"

# Pull output_dir and hub_model_id out of the JSON config without extra deps.
OUTPUT_DIR=$(python3 -c "import json;print(json.load(open('${CONFIG_FILE}'))['output_dir'])")
HUB_MODEL_ID=$(python3 -c "import json;print(json.load(open('${CONFIG_FILE}'))['hub_model_id'])")

mkdir -p "${OUTPUT_DIR}"

echo "=== output_dir: ${OUTPUT_DIR}"
echo "=== hub_model_id: ${HUB_MODEL_ID}"

fetch_latest_checkpoint () {
    # Prints the local path to the resumed checkpoint, or nothing if
    # none exists yet (first run).
    python3 - "$OUTPUT_DIR" "$HUB_MODEL_ID" <<'PYEOF'
import sys, os, re
from huggingface_hub import HfApi, snapshot_download

output_dir, hub_model_id = sys.argv[1], sys.argv[2]

# 1) Already have a checkpoint locally from this same session? Use it.
local_ckpts = []
if os.path.isdir(output_dir):
    for d in os.listdir(output_dir):
        m = re.match(r"checkpoint-(\d+)$", d)
        if m:
            local_ckpts.append((int(m.group(1)), os.path.join(output_dir, d)))
if local_ckpts:
    local_ckpts.sort()
    print(local_ckpts[-1][1])
    sys.exit(0)

# 2) Otherwise pull the latest checkpoint pushed to the Hub (handles the
#    case where Colab wiped local disk on restart).
try:
    api = HfApi()
    files = api.list_repo_files(hub_model_id)
except Exception:
    sys.exit(0)  # repo doesn't exist yet -> fresh run

ckpt_dirs = sorted(
    {f.split("/")[0] for f in files if re.match(r"checkpoint-\d+/", f)},
    key=lambda d: int(d.split("-")[1]),
)
if not ckpt_dirs:
    sys.exit(0)

latest = ckpt_dirs[-1]
local_path = os.path.join(output_dir, latest)
snapshot_download(
    repo_id=hub_model_id,
    allow_patterns=[f"{latest}/*"],
    local_dir=output_dir,
)
print(local_path)
PYEOF
}

attempt=0
while [ "$attempt" -lt "$MAX_RESTARTS" ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "=== Attempt ${attempt}/${MAX_RESTARTS} — $(date) ==="

    RESUME_PATH=$(fetch_latest_checkpoint)
    RESUME_ARGS=()
    if [ -n "${RESUME_PATH}" ]; then
        echo "=== Resuming from: ${RESUME_PATH}"
        RESUME_ARGS=(--resume_from_checkpoint "${RESUME_PATH}")
    else
        echo "=== No prior checkpoint found — starting fresh"
    fi

    # NOTE: verify the resume flag name for your checked-out version of
    # the repo first with:
    #   python "${FINETUNE_REPO_DIR}/run_vits_finetuning.py" --help | grep -i resum
    # If it differs from --resume_from_checkpoint, edit RESUME_ARGS above.

    accelerate launch "${FINETUNE_REPO_DIR}/run_vits_finetuning.py" \
        "${CONFIG_FILE}" \
        "${RESUME_ARGS[@]}"

    STATUS=$?
    if [ "$STATUS" -eq 0 ]; then
        echo "=== Training script exited cleanly (status 0). Done."
        exit 0
    fi

    echo "=== Training script exited with status ${STATUS}. Restarting in 15s..."
    sleep 15
done

echo "=== Hit MAX_RESTARTS (${MAX_RESTARTS}) without a clean finish. Stopping."
exit 1
