#!/usr/bin/env bash
# Start -- or resume -- ESSI-FR training in a detached tmux session.
#
# Run by essi-train.service at every boot, so a stopped-and-started instance
# picks training back up from runs/<experiment>/checkpoints/last.pt on its own
# (the config's `train.resume: auto` finds it). Safe to run by hand too.
#
#   bash scripts/aws/start_training.sh          # start or resume
#   tmux attach -t train                        # watch; Ctrl-B then D to leave
#   touch ~/FRS/NO_AUTORESUME                   # stop boots from auto-starting
#
# Overridable with environment variables: FRS_REPO, FRS_VENV, FRS_CONFIG,
# FRS_SESSION.

set -euo pipefail

REPO="${FRS_REPO:-$HOME/FRS}"
VENV="${FRS_VENV:-$HOME/venv}"
CONFIG="${FRS_CONFIG:-configs/essi_fr_v1_aws_1gpu.yaml}"
SESSION="${FRS_SESSION:-train}"
LOG="$REPO/train_console.log"

if [ -e "$REPO/NO_AUTORESUME" ]; then
    echo "essi-train: $REPO/NO_AUTORESUME exists; not starting training."
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "essi-train: tmux session '$SESSION' already running; nothing to do."
    exit 0
fi

# After a boot the NVIDIA driver can take a little while to come up.
for _ in $(seq 1 36); do
    nvidia-smi >/dev/null 2>&1 && break
    sleep 5
done
if ! nvidia-smi >/dev/null 2>&1; then
    echo "essi-train: GPU not available after 3 minutes; not starting." >&2
    exit 1
fi

cd "$REPO"
echo "===== $(date -u '+%Y-%m-%d %H:%M:%S UTC') starting $CONFIG =====" >> "$LOG"
tmux new-session -d -s "$SESSION" \
    "cd '$REPO' && source '$VENV/bin/activate' && python3 -m scripts.train --config '$CONFIG' 2>&1 | tee -a '$LOG'"
echo "essi-train: started tmux session '$SESSION' ($CONFIG). Attach with: tmux attach -t $SESSION"
