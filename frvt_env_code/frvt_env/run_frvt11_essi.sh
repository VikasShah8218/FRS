#!/bin/bash
# ESSI-FR v1 -- NIST FRVT (FRTE) 1:1 local validation, CPU only.
# Runs INSIDE the Ubuntu 24.04.3 container (see start_frvt_tmux.sh).
# Runs the official run_validate_11.sh unchanged and keeps a copy of its output.

cd /home/ubuntu/frvt/11 || exit 1
LOG=/home/ubuntu/frvt_logs/frvt11_essi_fr_v1_$(date +%Y%m%d_%H%M%S)

echo "=== ESSI-FR v1 : FRVT 1:1 validation   started $(date '+%F %T %Z') ==="
echo "=== OS            : $(lsb_release -ds)  (container), CPU only"
echo "=== console log   : $LOG.log"
echo "=== step times    : $LOG.timing.log"
echo
start=$(date +%s)

./run_validate_11.sh 2>&1 \
    | tee "$LOG.log" >(while IFS= read -r l; do printf '%s  %s\n' "$(date +%T)" "$l"; done > "$LOG.timing.log")
rc=${PIPESTATUS[0]}

el=$(( $(date +%s) - start ))
echo
echo "=== finished $(date '+%F %T %Z')   exit code $rc   elapsed $((el / 60)) min $((el % 60)) s ==="
exit $rc
