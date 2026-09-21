#!/bin/bash
# Start the ESSI-FR v1 FRVT 1:1 validation in a tmux session named "frvt".
#   watch:  tmux attach -t frvt        leave running: Ctrl-B then D
#
# The run happens in the Ubuntu 24.04.3 container built from
# /home/ubuntu/frvt_env/Dockerfile (image essi-frvt11:24.04.3), as user
# ubuntu, so every file it writes stays owned by ubuntu. No GPU is passed in.

IMAGE=essi-frvt11:24.04.3
ENV=/home/ubuntu/frvt_env

if tmux has-session -t frvt 2>/dev/null; then
    echo "tmux session 'frvt' already exists; attach with: tmux attach -t frvt"
    exit 1
fi
mkdir -p /home/ubuntu/frvt_logs

tmux new-session -d -s frvt -x 200 -y 50 "docker run --rm -it --name frvt11 \
    --user 1000:1000 -e HOME=/tmp -e TERM=xterm-256color \
    -v /home/ubuntu/frvt:/home/ubuntu/frvt \
    -v /home/ubuntu/frvt_logs:/home/ubuntu/frvt_logs \
    -v $ENV:$ENV:ro \
    $IMAGE bash $ENV/run_frvt11_essi.sh; \
    echo; echo '=== container exited. This window stays open: Ctrl-B then D to detach ==='; exec bash"
echo "started tmux session 'frvt' -- attach with: tmux attach -t frvt"
