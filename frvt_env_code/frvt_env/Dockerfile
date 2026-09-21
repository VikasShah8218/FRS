# Ubuntu 24.04.3 LTS -- the OS that NIST's FRVT 1:1 validation script requires
# (common/scripts/utils.sh: reqOS). The host runs a newer Ubuntu, so the
# official, unmodified script runs inside this container instead.
FROM ubuntu:noble-20260113

# base-files owns /etc/lsb-release: hold it so the image stays 24.04.3.
# Tools: compiler + cmake for the build, bc/gawk/sed/grep for the script's
# checks, lsb-release for the OS check, procps (top) for count_threads.sh,
# ncurses-bin (tput) for the script's bold text.
RUN apt-mark hold base-files \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ make cmake bc gawk sed grep lsb-release procps ncurses-bin \
 && rm -rf /var/lib/apt/lists/*
