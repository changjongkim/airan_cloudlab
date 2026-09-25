#!/usr/bin/env bash

# Source this file from an allocation shell. The deliberately short pipe path
# stays below the Unix-domain socket length limit used by the MPS control daemon.

softwall_mps_configure() {
    require_allocation || return
    export CUDA_MPS_PIPE_DIRECTORY="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/p"
    export CUDA_MPS_LOG_DIRECTORY="$SOFTWALL_ROOT/mps/$SLURM_JOB_ID/l"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}

softwall_mps_control_ready() {
    [[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]] || return 1
    echo get_default_active_thread_percentage | nvidia-cuda-mps-control 2>/dev/null \
        | grep -Eq '^[0-9]+([.][0-9]+)?$'
}

softwall_mps_start() {
    softwall_mps_configure || return
    if softwall_mps_control_ready; then
        echo "MPS control is already running at $CUDA_MPS_PIPE_DIRECTORY"
        return 0
    fi
    if pgrep -u "$USER" -f '^nvidia-cuda-mps-server($| )' >/dev/null; then
        echo "MPS server is still exiting; refusing to start a second control plane" >&2
        return 1
    fi
    # A clean daemon shutdown can leave a stale Unix socket. It must not be
    # mistaken for a live control plane on the next allocation/run.
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
    CUDA_VISIBLE_DEVICES="${SOFTWALL_MPS_GPU:-0}" nvidia-cuda-mps-control -d
    local attempt
    for attempt in {1..100}; do
        softwall_mps_control_ready && return 0
        sleep 0.05
    done
    echo "MPS control socket did not become ready" >&2
    return 1
}

softwall_mps_assert() {
    softwall_mps_control_ready || {
        echo "MPS control daemon is not responding at $CUDA_MPS_PIPE_DIRECTORY/control" >&2
        return 1
    }
}

softwall_mps_stop() {
    if [[ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ]]; then
        if ! timeout --signal=TERM 5s bash -c \
            'echo quit | nvidia-cuda-mps-control'; then
            echo "MPS quit command did not return within 5 seconds" >&2
        fi
    fi
    local attempt server_alive
    for attempt in {1..1200}; do
        server_alive=0
        pgrep -u "$USER" -f '^nvidia-cuda-mps-server($| )' >/dev/null \
            && server_alive=1
        if ! softwall_mps_control_ready && ((server_alive == 0)); then
            break
        fi
        sleep 0.05
    done
    if pgrep -u "$USER" -f '^nvidia-cuda-mps-server($| )' >/dev/null; then
        echo "MPS server did not exit within 60 seconds" >&2
        return 1
    fi
    if ! softwall_mps_control_ready; then
        rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
        mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
    fi
}
