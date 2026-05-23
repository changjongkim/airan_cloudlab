#!/usr/bin/env bash
# nvidia-smi dmon synchronous logger.
#
# Usage:
#   ./dmon_sync.sh start <out_dir>    # spawn dmon, save PID
#   ./dmon_sync.sh stop  <out_dir>    # kill dmon
#   ./dmon_sync.sh mark  <out_dir> <label>   # write timestamped marker
#
# Output files in <out_dir>:
#   dmon.csv     — nvidia-smi dmon -s mu -d 1 -o T output
#   dmon.pid     — bg PID
#   markers.txt  — manual sync markers (timestamp + label)
#
# For bimodal analysis: start dmon → run L1 + AI → mark before/after each
# measurement → stop dmon → join CSVs by timestamp in post-processing.

set -uo pipefail
cmd="${1:-}"
out_dir="${2:-./dmon_out}"

mkdir -p "$out_dir"

case "$cmd" in
  start)
    if [[ -f "$out_dir/dmon.pid" ]] && kill -0 "$(cat $out_dir/dmon.pid)" 2>/dev/null; then
      echo "dmon already running (pid $(cat $out_dir/dmon.pid))"
      exit 0
    fi
    # -s mu : memory + utilization
    # -d 1  : 1 second interval
    # -o DT : show date+time prefix
    nvidia-smi dmon -s mu -d 1 -o DT > "$out_dir/dmon.csv" 2>&1 &
    echo $! > "$out_dir/dmon.pid"
    echo "$(date -Iseconds) STARTED" >> "$out_dir/markers.txt"
    echo "dmon started, pid $(cat $out_dir/dmon.pid), out $out_dir/dmon.csv"
    ;;

  stop)
    if [[ ! -f "$out_dir/dmon.pid" ]]; then
      echo "no dmon.pid found in $out_dir"; exit 1
    fi
    pid=$(cat "$out_dir/dmon.pid")
    kill "$pid" 2>/dev/null && echo "dmon stopped (pid $pid)"
    rm -f "$out_dir/dmon.pid"
    echo "$(date -Iseconds) STOPPED" >> "$out_dir/markers.txt"
    wc -l "$out_dir/dmon.csv"
    ;;

  mark)
    label="${3:-unlabeled}"
    echo "$(date -Iseconds) $label" >> "$out_dir/markers.txt"
    echo "marked: $label"
    ;;

  *)
    echo "Usage: $0 {start|stop|mark} <out_dir> [<label>]"
    exit 1
    ;;
esac
