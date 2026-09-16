#!/usr/bin/env bash
for p in $(pgrep -f 'run_qdense.sh'); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f 'ens_qdense_mgpu'); do kill "$p" 2>/dev/null; done
echo "stopped"
