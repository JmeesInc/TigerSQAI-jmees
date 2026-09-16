#!/usr/bin/env bash
for p in $(pgrep -f 'ens_greedy_fast.py --greedy'); do kill "$p" 2>/dev/null; done
sleep 2; echo "残 $(pgrep -fc 'ens_greedy_fast.py --greedy' || true)"
