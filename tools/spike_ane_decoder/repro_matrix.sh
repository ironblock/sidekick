#!/bin/bash
# repro_matrix.sh <mlmodelc> <units> <C> <D> <Dh>: which object's lifetime decides the
# crash of a stateful multifunction program on Core ML's CPU path?
# 3 sequences x (prefill + 60 decode steps), each variant run 3 times; exit 139 = segfault.
# Build first: swiftc -O -o "$SPIKE_WORK/statecheck" tools/spike_ane_decoder/statecheck.swift
WORK="${SPIKE_WORK:-.dev-data/spike-ane-decoder}"
export CFFIXED_USER_HOME="$WORK/home"
for v in "" "KEEP_STATE=1" "KEEP_OUT=1" "KEEP_IN=1" "KEEP_STATE=1 KEEP_OUT=1 KEEP_IN=1" "PREFILL=0" "STATE_FROM=prefill"; do
  r=""
  for i in 1 2 3; do
    env SEQS=3 $v "$WORK/statecheck" "$1" "$2" 60 "$3" "$4" "$5" > /dev/null 2>&1
    r="$r $?"
  done
  echo "[${v:-none}] exits:$r"
done
