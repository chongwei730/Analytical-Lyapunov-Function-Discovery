#!/bin/bash
# Anvil port: summarize reproduction results per system+seed using the MOST
# RECENT .out file, pulling the [TEST RESULT] (expression, reward r, success).
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs_diffusion
cd "$ROOT"
printf "%-11s %-5s %-9s %-7s %-9s %s\n" SYSTEM SEED STATE SUCCESS REWARD EXPRESSION
printf '%.0s-' {1..115}; echo
tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  for k in 0 1 2 3 4; do
    out=$(ls -t $RUNS_BASE/$sys/lyad_${sys}_seed${k}_*.out 2>/dev/null | head -1)
    [[ -z "$out" ]] && continue
    line=$(grep -aE "\[TEST RESULT\] \{" "$out" 2>/dev/null | tail -1)
    succ=$(grep -oE "'success': (True|False)" <<<"$line" | head -1 | awk '{print $2}')
    rew=$(grep -oE "'r': [0-9.eE+-]+" <<<"$line" | head -1 | awk '{print $2}')
    expr=$(grep -oE "'expression': '[^']*'" <<<"$line" | head -1 | sed "s/'expression': //")
    if [[ -n "$line" ]]; then state=done
    elif grep -aqE "Traceback|Error|OOM|MemoryError" "$out" 2>/dev/null; then state=ERROR
    else state=running; fi
    printf "%-11s %-5s %-9s %-7s %-9s %s\n" "$sys" "$k" "$state" "${succ:-—}" "${rew:-—}" "${expr:-—}"
  done
done
