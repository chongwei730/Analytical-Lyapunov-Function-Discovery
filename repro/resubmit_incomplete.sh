#!/bin/bash
# Resubmit only the seeds that have NOT completed (no [TEST RESULT] in their .out)
# and are not currently RUNNING. Cancels stale PENDING tasks first so they re-run
# with the current run_one.sbatch resources (64g). Usage: ./resubmit_incomplete.sh [sys ...]
set -eo pipefail
ROOT=/users/9/chen8596/LyaDis
cd "$ROOT"
WANT=("$@")
want() { [[ ${#WANT[@]} -eq 0 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }

# 1) cancel my PENDING lya_ array tasks (they'd otherwise run with old resources)
squeue -h -u "$USER" -t PENDING -o "%i %j" | awk '$2 ~ /^lya_/ {print $1}' | xargs -r scancel
sleep 2

# 2) per system, find seeds needing a (re)run
tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  want "$sys" || continue
  running=$(squeue -h -u "$USER" -t RUNNING -n "lya_$sys" -o "%K" 2>/dev/null | tr '\n' ' ')
  need=()
  for k in 0 1 2 3 4; do
    if ls repro/runs/$sys/lya_${sys}_seed${k}_*.out >/dev/null 2>&1 && \
       grep -aqE "\[TEST RESULT\]" repro/runs/$sys/lya_${sys}_seed${k}_*.out 2>/dev/null; then
      continue   # already produced a result
    fi
    [[ " $running " == *" $k "* ]] && continue   # still running
    need+=("$k")
  done
  if [[ ${#need[@]} -gt 0 ]]; then
    idx=$(IFS=,; echo "${need[*]}")
    echo "[resubmit] $sys seeds(array idx)=$idx walltime=$wt"
    sbatch --job-name="lya_$sys" --time="$wt" --array="$idx" \
           --export=ALL,SYS="$sys",BM="$bm" \
           --chdir="$ROOT/repro/runs/$sys" \
           "$ROOT/repro/run_one.sbatch"
  else
    echo "[ok] $sys all seeds done or running"
  fi
done
