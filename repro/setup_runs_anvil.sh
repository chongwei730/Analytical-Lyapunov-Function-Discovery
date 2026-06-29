#!/bin/bash
# Anvil port: create one isolated run directory per system (full copy of the
# repo with the system-specific config.py swapped in).
set -euo pipefail
ROOT=/home/x-cchen47/Analytical-Lyapunov-Function-Discovery
RUNS_BASE=/anvil/scratch/x-cchen47/LyaDis/runs
cd "$ROOT"
mkdir -p "$RUNS_BASE"
WANT=("$@")
want() { [[ ${#WANT[@]} -eq 0 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }
tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  want "$sys" || continue
  dst="$RUNS_BASE/$sys"
  echo "[setup] $sys  <- $src  (benchmark=$bm, dim=$dim)"
  rm -rf "$dst"; mkdir -p "$dst"
  # copy repo (incl. compiled cyfunc*.so under libs), excluding heavy/irrelevant dirs
  tar cf - --exclude=.git --exclude=repro --exclude=log --exclude=logs --exclude='build' . \
    | ( cd "$dst" && tar xf - )
  # swap in the system-specific config (pendulum already uses the default config.py)
  cp "$src" "$dst/config.py"
  mkdir -p "$dst/log" "$dst/logs"
done
echo "[setup] done. run dirs under $RUNS_BASE/"
