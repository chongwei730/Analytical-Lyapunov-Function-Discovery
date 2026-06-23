#!/bin/bash
# Create one isolated run directory per system (full copy of the repo with the
# system-specific config.py swapped in). Keeps each of the 40 parallel jobs from
# clobbering the shared single-config.py workflow.
set -euo pipefail
ROOT=/users/9/chen8596/LyaDis
cd "$ROOT"
mkdir -p repro/runs
# Optional args: only (re)build these systems. No args = all. This avoids
# wiping run dirs of jobs that are already running.
WANT=("$@")
want() { [[ ${#WANT[@]} -eq 0 ]] && return 0; for w in "${WANT[@]}"; do [[ "$w" == "$1" ]] && return 0; done; return 1; }
tail -n +2 repro/systems.tsv | while IFS=$'\t' read -r sys src bm dim wt; do
  [[ -z "${sys:-}" || "$sys" == \#* ]] && continue
  want "$sys" || continue
  dst="repro/runs/$sys"
  echo "[setup] $sys  <- $src  (benchmark=$bm, dim=$dim)"
  rm -rf "$dst"; mkdir -p "$dst"
  # copy repo (incl. compiled cyfunc*.so under libs), excluding heavy/irrelevant dirs
  tar cf - --exclude=.git --exclude=repro --exclude=log --exclude=logs --exclude='build' . \
    | ( cd "$dst" && tar xf - )
  # swap in the system-specific config (pendulum already uses the default config.py)
  cp "$src" "$dst/config.py"
  mkdir -p "$dst/log" "$dst/logs"
done
echo "[setup] done. run dirs under repro/runs/"
