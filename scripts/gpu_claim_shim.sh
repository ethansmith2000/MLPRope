#!/bin/bash
# Minimal single-box gpu-claim shim (replaces the lost shared-queue tool).
# Supported surface (what MLPRope launchers actually call):
#   gpu-claim run --owner <o> --job <j> [--gpu any|<i>|<i,j,..>] [--wait] -- <cmd...>
#   gpu-claim status
# Claims are flock-held per-GPU lockfiles in $GPU_CLAIM_LOCK_DIR; the lock is
# held for the lifetime of the launched command. GPUs whose memory.used exceeds
# GPU_CLAIM_MAX_USED_MB (default 4000) are treated as busy even when unlocked,
# so we do not collide with other projects' processes on this shared box.
set -euo pipefail

LOCK_DIR="${GPU_CLAIM_LOCK_DIR:-/workspace/.gpu-claim}"
MAX_USED_MB="${GPU_CLAIM_MAX_USED_MB:-4000}"
POLL_SECONDS="${GPU_CLAIM_POLL_SECONDS:-15}"
mkdir -p "${LOCK_DIR}"

num_gpus() { nvidia-smi -L | wc -l; }

used_mb() { # $1 = gpu index
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

cmd="${1:-}"
case "${cmd}" in
  status)
    n="$(num_gpus)"
    for ((i = 0; i < n; i++)); do
      state="free"
      info=""
      if ! flock -n -x "${LOCK_DIR}/gpu${i}.lock" -c true 2>/dev/null; then
        state="claimed"
        info="$(cat "${LOCK_DIR}/gpu${i}.claim" 2>/dev/null || true)"
      fi
      mem="$(used_mb "${i}")"
      if [[ "${state}" == "free" && "${mem}" -gt "${MAX_USED_MB}" ]]; then
        state="busy-external"
      fi
      echo "gpu${i} ${state} mem_used=${mem}MiB ${info}"
    done
    exit 0
    ;;
  run) shift ;;
  *)
    echo "usage: gpu-claim run --owner <o> --job <j> [--gpu SEL] [--wait] -- <cmd...>" >&2
    echo "       gpu-claim status" >&2
    exit 2
    ;;
esac

OWNER="unknown"
JOB="unnamed"
SELECTOR="any"
WAIT=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner) OWNER="$2"; shift 2 ;;
    --job) JOB="$2"; shift 2 ;;
    --gpu) SELECTOR="$2"; shift 2 ;;
    --wait) WAIT=true; shift ;;
    --) shift; break ;;
    *) echo "gpu-claim: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "gpu-claim: no command given after --" >&2; exit 2; }

n="$(num_gpus)"
if [[ "${SELECTOR}" == "any" ]]; then
  CANDIDATES=($(seq 0 $((n - 1))))
else
  IFS=',' read -r -a CANDIDATES <<<"${SELECTOR}"
fi

while true; do
  for i in "${CANDIDATES[@]}"; do
    exec {fd}>"${LOCK_DIR}/gpu${i}.lock"
    if flock -n -x "${fd}"; then
      if [[ "$(used_mb "${i}")" -gt "${MAX_USED_MB}" ]]; then
        exec {fd}>&-  # locked but externally busy; release and keep looking
        continue
      fi
      echo "owner=${OWNER} job=${JOB} pid=$$ since=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >"${LOCK_DIR}/gpu${i}.claim"
      echo "[gpu-claim] job=${JOB} owner=${OWNER} -> gpu${i}" >&2
      set +e
      CUDA_VISIBLE_DEVICES="${i}" "$@"
      rc=$?
      set -e
      rm -f "${LOCK_DIR}/gpu${i}.claim"
      exit "${rc}"
    fi
    exec {fd}>&-
  done
  if [[ "${WAIT}" != "true" ]]; then
    echo "gpu-claim: no free GPU among [${CANDIDATES[*]}]" >&2
    exit 3
  fi
  sleep "${POLL_SECONDS}"
done
