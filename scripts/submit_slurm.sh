#!/usr/bin/env bash
# Example only: adapt partition, time, memory, and module setup to the local cluster.
#SBATCH --job-name=dws-eval
#SBATCH --time=5-00:00:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=42
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

METHOD="${1:-HEFT}"
VM_TYPES="${2:-6}"
VMS_PER_TYPE="${3:-4}"
ARRIVAL_RATE="${4:-5.4}"
WF_NUM="${5:-20000}"
RATE_DIST="${6:-}"
DATASET_ARG="${7:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SLURM_ARRAY_TASK_ID:-42}"
DATASET="${DATASET_ARG:-${SEED}}"

case "${METHOD^^}" in
  EST|PEFT|HEFT|GP|GOODRL)
    DRIVER="Step-1-${METHOD^^}.py"
    ;;
  ESRL|ERL|ERL-DWS)
    DRIVER="Step-1-ESRL.py"
    ;;
  *)
    echo "Unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ARGS=(
  --algo_seed "${SEED}"
  --wf_num "${WF_NUM}"
  --vm_types "${VM_TYPES}"
  --each_vm_type_num "${VMS_PER_TYPE}"
  --arr_rate "${ARRIVAL_RATE}"
  --data_name "${DATASET}"
  --use_wandb false
)

if [[ -n "${RATE_DIST}" ]]; then
  ARGS+=(--rate_dist "${RATE_DIST}")
fi

srun "${PYTHON_BIN}" "${DRIVER}" "${ARGS[@]}"
