#!/bin/bash
# =============================================================================
# tcs_stitch_full.sh — StitchNodes over the full period.
# Runs after all detect jobs complete successfully.
# Called by submit_tcs_parallel.sh — do not submit directly.
# GLOBAL_START and GLOBAL_END are injected via --export=ALL.
# =============================================================================

#SBATCH --partition=standard
#SBATCH --account=project_462000911
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/tcs_stitch_full_%j.out
#SBATCH --error=logs/tcs_stitch_full_%j.err

TC_DIR=/users/silvcapr/AQUA-diagnostics/frontier-diagnostics/tropical_cyclones

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "StitchNodes over full period ${GLOBAL_START} → ${GLOBAL_END}"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"

cd ${TC_DIR}
srun python -m tropical_cyclones.cli_tropical_cyclones \
    -c config/config_tcs.yaml \
    --startdate ${GLOBAL_START} \
    --enddate ${GLOBAL_END} \
    --stitch-only

echo "Finished: $(date)"
