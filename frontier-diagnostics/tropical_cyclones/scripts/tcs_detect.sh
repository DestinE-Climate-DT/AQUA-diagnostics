#!/bin/bash
# =============================================================================
# tcs_detect_month.sh — DetectNodes for a single month.
# Called by submit_tcs_parallel.sh — do not submit directly.
# START, END, MONTH_TMPDIR are injected via --export=ALL.
# =============================================================================

#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

TC_DIR=/users/silvcapr/AQUA-diagnostics/frontier-diagnostics/tropical_cyclones

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "DetectNodes for period ${START} → ${END}"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"

cd ${TC_DIR}
srun python -m tropical_cyclones.cli_tropical_cyclones \
    -c config/config_tcs.yaml \
    --startdate ${START} \
    --enddate ${END} \
    --override-tmpdir ${MONTH_TMPDIR} \
    --detect-only

mv ${MONTH_TMPDIR}/IFS-FESOM/historical-1990/tempest_output_*.txt \
   /users/silvcapr/tc_analysis/tmpdir/IFS-FESOM/historical-1990/

echo "Finished: $(date)"
