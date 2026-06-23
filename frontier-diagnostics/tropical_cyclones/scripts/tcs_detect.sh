#!/bin/bash
# =============================================================================
# tcs_detect.sh — DetectNodes for a single year.
# Called by submit_tcs_parallel.sh — do not submit directly.
# START, END, YEAR_TMPDIR are injected via --export=ALL.
# =============================================================================
 
#SBATCH --partition=standard
#SBATCH --account=project_465002727
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
 
TC_DIR=/users/silvcapr/AQUA-diagnostics/frontier-diagnostics/tropical_cyclones
 
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
 
echo "DetectNodes for period ${START} → ${END}"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  Tmpdir  : ${YEAR_TMPDIR}"
 
cd ${TC_DIR}
srun python -m tropical_cyclones.cli_tropical_cyclones \
    -c config/config_tcs.yaml \
    --startdate ${START} \
    --enddate ${END} \
    --override-tmpdir ${YEAR_TMPDIR} \
    --detect-only
 
echo "Finished: $(date)"