#!/bin/bash
# =============================================================================
# submit_tcs_parallel.sh — Submit one detect job per year in parallel (1990–2014).
# Usage: ./submit_tcs_parallel.sh
# =============================================================================
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
mkdir -p logs
 
for YEAR in 1990; do
 
    export START="${YEAR}0101"
    export END="${YEAR}-12-31T23:00:00"
    export YEAR_TMPDIR=/scratch/project_465002727/scaprioli/TC_analysis/tmpdir/${YEAR}
 
    mkdir -p ${YEAR_TMPDIR}
 
    JOB_ID=$(sbatch --parsable \
        --job-name="tcs_detect_${YEAR}" \
        --account=project_465002727 \
        --partition=standard \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --time=48:00:00 \
        --output="logs/tcs_detect_${YEAR}_%j.out" \
        --error="logs/tcs_detect_${YEAR}_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/tcs_detect.sh")
 
    echo "Submitted detect job for ${YEAR} (${START} → ${END}) : job ID ${JOB_ID}"
 
done
 
echo ""
echo "All years submitted. Monitor with:"
echo "  squeue -u \$USER --format='%.10i %.20j %.8T %.10M %.9l' | grep tcs_detect"
 