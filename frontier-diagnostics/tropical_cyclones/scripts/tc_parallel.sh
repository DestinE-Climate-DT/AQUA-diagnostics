#!/bin/bash
# =============================================================================
# submit_tcs_parallel.sh — Submit one detect job per month in parallel,
# then chain a single stitch job that runs after all of them complete.
# Usage: ./submit_tcs_parallel.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MONTHS=(
    "19900501 19900601"
    "19900601 19900701"
    "19900701 19900801"
    "19900801 19900901"
)

# calcola automaticamente start e end dell'intero range
export GLOBAL_START=$(echo ${MONTHS[0]} | awk '{print $1}')
export GLOBAL_END=$(echo ${MONTHS[-1]} | awk '{print $2}')

mkdir -p logs

# --- submit one detect job per month -----------------------------------------
DETECT_IDS=()

for month in "${MONTHS[@]}"; do
    export START=$(echo $month | awk '{print $1}')
    export END=$(echo $month | awk '{print $2}')
    export MONTH_TMPDIR=/users/silvcapr/tc_analysis/tmpdir/${START}

    JOB_ID=$(sbatch --parsable \
        --job-name="tcs_detect_${START}" \
        --account=project_462000911 \
        --partition=standard \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=128 \
        --time=08:00:00 \
        --output="logs/tcs_detect_${START}_%j.out" \
        --error="logs/tcs_detect_${START}_%j.err" \
        --export=ALL \
        "${SCRIPT_DIR}/tcs_detect.sh")

    echo "Submitted detect job for ${START} → ${END} : job ID ${JOB_ID}"
    DETECT_IDS+=($JOB_ID)
done

# --- build afterok dependency string -----------------------------------------
DEPENDENCY=$(IFS=:; echo "afterok:${DETECT_IDS[*]}")

# --- submit stitch job that waits for all detect jobs ------------------------
STITCH_ID=$(sbatch --parsable \
    --job-name="tcs_stitch_full" \
    --account=project_462000911 \
    --partition=standard \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=128 \
    --time=04:00:00 \
    --output="logs/tcs_stitch_%j.out" \
    --error="logs/tcs_stitch_%j.err" \
    --dependency=${DEPENDENCY} \
    --export=ALL \
    "${SCRIPT_DIR}/tcs_stitch.sh")

echo ""
echo "Submitted stitch job : ${STITCH_ID} (depends on: ${DETECT_IDS[*]})"
echo "Full period          : ${GLOBAL_START} → ${GLOBAL_END}"
echo ""
echo "Monitor with:"
echo "  squeue -j $(IFS=,; echo "${DETECT_IDS[*]},${STITCH_ID}")"
