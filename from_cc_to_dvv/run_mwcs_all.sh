#!/usr/bin/env bash
set -euo pipefail

# =========================
# Check arguments
# =========================
if [ $# -ne 2 ]; then
    echo "Usage: $0 <project> <filterid_mwcs>"
    echo "Example: $0 M5eq_1 1"
    exit 1
fi

# =========================
# Parameters
# =========================
node_rank=0
num_nodes=1

project="$1"
filterid_mwcs="$2"

components=(EE EN EZ NE NN NZ ZE ZN ZZ)

# Validate filter ID
if ! [[ "$filterid_mwcs" =~ ^[0-9]+$ ]]; then
    echo "ERROR: filterid_mwcs must be a non-negative integer."
    exit 1
fi

# =========================
# Run MWCS
# =========================
for component in "${components[@]}"; do

    echo "========================================"
    echo "Starting component: $component"
    echo "Project: $project"
    echo "Filter ID: $filterid_mwcs"
    echo "Time: $(date)"
    echo "========================================"

    python run_mwcs_zarr.py \
        --node_rank "$node_rank" \
        --num_nodes "$num_nodes" \
        --project "$project" \
        --filterid_mwcs "$filterid_mwcs" \
        --component "$component"

    echo "Finished component: $component"
    echo "Time: $(date)"

done

echo "========================================"
echo "All components finished successfully!"
echo "Project: $project"
echo "Filter ID: $filterid_mwcs"
echo "Time: $(date)"
echo "========================================"
