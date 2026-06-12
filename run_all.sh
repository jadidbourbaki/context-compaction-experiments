#!/usr/bin/env bash
# Run both arms (claude, openai) on the URL dataset, then plot.
#
# N=15000 (~325K tokens) fits OpenAI's per-request TPM (400K on our org)
# and clears Claude's 50K-token compaction trigger. One seed suffices.
#
# Overrides: N=10000 SEEDS="42 43" NUM_QUERIES=100 ./run_all.sh

set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && . ./.env

# Tee all output to a committed log for reproducibility.
mkdir -p results
exec > >(tee results/run.log) 2>&1
echo "# run started $(date -u +%Y-%m-%dT%H:%M:%SZ)"

N="${N:-15000}"
SEEDS="${SEEDS:-42}"
NUM_QUERIES="${NUM_QUERIES:-200}"

if [ ! -f data/malicious_phish.csv ]; then
    echo "data/malicious_phish.csv missing; see data/README.md" >&2
    exit 1
fi
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY}"

for seed in $SEEDS; do
    for arm in claude openai; do
        echo "=== arm=$arm n=$N seed=$seed ==="
        uv run experiment.py run \
            --arm "$arm" \
            --n "$N" \
            --num-queries "$NUM_QUERIES" \
            --seed "$seed"
    done
done

uv run experiment.py plot
