#!/bin/bash
#SBATCH --job-name=env_smoke
#SBATCH --account=lusu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gautschi/ou57/logs/%x-%j.out

# Zero-model check that a compute node can reach the registry and build/write a Singularity sandbox.
# Usage: mkdir -p /scratch/gautschi/ou57/logs && sbatch experiments/slurm/env_smoke.sbatch
set -u
cd /scratch/gautschi/ou57/mini-swe-agent
source .venv/bin/activate
export TMPDIR=/scratch/gautschi/ou57/tmp APPTAINER_CACHEDIR=/scratch/gautschi/ou57/.apptainer_cache
mkdir -p "$TMPDIR"

echo "host: $(hostname)  arch: $(uname -m)  start: $(date)"
echo "registry reachable:"; curl -sS -m 20 -o /dev/null -w "%{http_code}\n" https://registry-1.docker.io/v2/ || echo "NO NETWORK"

python - <<'EOF'
import time
from minisweagent.environments.singularity import SingularityEnvironment

t = time.time()
env = SingularityEnvironment(
    image="docker://docker.io/swebench/sweb.eval.x86_64.sqlfluff_1776_sqlfluff-1517:latest", cwd="/testbed", timeout=120
)
print(f"sandbox built in {time.time() - t:.0f}s")
print(env.execute({"command": "git status | head -3; touch x_probe && echo writable-ok"}))
env.cleanup()
EOF
echo "end: $(date)"
