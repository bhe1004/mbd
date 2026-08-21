#!/usr/bin/env bash
# Run a module in the GPU environment.
#
# The CPU experiments live in the `playground` conda env, whose jaxlib is built
# against a cuDNN newer than the one installed there, so JAX cannot compile for
# the GPU. The `mjx` env holds a matched jax[cuda12] + mujoco 3.9.0 (the version
# the CPU study uses, so the plant is the same model).
#
# The nvidia wheels in that env must precede /usr/local/cuda-12.8 on the library
# path: the system CUDA ships nvJitLink 12.8 and the wheels' cuSPARSE needs the
# 12.9 symbols.
#
#   exper/mjx.sh -m exper.run_gpu --config exp_b
set -euo pipefail

ENV_DIR=/home/home/anaconda3/envs/mjx
SITE="$ENV_DIR/lib/python3.11/site-packages"
NVLIB=$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | tr '\n' ':')

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
export LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:-}"

# The rollout is a scan over the horizon wrapping a scan over the substeps,
# vmapped over the candidates, and XLA takes minutes to build it. Cache the
# result on disk so only the first run of a given shape pays for it.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/exper/out/.jax_cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

exec "$ENV_DIR/bin/python" "$@"
