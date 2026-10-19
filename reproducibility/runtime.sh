export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SCEPI_CONDA_ENV="${SCEPI_CONDA_ENV:?Set SCEPI_CONDA_ENV to an absolute system Conda environment path}"
case "$SCEPI_CONDA_ENV" in
  /*) ;;
  *) echo "SCEPI_CONDA_ENV must be absolute" >&2; return 1 ;;
esac
case "$(realpath "$SCEPI_CONDA_ENV")" in
  "$PROJECT_ROOT"|"$PROJECT_ROOT"/*) echo "Keep the environment outside the project" >&2; return 1 ;;
esac
test -x "$SCEPI_CONDA_ENV/bin/python" || return 1
export CONDA_PREFIX="$SCEPI_CONDA_ENV"
export PATH="$SCEPI_CONDA_ENV/bin:$PATH"
export TMPDIR="$PROJECT_ROOT/work/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache"
export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export KERAS_HOME="$PROJECT_ROOT/cache/keras"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TF_FORCE_GPU_ALLOW_GROWTH=true
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$PROJECT_ROOT/cache" "$PROJECT_ROOT/runs"
