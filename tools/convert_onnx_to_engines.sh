#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_DIR="${ONNX_DIR:-"$ROOT_DIR/ori_model"}"
ENGINE_DIR="${ENGINE_DIR:-"$ROOT_DIR/model"}"
PRECISION_FLAGS="${PRECISION_FLAGS---fp16}"
SKIP_INFERENCE="${SKIP_INFERENCE:-1}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec was not found in PATH." >&2
  echo "Install NVIDIA TensorRT on the deployment machine and export its bin directory to PATH." >&2
  exit 127
fi

mkdir -p "$ENGINE_DIR"

build_engine() {
  local onnx_name="$1"
  local engine_name="$2"
  local input_name="$3"
  local input_shape="$4"

  local onnx_path="$ONNX_DIR/$onnx_name"
  local engine_path="$ENGINE_DIR/$engine_name"

  if [[ ! -f "$onnx_path" ]]; then
    echo "ERROR: missing ONNX model: $onnx_path" >&2
    exit 1
  fi

  echo
  echo "==> Building $engine_name from $onnx_name"

  local args=(
    "--onnx=$onnx_path"
    "--saveEngine=$engine_path"
    "--minShapes=${input_name}:${input_shape}"
    "--optShapes=${input_name}:${input_shape}"
    "--maxShapes=${input_name}:${input_shape}"
  )

  if [[ -n "$PRECISION_FLAGS" ]]; then
    # shellcheck disable=SC2206
    local precision_args=( $PRECISION_FLAGS )
    args+=( "${precision_args[@]}" )
  fi

  if [[ "$SKIP_INFERENCE" == "1" ]]; then
    args+=( "--skipInference" )
  fi

  trtexec "${args[@]}"
}

build_engine "end2end_tunnelfaceseg.onnx" "seg_tf.engine" "input" "1x3x512x512"
build_engine "classification_posui.onnx" "cla_posui.engine" "input_1" "1x299x299x3"
build_engine "end2end_waterseg.onnx" "seg_water.engine" "input" "1x3x512x512"
build_engine "end2end_YTLX.onnx" "cla_ytlx.engine" "input" "1x3x224x224"
build_engine "end2end_FHCD.onnx" "cla_fhcd.engine" "input" "1x3x224x224"

echo
echo "Done. Engines were written to: $ENGINE_DIR"
