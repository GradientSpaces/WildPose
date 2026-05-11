#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

seven_scenes_configs=(
  "configs/Static/seven_scenes/chess.yaml"
  "configs/Static/seven_scenes/fire.yaml"
  "configs/Static/seven_scenes/heads.yaml"
  "configs/Static/seven_scenes/office.yaml"
  "configs/Static/seven_scenes/pumpkin.yaml"
  "configs/Static/seven_scenes/redkitchen.yaml"
  "configs/Static/seven_scenes/stairs.yaml"
)

tum_ablation_configs=(
  "configs/Static/TUM_ablation/fr1_360.yaml"
  "configs/Static/TUM_ablation/fr1_desk.yaml"
  "configs/Static/TUM_ablation/fr1_desk2.yaml"
  "configs/Static/TUM_ablation/fr1_floor.yaml"
  "configs/Static/TUM_ablation/fr1_plant.yaml"
  "configs/Static/TUM_ablation/fr1_room.yaml"
  "configs/Static/TUM_ablation/fr1_rpy.yaml"
  "configs/Static/TUM_ablation/fr1_teddy.yaml"
  "configs/Static/TUM_ablation/fr1_xyz.yaml"
)

usage() {
  echo "Usage: $0 [seven_scenes] [tum_ablation]"
  echo
  echo "If no groups are provided, all static groups are run."
}

run_config() {
  local config="$1"
  echo "==> Running ${config}"
  python run.py "${config}"
}

run_configs() {
  local configs=("$@")
  local config
  for config in "${configs[@]}"
  do
    run_config "${config}"
  done
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

groups=("$@")
if [ "${#groups[@]}" -eq 0 ]; then
  groups=("seven_scenes" "tum_ablation")
fi

for group in "${groups[@]}"
do
  case "${group}" in
    seven_scenes)
      run_configs "${seven_scenes_configs[@]}"
      ;;
    tum_ablation)
      run_configs "${tum_ablation_configs[@]}"
      ;;
    *)
      echo "Unknown static group: ${group}" >&2
      usage >&2
      exit 1
      ;;
  esac
done