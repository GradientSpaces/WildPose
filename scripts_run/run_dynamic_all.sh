#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

wild_slam_mocap_configs=(
  "configs/Dynamic/Wild_SLAM_Mocap/ANYmal1.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/ANYmal2.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/ball.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/crowd.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/person_tracking.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/racket.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/stones.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/table_tracking1.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/table_tracking2.yaml"
  "configs/Dynamic/Wild_SLAM_Mocap/umbrella.yaml"
)

bonn_configs=(
  "configs/Dynamic/Bonn/bonn_balloon.yaml"
  "configs/Dynamic/Bonn/bonn_balloon2.yaml"
  "configs/Dynamic/Bonn/bonn_crowd.yaml"
  "configs/Dynamic/Bonn/bonn_crowd2.yaml"
  "configs/Dynamic/Bonn/bonn_moving_nonobstructing_box.yaml"
  "configs/Dynamic/Bonn/bonn_moving_nonobstructing_box2.yaml"
  "configs/Dynamic/Bonn/bonn_person_tracking.yaml"
  "configs/Dynamic/Bonn/bonn_person_tracking2.yaml"
)

tum_rgbd_configs=(
  "configs/Dynamic/TUM_RGBD/freiburg2_desk_with_person.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_sitting_halfsphere_static.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_sitting_halfsphere.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_sitting_rpy.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_sitting_xyz.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere_static.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_walking_halfsphere.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml"
  "configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml"
)

usage() {
  echo "Usage: $0 [wild_slam_mocap] [bonn] [tum_rgbd]"
  echo
  echo "If no groups are provided, all dynamic groups are run."
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
  groups=("wild_slam_mocap" "bonn" "tum_rgbd")
fi

for group in "${groups[@]}"
do
  case "${group}" in
    wild_slam_mocap)
      run_configs "${wild_slam_mocap_configs[@]}"
      ;;
    bonn)
      run_configs "${bonn_configs[@]}"
      ;;
    tum_rgbd)
      run_configs "${tum_rgbd_configs[@]}"
      ;;
    *)
      echo "Unknown dynamic group: ${group}" >&2
      usage >&2
      exit 1
      ;;
  esac
done
