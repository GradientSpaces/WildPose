#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/download_wild_slam_mocap_scene1.sh"
bash "${script_dir}/download_wild_slam_mocap_scene2.sh"
bash "${script_dir}/download_bonn.sh"
bash "${script_dir}/download_tum.sh"
