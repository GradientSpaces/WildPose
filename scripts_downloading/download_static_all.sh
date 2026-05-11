#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/download_seven_scenes_static.sh"
bash "${script_dir}/download_tum_static.sh"
