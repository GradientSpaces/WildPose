#!/bin/bash

mkdir -p datasets/7-scenes
cd datasets/7-scenes

base_url="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

scenes=(
 "chess"
 "fire"
 "heads"
 "office"
 "pumpkin"
 "redkitchen"
 "stairs"
)

for scene in "${scenes[@]}"
do
 echo "Processing scene: $scene"

 if [ ! -d "$scene" ]; then
 zip_file="${scene}.zip"
 wget "${base_url}/${zip_file}"

 if [ $? -eq 0 ]; then
 echo "Successfully downloaded ${zip_file}"
 unzip -q "${zip_file}"
 if [ $? -eq 0 ]; then
 echo "Successfully extracted ${zip_file}"
 rm "${zip_file}"
 else
 echo "Failed to extract ${zip_file}"
 fi
 else
 echo "Failed to download ${zip_file}"
 fi
 else
 echo "Folder $scene already exists, skipping scene download"
 fi

 if [ -d "$scene" ]; then
 for seq_zip in "$scene"/seq-*.zip
 do
 if [ -f "$seq_zip" ]; then
 seq_dir="${seq_zip%.zip}"
 if [ ! -d "$seq_dir" ]; then
 echo "Extracting $(basename "$seq_zip")"
 unzip -q "$seq_zip" -d "$scene"
 fi
 rm "$seq_zip"
 fi
 done
 fi

 echo "Finished processing ${scene}"
 echo "-----------------------------"
done

python - <<'PY'
from pathlib import Path
import numpy as np


def rotmat_to_quat_xyzw(rot):
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        diag = np.diag(rot)
        if diag[0] > diag[1] and diag[0] > diag[2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif diag[1] > diag[2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / np.linalg.norm(quat)


root = Path(".")
for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    seq_dir = scene_dir / "seq-01"
    if not seq_dir.is_dir():
        continue

    rows = []
    for idx, pose_path in enumerate(sorted(seq_dir.glob("*.pose.txt"))):
        pose = np.loadtxt(pose_path)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid pose file: {pose_path}")

        trans = pose[:3, 3]
        quat = rotmat_to_quat_xyzw(pose[:3, :3])
        timestamp = f"{idx:06d}"
        values = " ".join(f"{v:.9f}" for v in (*trans, *quat))
        rows.append(f"{timestamp} {values}")

    if rows:
        out_path = root / f"{scene_dir.name}.txt"
        out_path.write_text("\n".join(rows) + "\n")
        print(f"Wrote {out_path} with {len(rows)} poses")
PY
