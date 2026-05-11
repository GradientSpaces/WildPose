#!/bin/bash

mkdir -p datasets/TUM_RGBD
cd datasets/TUM_RGBD

sequences=(
 "rgbd_dataset_freiburg1_360"
 "rgbd_dataset_freiburg1_desk"
 "rgbd_dataset_freiburg1_desk2"
 "rgbd_dataset_freiburg1_floor"
 "rgbd_dataset_freiburg1_plant"
 "rgbd_dataset_freiburg1_room"
 "rgbd_dataset_freiburg1_rpy"
 "rgbd_dataset_freiburg1_teddy"
 "rgbd_dataset_freiburg1_xyz"
)

for sequence in "${sequences[@]}"
do
 echo "Processing sequence: $sequence"

 if [ -d "$sequence" ]; then
 echo "Folder $sequence already exists, skipping download"
 else
 archive="${sequence}.tgz"
 wget "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/${archive}"

 if [ $? -eq 0 ]; then
 echo "Successfully downloaded ${archive}"
 tar -xvzf "${archive}"
 if [ $? -eq 0 ]; then
 echo "Successfully extracted ${archive}"
 rm "${archive}"
 echo "Removed ${archive}"
 else
 echo "Failed to extract ${archive}"
 fi
 else
 echo "Failed to download ${archive}"
 fi
 fi

 echo "Finished processing ${sequence}"
 echo "-----------------------------"
done
