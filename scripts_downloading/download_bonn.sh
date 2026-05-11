#!/bin/bash

mkdir -p datasets/Bonn
cd datasets/Bonn

scenes=(
 "balloon"
 "balloon2"
 "crowd"
 "crowd2"
 "person_tracking"
 "person_tracking2"
 "moving_nonobstructing_box"
 "moving_nonobstructing_box2"
)

for scene in "${scenes[@]}"
do
 echo "Processing scene: $scene"
 
 # Check if the folder already exists
 if [ -d "$scene" ]; then
 echo "Folder $scene already exists, skipping download"
 else
 zip_file="rgbd_bonn_${scene}.zip"
 wget "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/${zip_file}"
 
 if [ $? -eq 0 ]; then
 echo "Successfully downloaded ${zip_file}"
 unzip -q "${zip_file}"
 if [ $? -eq 0 ]; then
 echo "Successfully extracted ${zip_file}"
 rm "${zip_file}"
 echo "Removed ${zip_file}"
 else
 echo "Failed to extract ${zip_file}"
 fi
 else
 echo "Failed to download ${zip_file}"
 fi
 fi
 
 echo "Finished processing ${scene}"
 echo "-----------------------------"
done
