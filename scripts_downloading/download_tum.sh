#!/bin/bash

mkdir -p datasets/TUM_RGBD
cd datasets/TUM_RGBD

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk_with_person.tgz
tar -xvzf rgbd_dataset_freiburg2_desk_with_person.tgz
rm rgbd_dataset_freiburg2_desk_with_person.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_static.tgz
tar -xvzf rgbd_dataset_freiburg3_sitting_static.tgz
rm rgbd_dataset_freiburg3_sitting_static.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz
tar -xvzf rgbd_dataset_freiburg3_sitting_xyz.tgz
rm rgbd_dataset_freiburg3_sitting_xyz.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz
tar -xvzf rgbd_dataset_freiburg3_sitting_halfsphere.tgz
rm rgbd_dataset_freiburg3_sitting_halfsphere.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_rpy.tgz
tar -xvzf rgbd_dataset_freiburg3_sitting_rpy.tgz
rm rgbd_dataset_freiburg3_sitting_rpy.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz
tar -xvzf rgbd_dataset_freiburg3_walking_static.tgz
rm rgbd_dataset_freiburg3_walking_static.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz
tar -xvzf rgbd_dataset_freiburg3_walking_xyz.tgz
rm rgbd_dataset_freiburg3_walking_xyz.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz
tar -xvzf rgbd_dataset_freiburg3_walking_halfsphere.tgz
rm rgbd_dataset_freiburg3_walking_halfsphere.tgz

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz
tar -xvzf rgbd_dataset_freiburg3_walking_rpy.tgz
rm rgbd_dataset_freiburg3_walking_rpy.tgz

