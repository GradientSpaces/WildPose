# WildPose: A Unified Framework for Robust Pose Estimation in the Wild

[Jianhao Zheng](https://jianhao-zheng.github.io/), [Liyuan Zhu](https://www.zhuliyuan.net/), [Zihan Zhu](https://zzh2000.github.io), [Iro Armeni](https://ir0.github.io/)

Computer Vision and Pattern Recognition (CVPR) 2026

Paper Coming Soon | [Video](https://youtu.be/j-1I1y2fxgU) | [Project Page](https://wildpose.github.io/)

![WildPose teaser](media/teaser.png)

WildPose is a unified monocular camera pose-estimation framework for in-the-wild videos, including dynamic scenes, static scenes, and low-ego-motion sequences. It combines 3D-aware features from a frozen MASt3R backbone with differentiable dense bundle adjustment and learned motion masks for robust trajectory estimation.

## 📚 Table of Contents

1. [📝 TODO](#-todo)
2. [🛠 Installation](#-installation)
3. [📦 Checkpoints](#-checkpoints)
4. [🚀 Run WildPose](#-run-wildpose)
5. [🙏 Acknowledgements](#-acknowledgements)
6. [📚 Citation](#-citation)
7. [✉ Contact](#-contact)

## 📝 TODO

- Add run and evaluation scripts for the Sintel dataset.
- Add scripts for customized videos.
- Release training code.

## 🛠 Installation

The following setup follows the working notes in `log of installation.md`. The tested environment uses Python 3.10, PyTorch 2.5.1, CUDA 12.4 wheels, MASt3R, lietorch, and the local CUDA extension in `setup.py`.

1. Clone the repository and initialize submodules.

```bash
git clone --recursive https://github.com/GradientSpaces/WildPose.git
cd WildPose
git submodule update --init --recursive
```

2. Create and activate the conda environment.

```bash
conda create -n wildpose python=3.10 ninja mkl mkl-include -c conda-forge -y
conda activate wildpose
```

3. Install PyTorch and torch-scatter.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

4. Install local dependencies.

```bash
# Optional: set this to match your GPU architectures before compiling CUDA extensions.
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"

pip install --no-build-isolation thirdparty/lietorch
pip install --no-build-isolation -e thirdparty/mast3r
```

5. Build WildPose's CUDA backend and install Python requirements.

```bash
pip install --no-build-isolation .
pip install -r requirements.txt
```

6. Check the installation.

```bash
python - <<'PY'
import torch
import lietorch
import mast3r
import droid_backends
print("CUDA available:", torch.cuda.is_available())
PY
```

## 📦 Checkpoints

Create a `pretrained/` directory and download both checkpoints from the WildPose Hugging Face repository:

```bash
mkdir -p pretrained/
wget https://huggingface.co/gradient-spaces/WildPose/resolve/main/wildpose_v0.pth -P pretrained/
wget https://huggingface.co/gradient-spaces/WildPose/resolve/main/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P pretrained/
```

## 🚀 Run WildPose

### Benchmark Dynamic

Download the dynamic benchmark datasets:

```bash
bash scripts_downloading/download_dynamic_all.sh
```

This downloads the Wild-SLAM Mocap, Bonn Dynamic, and TUM RGB-D dynamic sequences used by `scripts_run/run_dynamic_all.sh` into:

```text
datasets/
  Wild_SLAM_Mocap/
  Bonn/
  TUM_RGBD/
```

Run all dynamic benchmarks:

```bash
bash scripts_run/run_dynamic_all.sh
```

You can also run one benchmark group at a time:

```bash
bash scripts_run/run_dynamic_all.sh wild_slam_mocap
bash scripts_run/run_dynamic_all.sh bonn
bash scripts_run/run_dynamic_all.sh tum_rgbd
```

Results are saved under the `output/` directory. Each benchmark has one folder, and each scene/sequence has its own run folder:

```text
output/
  <benchmark>/
    <scene>/
      cfg.yaml
      video.npz
      traj/
        est_poses_full.txt
        metrics_full_traj.txt
        metrics_kf_traj.txt
        full_traj_2d.png
        kf_traj_2d.png
        before_final_ba/
          metrics_kf_traj.txt
          kf_traj_2d.png
```

For example, a Wild-SLAM Mocap run writes to `output/Wild_SLAM_Mocap/crowd/`, while a 7-Scenes run writes to `output/7scenes/chess/`. The estimated full trajectory is stored in TUM format at `traj/est_poses_full.txt`, and the main pose metrics are summarized in `traj/metrics_full_traj.txt`.

### Benchmark Static

Download the static benchmark datasets:

```bash
bash scripts_downloading/download_static_all.sh
```

This downloads the 7-Scenes and TUM RGB-D static benchmark sequences used by `scripts_run/run_static_all.sh` into:

```text
datasets/
  7-scenes/
  TUM_RGBD/
```

Run all static benchmarks:

```bash
bash scripts_run/run_static_all.sh
```

You can also run one benchmark group at a time:

```bash
bash scripts_run/run_static_all.sh seven_scenes
bash scripts_run/run_static_all.sh tum_ablation
```

## 🙏 Acknowledgements

This repository is initialized from [WildGS-SLAM](https://github.com/GradientSpaces/WildGS-SLAM), and the README structure follows its release style. We thank the WildGS-SLAM authors for releasing their codebase.

WildPose also builds on ideas and components from DROID-SLAM, lietorch, MASt3R/DUSt3R, GO-SLAM/GlORIE-SLAM-style factor graph optimization, MoGe, and standard RGB-D trajectory evaluation tools. We thank the authors of these projects for making their work publicly available.

This work is supported by the Center for Integrated Facility Engineering (CIFE) and the Stanford Robotics Center (SRC). We also thank Stanford's Marlowe and Sherlock clusters for providing GPU computing resources for model training and evaluation.

## 📚 Citation

If you find this code or paper useful, please cite:

```bibtex
@inproceedings{Zheng2026WildPose,
  author    = {Zheng, Jianhao and Zhu, Liyuan and Zhu, Zihan and Armeni, Iro},
  title     = {WildPose: A Unified Framework for Robust Pose Estimation in the Wild},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## ✉ Contact

For questions, comments, and bug reports, please contact [Jianhao Zheng](mailto:jianhao@stanford.edu).
