import numpy as np
import torch
import argparse
import os

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from colorama import Fore,Style
import shutil

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(args.config)
    setup_seed(cfg['setup_seed'])

    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['scene']}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg)

    slam = SLAM(cfg,dataset)
    slam.run()


    if cfg['delete_mono_priors']:
        shutil.rmtree(f"{output_dir}/mono_priors")
