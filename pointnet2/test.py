import os
import sys

sys.path.insert(0,"/kaggle/working/Pointnet2_PyTorch/")
sys.path.insert(0,"/kaggle/working/Pointnet2_PyTorch/pointnet2_ops_lib/")
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))