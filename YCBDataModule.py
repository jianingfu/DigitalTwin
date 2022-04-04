import os
import torch
import random
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb

class YCBDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        print("preparing data...")
        os.system('./download.sh')
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        print("setting up data")
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = PoseDataset_ycb('train', cfg = self.cfg)
        self.test_dataset = PoseDataset_ycb('test', cfg = self.cfg)
        self.sym_list = self.train_dataset.get_sym_list()
        self.num_points_mesh = self.train_dataset.get_num_points_mesh()
        print("num points mesh", self.num_points_mesh)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.workers)
