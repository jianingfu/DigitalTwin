import os
import torch
import random
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ycb.dataset import PoseDataset as PoseDataset_linemod # TODO: add linmond dataset

class LinemodDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.refine = False

    def prepare_data(self):
        print("preparing data...")
        os.system('./download.sh')
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = PoseDataset_linemod('train', self.opt.num_points, True, self.opt.dataset_root, self.opt.noise_trans, self.opt.refine_start)
        self.test_dataset = PoseDataset_linemod('test', self.opt.num_points, False, self.opt.dataset_root, 0.0, self.opt.refine_start)
        self.sym_list = self.train_dataset.get_sym_list()
        self.num_points_mesh = self.train_dataset.get_num_points_mesh()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=self.opt.workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.opt.workers)
