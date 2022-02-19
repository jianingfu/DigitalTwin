import os
import torch
import random
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb

class YCBDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def prepare_data(self):
        print("preparing data...")
        os.system('./download.sh')
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        print("setting up data")
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = PoseDataset_ycb('train', self.opt.num_points, True, self.opt.dataset_root, self.opt.noise_trans, 
                            num_rot_bins = self.opt.num_rot_bins, image_size = self.opt.image_size, append_depth_to_image=self.opt.append_depth_to_image)
        self.test_dataset = PoseDataset_ycb('test', self.opt.num_points, False, self.opt.dataset_root, 
                            0.0, num_rot_bins = self.opt.num_rot_bins, image_size = self.opt.image_size, append_depth_to_image=self.opt.append_depth_to_image)
        self.sym_list = self.train_dataset.get_sym_list()
        self.num_points_mesh = self.train_dataset.get_num_points_mesh()
        print("num points mesh", self.num_points_mesh)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=self.opt.workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.workers)
