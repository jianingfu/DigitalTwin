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
        self.refine = False
        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        num_objects = 21 #number of object classes in the dataset
        num_points = 1000 #number of points on the input pointcloud
        outf = 'trained_models/ycb' #folder to save trained models
        log_dir = 'experiments/logs/ycb' #folder to save logs
        repeat_epoch = 1 #number of repeat times for one epoch training

    def prepare_data(self):
        os.system('./download.sh')
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if not self.refine:
            self.train_dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, True)
            self.test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, True)
        else:
            self.train_dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            self.test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
        self.sym_list = self.train_dataset.get_sym_list()
        self.num_points_mesh = self.train_dataset.get_num_points_mesh()

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    def val_dataloader(self):
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)