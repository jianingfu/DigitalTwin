import os
import torch
from torch import nn
import pytorch_lightning as pl

class YCBDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
    def train_dataloader(self):
        train_split = Dataset(...)
        return DataLoader(train_split)
    def val_dataloader(self):
        val_split = Dataset(...)
        return DataLoader(val_split)
    def test_dataloader(self):
        test_split = Dataset(...)
        return DataLoader(test_split)
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP