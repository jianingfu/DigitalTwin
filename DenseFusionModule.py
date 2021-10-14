import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import pytorch_lightning as pl

class DenseFusionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions\
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        
        return loss

    def validation_step(self, batch, batch_idx):


    def test_step(self, batch, batch_idx):


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
