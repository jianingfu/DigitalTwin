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
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import pytorch_lightning as pl


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()


class DenseFusionModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
        # TODO: import num_points_mesh from DataModule
        criterion = Loss(opt.num_points_mesh, opt.sym_list)
        criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions\
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        points, choose, img, target, model_points, idx = batch
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
        
        if opt.refine_start:
            for ite in range(0, opt.iteration):
                pred_r, pred_t = refiner(new_points, emb, idx)
                dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                dis.backward()
        else:
            loss.backward()

        train_dis_avg += dis.item()

        # TODO: save ckpt & log

        return loss

    def validation_step(self, batch, batch_idx):
        estimator.eval()
        refiner.eval()

    def test_step(self, batch, batch_idx):
        points, choose, img, target, model_points, idx = batch
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

        if opt.refine_start:
            for ite in range(0, opt.iteration):
                pred_r, pred_t = refiner(new_points, emb, idx)
                dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

        test_dis += dis.item()
        logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        # TODO:  test_dis = test_dis / test_count & log

    # TODO: ckpt save bst model

    def configure_optimizers(self):
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        est_optimizer = torch.optim.Adam(estimator.parameters(), lr=opt.lr)
        
        opt.batch_size = int(opt.batch_size / opt.iteration)
        ref_optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
        
        return [est_optimizer, ref_optimizer]
