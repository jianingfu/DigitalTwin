import _init_paths
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


class DenseFusionModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.opt.manualSeed = random.randint(1, 10000) 
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)

        self.opt.sym_list = self.trainer.datamodule.sym_list
        self.opt.num_points_mesh = self.trainer.datamodule.num_points_mesh
        self.estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
        self.refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
        if opt.start_epoch == 1:
            for log in os.listdir(self.opt.log_dir):
                os.remove(os.path.join(self.opt.log_dir, log))

    def on_train_start(self):
        self.criterion = Loss(opt.num_points_mesh, opt.sym_list)
        self.criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)
        self.best_test = np.Inf

    def on_train_epoch_start(self):
        self.train_dis_avg = 0.0
        # TODO: do we need this?
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        points, choose, img, target, model_points, idx = batch
        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
        
        if opt.refine_start:
            for ite in range(0, opt.iteration):
                pred_r, pred_t = self.refiner(new_points, emb, idx)
                dis, new_points, new_target = self.criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                dis.backward()
        else:
            loss.backward()

        # TODO: average just for printing?? return what?
        train_dis_avg += dis.item()

        self.log(train_dis_avg / opt.batch_size)

        return loss

    def on_validation_epoch_start(self):
        self.test_dis = 0.0

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, target, model_points, idx = batch
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

        if opt.refine_start:
            for ite in range(0, opt.iteration):
                pred_r, pred_t = refiner(new_points, emb, idx)
                dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

        self.test_dis += dis.item()
        val_loss = self.test_dis / batch_idx
        self.log('val_loss', val_loss)
        # logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        return val_loss

    def on_validation_epoch_end(self, outputs):
        test_dis = outputs
        if test_dis <= self.best_test:
            best_test = test_dis

        if self.best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            self.trainer.optimizers[0] = optim.Adam(estimator.parameters(), lr=opt.lr)

        if self.best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            self.trainer.optimizers[0] = optim.Adam(refiner.parameters(), lr=opt.lr)

            # re-setup dataset, TODO: double check/test this
            self.trainer.datamodule.refine = True
            self.trainer.datamodule.setup()

            self.criterion = Loss(opt.num_points_mesh, opt.sym_list)
            self.criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    def configure_optimizers(self):
        if opt.resume_posenet != '':
            estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

        if opt.resume_refinenet != '':
            refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
            opt.refine_start = True
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
        else:
            opt.refine_start = False
            opt.decay_start = False
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        
        return optimizer