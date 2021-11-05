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
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        self.estimator = PoseNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects)
        self.refiner = PoseRefineNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects)
        if self.opt.start_epoch == 1:
            for log in os.listdir(self.opt.log_dir):
                os.remove(os.path.join(self.opt.log_dir, log))

    def on_pretrain_routine_start(self):
        self.opt.sym_list = self.trainer.datamodule.sym_list
        self.opt.num_points_mesh = self.trainer.datamodule.num_points_mesh
        print(self.opt.sym_list)
        print(self.opt.num_points_mesh)
        self.criterion = Loss(self.opt.num_points_mesh, self.opt.sym_list)
        self.criterion_refine = Loss_refine(self.opt.num_points_mesh, self.opt.sym_list)
        self.best_test = np.Inf

    def on_train_epoch_start(self):
        self.train_dis_avg = 0.0
        # TODO: do we need this?
        if self.opt.refine_start:
            self.estimator.eval()
            self.refiner.train()
        else:
            self.estimator.train()
    
    # TODO: check refine
    def backward(self, loss, optimizer, optimizer_idx):
        if not self.opt.refine_start:
            loss.backward()

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        points, choose, img, target, model_points, idx = batch
        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        torch.cuda.empty_cache()
        loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, target, 
                                                            model_points, idx, points, self.opt.w, 
                                                            self.opt.refine_start)

                        
        
        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_r, pred_t = self.refiner(new_points, emb, idx)
                dis, new_points, new_target = self.criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                dis.backward()

        self.log('train_dis', dis, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_start(self):
        self.test_dis = 0.0

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, target, model_points, idx = batch
        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        _, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, self.opt.w, self.opt.refine_start)

        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_r, pred_t = self.refiner(new_points, emb, idx)
                dis, new_points, new_target = self.criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

        self.test_dis += dis.item()
        val_loss = dis.item()
        self.log('val_loss', val_loss)
        # logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        return val_loss

    def validation_epoch_end(self, outputs):
        test_dis = np.average(np.array(outputs))
        if test_dis <= self.best_test:
            self.best_test = test_dis
        print("best_test: ", self.best_test) 

        if self.best_test < self.opt.decay_margin and not self.opt.decay_start:
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.trainer.optimizers[0] = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        if self.best_test < self.opt.refine_margin and not self.opt.refine_start:
            print('======Refine started!========')
            self.opt.refine_start = True
            self.opt.batch_size = int(self.opt.batch_size / self.opt.iteration)
            self.trainer.optimizers[0] = optim.Adam(self.refiner.parameters(), lr=self.opt.lr)

            # re-setup dataset, TODO: double check/test this
            self.trainer.datamodule.refine = True
            self.trainer.datamodule.setup(None)
            self.opt.sym_list = self.trainer.datamodule.sym_list
            self.opt.num_points_mesh = self.trainer.datamodule.num_points_mesh
            self.criterion = Loss(self.opt.num_points_mesh, self.opt.sym_list)
            self.criterion_refine = Loss_refine(self.opt.num_points_mesh, self.opt.sym_list)
            self.trainer.datamodule.train_dataloader = self.trainer.datamodule.train_dataloader()
            self.trainer.datamodule.val_dataloader = self.trainer.datamodule.val_dataloader()
            

    def configure_optimizers(self):
        if self.opt.resume_posenet != '':
            self.estimator.load_state_dict(torch.load('{0}/{1}'.format(self.opt.outf, self.opt.resume_posenet)))

        if self.opt.resume_refinenet != '':
            self.refiner.load_state_dict(torch.load('{0}/{1}'.format(self.opt.outf, self.opt.resume_refinenet)))
            self.opt.refine_start = True
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.opt.batch_size = int(self.opt.batch_size / self.opt.iteration)
            optimizer = optim.Adam(self.refiner.parameters(), lr=self.opt.lr)
        else:
            self.opt.refine_start = False
            self.opt.decay_start = False
            optimizer = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)
        
        return optimizer