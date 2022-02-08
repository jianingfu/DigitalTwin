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
from datetime import datetime


class DenseFusionModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.opt.manualSeed = random.randint(1, 10000) 
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        self.estimator = PoseNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects, num_rot_bins = self.opt.num_rot_bins)
        self.refiner = PoseRefineNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects, num_rot_bins = self.opt.num_rot_bins)
        # if self.opt.start_epoch == 1:
        #     if os.path.isdir(self.opt.log_dir):
        #         for log in os.listdir(self.opt.log_dir):
        #             os.remove(os.path.join(self.opt.log_dir, log))
        self.best_test = np.Inf
        self.criterion = Loss(self.opt.num_rot_bins)
        self.criterion_refine = Loss_refine(self.opt.num_rot_bins)

    def on_train_epoch_start(self):
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
        if self.opt.profile:
                    print("starting training sample {0} {1}".format(batch_idx, datetime.now()))
        # training_step defined the train loop.
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch

        pred_front, pred_rot_bins, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)

        # torch.cuda.empty_cache()
        if self.opt.profile:
            print("finished forward pass {0} {1}".format(batch_idx, datetime.now()))

        loss, new_points, new_rot_bins, new_t, front_loss, rot_bins_loss, t_loss, mean_pred_c, max_pred_c = self.criterion(pred_front, pred_rot_bins, pred_t, 
                                                                pred_c, front_r, rot_bins, front_orig, 
                                                                t, idx, model_points, points, self.opt.w, 
                                                                self.opt.refine_start)

        if self.opt.profile:
            print("finished loss {0} {1}".format(batch_idx, datetime.now()))
        
        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)
                loss, new_points, new_rot_bins, new_t, front_loss, rot_loss, t_loss = self.criterion_refine(pred_front, pred_rot_bins, 
                                    pred_t, front_r, new_rot_bins, front_orig, new_t, idx, new_points)
                loss.backward()

        # self.log_dict({'train_dis':dis, 'loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('front_loss', front_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('rot_bins_loss', rot_bins_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('t_loss', t_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('mean_pred_c', mean_pred_c, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('max_pred_c', max_pred_c, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.opt.profile:
                    print("finished training sample {0} {1}".format(batch_idx, datetime.now()))
        return loss

    def on_validation_epoch_start(self):
        self.test_loss = 0.0

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch
        pred_front, pred_rot_bins, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        loss, new_points, new_rot_bins, new_t, front_loss, rot_bins_loss, t_loss, mean_pred_c, max_pred_c  = self.criterion(pred_front, pred_rot_bins, pred_t, 
                                        pred_c, front_r, rot_bins, front_orig, t, idx, model_points, 
                                        points, self.opt.w, self.opt.refine_start)

        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)
                loss, new_points, new_rot_bins, new_t = self.criterion_refine(pred_front, pred_rot_bins, 
                            pred_t, front_r, new_rot_bins, front_orig, new_t, idx, new_points)
        self.test_loss += loss.item()
        val_loss = loss.item()
        self.log('val_loss', val_loss, logger=True)
        self.log('front_loss', front_loss, logger=True)
        self.log('rot_bins_loss', rot_bins_loss, logger=True)
        self.log('t_loss', t_loss, logger=True)
        self.log('mean_pred_c', mean_pred_c, logger=True)
        self.log('max_pred_c', max_pred_c, logger=True)
        # logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        return val_loss

    def validation_epoch_end(self, outputs):
        test_loss = np.average(np.array(outputs))
        if test_loss <= self.best_test:
            self.best_test = test_loss
            if self.opt.refine_start:
                torch.save(self.refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(self.opt.outf, self.current_epoch, test_loss))
            else:
                torch.save(self.estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, self.current_epoch, test_loss))
        print("best_test: ", self.best_test) 

        if self.best_test < self.opt.decay_margin and not self.opt.decay_start:
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.trainer.optimizers[0] = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        if self.best_test < self.opt.refine_margin and not self.opt.refine_start:
            print('======Refine started!========')
            self.opt.refine_start = True
            # self.opt.batch_size = int(self.opt.batch_size / self.opt.iteration)
            self.trainer.optimizers[0] = optim.Adam(self.refiner.parameters(), lr=self.opt.lr)

            # re-setup dataset, TODO: double check/test this
            self.trainer.datamodule.setup(None)
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