from tkinter import VERTICAL
import _init_paths
import os
import random
import copy
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
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ExponentialLR
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import pytorch_lightning as pl
from lib.randla_utils import randla_processing
from lib.tools import compute_rotation_matrix_from_ortho6d
from cfg.config import write_config


class DenseFusionModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.cfg.manualSeed = random.randint(1, 10000)
        random.seed(self.cfg.manualSeed)
        torch.manual_seed(self.cfg.manualSeed)

        self.estimator = PoseNet(cfg = self.cfg)
        self.refiner = PoseRefineNet(cfg = self.cfg)
        self.best_test = np.Inf
        

    def on_pretrain_routine_start(self):
        self.cfg.sym_list = self.trainer.datamodule.sym_list
        self.cfg.num_points_mesh = self.trainer.datamodule.num_points_mesh
        self.criterion = Loss(self.cfg.num_points_mesh, self.cfg.sym_list, self.cfg.use_normals, self.cfg.use_confidence)
        self.criterion_refine = Loss_refine(self.cfg.num_points_mesh, self.cfg.sym_list, self.cfg.use_normals)

    def on_train_epoch_start(self):
        # TODO: do we need this?
        if self.cfg.refine_start:
            self.estimator.eval()
            self.refiner.train()
        else:
            self.estimator.train()
    
    # TODO: check refine
    def backward(self, loss, cfgimizer, cfgimizer_idx):
        if not self.cfg.refine_start:
            loss.backward()

    def on_train_epoch_start(self):
        write_config(self.cfg, os.path.join(self.cfg.log_dir, "config_current.yaml"))

    def training_step(self, batch, batch_idx):
        end_points = batch
        if self.cfg.pcld_encoder == "randlanet":
            end_points = randla_processing(end_points, self.cfg)

        #pred_r, pred_t, pred_c, emb = estimator(end_points)
        end_points = self.estimator(end_points)

        #loss, dis, new_points, new_target, new_target_front = criterion(pred_r, pred_t, pred_c, end_points, opt.w, opt.refine_start)
        loss, dis, end_points = self.criterion(end_points, self.cfg.w, self.cfg.refine_start)
        
        if self.cfg.refine_start:
            for ite in range(0, self.cfg.iteration):
                end_points = self.refiner(end_points, ite)
                loss, dis, end_points = self.criterion_refine(end_points, ite)
                loss.backward()

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('dis', dis, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):

        end_points = batch
        if self.cfg.pcld_encoder == "randlanet":
            end_points = randla_processing(end_points, self.cfg)

        #pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        end_points = self.estimator(end_points)


        #_, dis, new_points, new_target, new_target_front = criterion(pred_r, pred_t, pred_c, target, target_front, model_points, front, idx, points, opt.w, opt.refine_start)
        _, dis, end_points = self.criterion(end_points, self.cfg.w, self.cfg.refine_start)

        if self.cfg.refine_start:
            for ite in range(0, self.cfg.iteration):
                end_points = self.refiner(end_points, ite)
                _, dis, end_points = self.criterion_refine(end_points, ite)
               
        # visualize
        if batch_idx == 0 and self.cfg.visualize:
            bs, num_p, _ = pred_c.shape
            pred_c = pred_c.view(bs, num_p)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_p, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).unsqueeze(0).unsqueeze(0)

            my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

            my_t = (points.view(bs * num_p, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

            my_r = copy.deepcopy(my_rot_mat)


            #projected depth image
            projected_vis = self.visualize_pointcloud(points)
            projected_color = np.zeros((projected_vis.shape))
            projected_color[:,:,2] = 100

            pred_vis = self.visualize_points(model_points, my_t, my_r)
            target_vis = self.visualize_pointcloud(target)

            t_vis = np.concatenate((target_vis, pred_vis, projected_vis), axis=1)
            gt_t_color = np.zeros((target_vis.shape))
            gt_t_color[:,:,0] = 200
            pred_t_color = np.zeros((pred_vis.shape))
            pred_t_color[:,:,1] = 200
            t_colors = np.concatenate((gt_t_color, pred_t_color, projected_color), axis=1)
            self.logger.experiment.add_mesh(str(self.current_epoch) + 't_vis ', vertices=t_vis, colors=t_colors)

        val_loss = dis.item()
        self.log('val_dis', val_loss, logger=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        torch.save(self.estimator.state_dict(), '{0}/pose_model_current.pth'.format(self.cfg.outf))

        test_loss = np.average(np.array(outputs))
        if test_loss <= self.best_test:
            self.best_test = test_loss
            if self.cfg.refine_start:
                torch.save(self.refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(self.cfg.outf,
                                                                                                 self.current_epoch,
                                                                                                 test_loss))
            else:
                torch.save(self.estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.cfg.outf,
                                                                                            self.current_epoch,
                                                                                            test_loss))
        print("best_test: ", self.best_test)


        if self.best_test < self.cfg.decay_margin and not self.cfg.decay_start:
            self.cfg.decay_start = True
            self.cfg.w *= self.cfg.w_rate
            self.trainer.optimizers[0] = optim.Adam(self.estimator.parameters(), lr=self.cfg.lr)

        if (self.current_epoch >= self.cfg.refine_epoch or self.best_test < self.cfg.refine_margin) and not self.cfg.refine_start:
            print('======Refine started!========')
            self.cfg.refine_start = True
            # if self.cfg.old_batch_mode:
            #     self.old_batch_size = int(self.old_batch_size / self.cfg.iteration)
            self.trainer.optimizers[0] = optim.Adam(self.refiner.parameters(), lr=self.cfg.lr)

            # re-setup dataset
            self.trainer.datamodule.setup(None)
            self.cfg.sym_list = self.trainer.datamodule.sym_list
            self.cfg.num_points_mesh = self.trainer.datamodule.num_points_mesh
            print("start reloading data")
            self.trainer.datamodule.train_dataloader()
            self.trainer.datamodule.val_dataloader()
            self.criterion = Loss(self.cfg.num_points_mesh, self.cfg.sym_list, self.cfg.use_normals, self.cfg.use_confidence)
            self.criterion_refine = Loss_refine(self.cfg.num_points_mesh, self.cfg.sym_list, self.cfg.use_normals)
            

    def configure_optimizers(self):
        # if self.cfg.resume_posenet != '':
            # self.estimator.load_state_dict(torch.load('{0}/{1}'.format(self.cfg.outf, self.cfg.resume_posenet)))
            # self.estimator.load_state_dict(torch.load(self.cfg.resume_posenet))

        if self.cfg.refine_start:
            # self.refiner.load_state_dict(torch.load('{0}/{1}'.format(self.cfg.outf, self.cfg.resume_refinenet)))
            self.refiner.load_state_dict(torch.load(self.cfg.resume_refinenet))
            self.cfg.w *= self.cfg.w_rate
            # if self.cfg.old_batch_mode:
            #     self.old_batch_size = int(self.old_batch_size / self.cfg.iteration)
            optimizer = optim.Adam(self.refiner.parameters(), lr=self.cfg.lr)
        else:
            optimizer = optim.Adam(self.estimator.parameters(), lr=self.cfg.lr)
        
        if self.cfg.lr_scheduler == "cyclic":
            clr_div = 6
            lr_scheduler = CyclicLR(
                optimizer, base_lr=1e-5, max_lr=3e-4,
                cycle_momentum=False,
                step_size_up=self.cfg.nepoch * (len(dataset) / self.cfg.batch_size) // clr_div,
                step_size_down=self.cfg.nepoch * (len(dataset) / self.cfg.batch_size) // clr_div,
                mode='triangular'
            )
        elif self.cfg.lr_scheduler == "cosine":
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.nepoch * (len(dataset) / self.cfg.batch_size))
        elif self.cfg.lr_scheduler == "exponential":
            lr_scheduler = ExponentialLR(optimizer, 0.9)
        else:
            lr_scheduler = None
        
        return [optimizer], [lr_scheduler]


    def visualize_pointcloud(self, points):

        points = points.cpu().detach().numpy()

        points = points.reshape((-1, 3))
        points = torch.tensor(points[None,:])
        return points

    def visualize_points(model_points, t, rot_mat, label):

        model_points = model_points.cpu().detach().numpy()

        pts = (model_points @ rot_mat.T + t).squeeze()
        points = torch.tensor(pts[None,:])
        return points
