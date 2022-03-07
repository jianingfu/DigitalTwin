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
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
import pytorch_lightning as pl
from lib.tools import compute_rotation_matrix_from_ortho6d


class DenseFusionModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.opt.manualSeed = random.randint(1, 10000) 
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        self.estimator = PoseNet(num_points=self.opt.num_points, num_obj=self.opt.num_objects, use_normals=self.opt.use_normals)
        self.refiner = PoseRefineNet(num_points=self.opt.num_points, num_obj=self.opt.num_objects, use_normals=self.opt.use_normals)
        self.best_test = np.Inf
        self.criterion = Loss(self.opt.num_rot_bins, self.opt.sym_list, self.opt.use_normals)
        self.criterion_refine = Loss_refine(self.opt.num_rot_bins, self.opt.sym_list, self.opt.use_normals)
        if self.opt.old_batch_mode:
            self.automatic_optimization = False
            self.old_batch_size = opt.batch_size
            self.opt.batch_size = 1
            self.opt.image_size = -1

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
        points, choose, img, target, target_front, model_points, front, idx = batch

        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)

        loss, dis, new_points, new_target, new_target_front = self.criterion(pred_r, pred_t, pred_c, target, target_front,
                                                                        model_points, front, idx, points, self.opt.w,
                                                                        self.opt.refine_start)

        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_r, pred_t = self.refiner(new_points, emb, idx)
                loss, dis, new_points, new_target, new_target_front = self.criterion_refine(pred_r, pred_t, new_target,
                                                                                            new_target_front, model_points,
                                                                                            front, idx, new_points)
                loss.backward()

        if self.opt.old_batch_mode:
            if batch_idx != 0 and batch_idx % self.old_batch_size == 0:
                opt = self.optimizers()
                opt.step()
                opt.zero_grad()

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, target, target_front, model_points, front, idx = batch
        pred_r, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        _, dis, new_points, new_target, new_target_front = self.criterion(pred_r, pred_t, pred_c, target, target_front,
                                                                          model_points, front, idx, points, self.opt.w,
                                                                          self.opt.refine_start)

        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_r, pred_t = self.refiner(new_points, emb, idx)
                loss, dis, new_points, new_target, new_target_front = self.criterion_refine(pred_r, pred_t, new_target,
                                                                                            new_target_front, model_points,
                                                                                            front, idx, new_points)
        
        # visualize
        if batch_idx == 0 and self.opt.visualize:
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

        val_loss = loss.item()
        self.log('val_loss', val_loss, logger=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        test_loss = np.average(np.array(outputs))
        if test_loss <= self.best_test:
            self.best_test = test_loss
            if self.opt.refine_start:
                torch.save(self.refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(self.opt.outf,
                                                                                                 self.current_epoch,
                                                                                                 test_loss))
            else:
                torch.save(self.estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf,
                                                                                            self.current_epoch,
                                                                                            test_loss))
        print("best_test: ", self.best_test)


        if self.best_test < self.opt.decay_margin and not self.opt.decay_start:
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.trainer.optimizers[0] = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        if (self.current_epoch >= self.opt.refine_epoch or self.best_test < self.opt.refine_margin) and not self.opt.refine_start:
            print('======Refine started!========')
            self.opt.refine_start = True
            if self.opt.old_batch_mode:
                self.old_batch_size = int(self.old_batch_size / self.opt.iteration)
            self.trainer.optimizers[0] = optim.Adam(self.refiner.parameters(), lr=self.opt.lr)

            # re-setup dataset
            self.trainer.datamodule.setup(None)
            self.opt.sym_list = self.trainer.datamodule.sym_list
            self.opt.num_points_mesh = self.trainer.datamodule.num_points_mesh
            self.criterion = Loss(self.opt.num_points_mesh, self.opt.sym_list, self.opt.use_normals)
            self.criterion_refine = Loss_refine(self.opt.num_points_mesh, self.opt.sym_list, self.opt.use_normals)
            print("start reloading data")
            self.trainer.datamodule.train_dataloader()
            self.trainer.datamodule.val_dataloader()
            

    def configure_optimizers(self):
        # if self.opt.resume_posenet != '':
            # self.estimator.load_state_dict(torch.load('{0}/{1}'.format(self.opt.outf, self.opt.resume_posenet)))
            # self.estimator.load_state_dict(torch.load(self.opt.resume_posenet))

        if self.opt.resume_refinenet != '':
            # self.refiner.load_state_dict(torch.load('{0}/{1}'.format(self.opt.outf, self.opt.resume_refinenet)))
            self.refiner.load_state_dict(torch.load(self.opt.resume_refinenet))
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            if self.opt.old_batch_mode:
                self.old_batch_size = int(self.old_batch_size / self.opt.iteration)
            optimizer = optim.Adam(self.refiner.parameters(), lr=self.opt.lr)
        else:
            self.opt.refine_start = False
            self.opt.decay_start = False
            optimizer = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)
        
        return optimizer


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
