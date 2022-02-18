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
from lib.network import PoseNet
from lib.loss import Loss
from lib.utils import setup_logger
import pytorch_lightning as pl
from datetime import datetime
from tensorboard.plugins.mesh import summary as mesh_summary
from lib.transformations import rotation_matrix_of_axis_angle, \
        rotation_matrix_from_vectors_procedure


class DenseFusionModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.opt.manualSeed = random.randint(1, 10000) 
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        self.estimator = PoseNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects, num_rot_bins = self.opt.num_rot_bins)

        self.best_test = np.Inf
        self.criterion = Loss(self.opt.num_rot_bins)


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch

        pred_front, pred_rot_bins, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)

        # torch.cuda.empty_cache()

        loss, new_points, new_rot_bins, new_t, new_front_r = self.criterion(pred_front, 
                        pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, 
                        model_points, points, self.opt.w)

        # self.log_dict({'train_dis':dis, 'loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.test_loss = 0.0

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch
        pred_front, pred_rot_bins, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)

        loss, new_points, new_rot_bins, new_t, new_front_r = self.criterion(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, self.opt.w)

        # if self.opt.refine_start:
        #     for ite in range(0, self.opt.iteration):
        #         pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)
        #         loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = \
        #             self.criterion_refine(pred_front, pred_rot_bins, pred_t, new_front_r, 
        #                                 new_rot_bins, new_front_orig, new_t, idx, new_points)       
        
        # visualize
        if (batch_idx == 0 and self.opt.visualize):
            gt_vis, pred_vis = self.get_vis_points(model_points[0], points[0], pred_front[0], pred_rot_bins[0], pred_t[0], pred_c[0], 
                    emb[0], front_r[0], rot_bins[0], front_orig[0], t[0], new_points[0], new_t[0], new_front_r[0])

            vis = np.concatenate((gt_vis, pred_vis), axis = 1)
            gt_color = np.zeros((gt_vis.shape))
            gt_color[:,:,0] = 200
            pred_color = np.zeros((pred_vis.shape))
            pred_color[:,:,1] = 200
            colors = np.concatenate((gt_color, pred_color), axis = 1)
            # summary = mesh_summary.op('point_cloud', vertices=gt_vis)
            self.logger.experiment.add_mesh('comb_vis', vertices=vis, colors= colors)    

        self.test_loss += loss.item()
        val_loss = loss.item()
        self.log('val_loss', val_loss, logger=True)
        # logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        return val_loss

    def validation_epoch_end(self, outputs):
        test_loss = np.average(np.array(outputs))
        if test_loss <= self.best_test:
            self.best_test = test_loss
            torch.save(self.estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, self.current_epoch, test_loss))
        print("best_test: ", self.best_test) 


    def configure_optimizers(self):
        optimizer = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)
        
        return optimizer

    def get_vis_points(self, model_points, points, pred_front, pred_rot_bins, pred_t, pred_c, 
                    emb, front_r, rot_bins, front_orig, t, new_points, new_t, new_front_r):
         #immediately shift pred_front to model coordinates
        pred_front = pred_front - pred_t

        num_p, _ = pred_c.shape
        bs = 1 # just visualizting the first one

        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)

        pred_t_points_vis = self.visualize_pointcloud(pred_t + points)

        gt_t_vis = self.visualize_pointcloud(t)
        pred_t_vis = self.visualize_pointcloud(t- new_t)

        gt_front_vis = self.visualize_fronts(front_r, t)
        pred_fronts_estimator_vis = self.visualize_fronts(pred_front, pred_t + points)

        new_points_vis = self.visualize_pointcloud(new_points)
        new_gt_front = self.visualize_fronts(new_front_r, new_t)

        #which_max -> bs
        #pred_t -> bs * num_p * 3
        #points -> bs * num_p * 3

        which_max_3 = which_max.view(bs, 1, 1).repeat(1, 1, 3)
        which_max_rot_bins = which_max.view(bs, 1, 1).repeat(1, 1, self.opt.num_rot_bins)

        # delete these for bs > 1
        which_max_3 = which_max_3[0]
        which_max_rot_bins = which_max_rot_bins[0]

        #best_c_pred_t -> bs * 1 * 3
        best_c_pred_t = pred_t[which_max] + points[which_max]
        # best_c_pred_t = torch.gather(pred_t, 1, which_max_3) + torch.gather(points, 1, which_max_3)

        #we need to calculate the actual transformation that our rotation rep. represents

        best_c_pred_front = pred_front[which_max].squeeze(1)
        best_c_rot_bins = pred_rot_bins[which_max].squeeze(1)
        # best_c_pred_front = torch.gather(pred_front, 1, which_max_3).squeeze(1)
        # best_c_rot_bins = torch.gather(pred_rot_bins, 1, which_max_rot_bins).squeeze(1)

        #get the angle in radians based on highest histogram bin
        #angle -> bs * 1
        angle = (torch.argmax(best_c_rot_bins, axis=1) / best_c_rot_bins.shape[1] * 2 * np.pi)

        my_front = best_c_pred_front.squeeze()
        my_theta = angle.squeeze()
        my_t = best_c_pred_t.squeeze()

        gt_front = front_r.squeeze()
        gt_theta = torch.argmax(rot_bins) / self.opt.num_rot_bins * 2 * np.pi
        gt_t = t.squeeze()


        selected_pred_front_vis = self.visualize_fronts(my_front, my_t)

        #pts_gt = get_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        #pts_pred = get_points(model_points, front_orig, my_front, my_theta, my_t)

        #dist = dist2(pts_gt, pts_pred)
        #dists.append(np.mean(np.min(dist, axis=1)))

        gt_vis = self.visualize_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        pred_vis = self.visualize_points(model_points, front_orig, my_front, my_theta, my_t)
        projected_depth_vis = self.visualize_pointcloud(points)

        return gt_vis, pred_vis

    def visualize_fronts(self, fronts, t):
        fronts = fronts.cpu().detach().numpy()
        fronts = fronts.reshape((-1, 3))

        t = t.cpu().detach().numpy()
        t = t.reshape((-1, 3))

        front_points = fronts + t
        front_points = torch.tensor(front_points[None,:])
        return front_points

    def visualize_points(self, model_points, front_orig, front, angle, t):

        model_points = model_points.cpu().detach().numpy()
        front_orig = front_orig.cpu().detach().numpy().squeeze()
        front = front.cpu().detach().numpy()
        angle = angle.cpu().detach().numpy()
        t = t.cpu().detach().numpy()

        Rf = rotation_matrix_from_vectors_procedure(front_orig, front)

        R_axis = rotation_matrix_of_axis_angle(front, angle)

        R_tot = (R_axis @ Rf)

        pts = (model_points @ R_tot.T + t).squeeze()
        pts = torch.tensor(pts[None,:])
        return pts

    def visualize_pointcloud(self, points):

        points = points.cpu().detach().numpy()

        points = points.reshape((-1, 3))
        points = torch.tensor(points[None,:])
        return points