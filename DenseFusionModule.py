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
from lib.network import PoseNet
from lib.loss import Loss
from lib.utils import setup_logger
import pytorch_lightning as pl
from datetime import datetime
from tensorboard.plugins.mesh import summary as mesh_summary
from lib.transformations import rotation_matrix_of_axis_angle, \
        rotation_matrix_from_vectors_procedure
from lib.meanshift_pytorch import MeanShiftTorch


class DenseFusionModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.opt.manualSeed = random.randint(1, 10000) 
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        num_image_channels = 3
        if self.opt.append_depth_to_image:
            num_image_channels += 3

        self.estimator = PoseNet(num_points = self.opt.num_points, num_obj = self.opt.num_objects, num_rot_bins = self.opt.num_rot_bins, num_image_channels=num_image_channels)

        self.best_test = np.Inf
        self.criterion = Loss(self.opt.num_rot_bins)


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch

        pred_front, pred_rot_bins, pred_t, emb = self.estimator(img, points, choose, idx)

        # torch.cuda.empty_cache()

        loss, front_loss, rot_loss, t_loss = self.criterion(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, t)

        # self.log_dict({'train_dis':dis, 'loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('front loss', front_loss, on_step=True, on_epoch=True, logger=True)
        self.log('rot loss', rot_loss, on_step=True, on_epoch=True, logger=True)
        self.log('t loss', t_loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.test_loss = 0.0

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch
        pred_front, pred_rot_bins, pred_t, emb = self.estimator(img, points, choose, idx)

        loss, front_loss, rot_loss, t_loss = self.criterion(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, t)

        # if self.opt.refine_start:
        #     for ite in range(0, self.opt.iteration):
        #         pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)
        #         loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = \
        #             self.criterion_refine(pred_front, pred_rot_bins, pred_t, new_front_r, 
        #                                 new_rot_bins, new_front_orig, new_t, idx, new_points)       
        
        # visualize
        if (batch_idx == 0 and self.opt.visualize):

            model_points = model_points[0]
            points = points[0]
            pred_front = pred_front[0]
            pred_rot_bins = pred_rot_bins[0]
            pred_t = pred_t[0]
            emb = emb[0]
            front_r = front_r[0]
            rot_bins = rot_bins[0]
            front_orig = front_orig[0]
            t = t[0]


            #projected depth image
            projected_vis = self.visualize_pointcloud(points)
            projected_color = np.zeros((projected_vis.shape))
            projected_color[:,:,2] = 100

            #visualize translation votes
            gt_t_vis = self.visualize_pointcloud(t)
            pred_t_vis = self.visualize_pointcloud(pred_t)

            t_vis = np.concatenate((gt_t_vis, pred_t_vis, projected_vis), axis=1)
            gt_t_color = np.zeros((gt_t_vis.shape))
            gt_t_color[:,:,0] = 200
            pred_t_color = np.zeros((pred_t_vis.shape))
            pred_t_color[:,:,1] = 200
            t_colors = np.concatenate((gt_t_color, pred_t_color, projected_color), axis=1)
            self.logger.experiment.add_mesh('t_vis ' + str(self.current_epoch), vertices=t_vis, colors=t_colors)

            #visualize front votes
            gt_front_vis = self.visualize_fronts(front_r, t)
            pred_front_vis = self.visualize_pointcloud(pred_front)

            front_vis = np.concatenate((gt_front_vis, pred_front_vis, projected_vis), axis=1)
            gt_front_color = np.zeros((gt_front_vis.shape))
            gt_front_color[:,:,0] = 200
            pred_front_color = np.zeros((pred_front_vis.shape))
            pred_front_color[:,:,1] = 200
            front_colors = np.concatenate((gt_front_color, pred_front_color, projected_color), axis=1)
            self.logger.experiment.add_mesh('front_vis ' + str(self.current_epoch), vertices=front_vis, colors=front_colors)

            #visualize predicted model
            gt_vis, pred_vis = self.get_vis_models(model_points, points, pred_front, pred_rot_bins, pred_t, 
                    emb, front_r, rot_bins, front_orig, t)

            vis = np.concatenate((gt_vis, pred_vis), axis=1)
            gt_color = np.zeros((gt_vis.shape))
            gt_color[:,:,0] = 200
            pred_color = np.zeros((pred_vis.shape))
            pred_color[:,:,1] = 200
            colors = np.concatenate((gt_color, pred_color), axis=1)
            # summary = mesh_summary.op('point_cloud', vertices=gt_vis)
            self.logger.experiment.add_mesh('models_vis ' + str(self.current_epoch) , vertices=vis, colors=colors)    

        self.test_loss += loss.item()
        self.log('val_loss', loss, logger=True)
        self.log('val front loss', front_loss, logger=True)
        self.log('val rot loss', rot_loss, logger=True)
        self.log('val t loss', t_loss, logger=True)
        # logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
        return loss.item()

    def validation_epoch_end(self, outputs):
        torch.save(self.estimator.state_dict(), '{0}/pose_model_current.pth'.format(self.opt.outf))

        test_loss = np.average(np.array(outputs))
        if test_loss <= self.best_test:
            self.best_test = test_loss
            torch.save(self.estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, self.current_epoch, test_loss))
        print("best_test: ", self.best_test) 


    def configure_optimizers(self):
        optimizer = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)
        
        return optimizer


    """

        visualize_pointcloud(t, "{0}_gt_t".format(i))
        visualize_pointcloud(pred_t, "{0}_pred_t".format(i))

        visualize_fronts(front_r, t, "{0}_gt_front".format(i))
        visualize_fronts(pred_front - points, points, "{0}_pred_front".format(i))

        print(pred_front.shape, pred_rot_bins.shape, pred_t.shape)

        mean_front = torch.mean(pred_front, 1)
        mean_t = torch.mean(pred_t, 1)

        #batch size = 1 always
        my_front, front_labels = ms.fit(pred_front.squeeze(0))
        my_t, t_labels = ms.fit(pred_t.squeeze(0))

        #switch to vector from center of object
        my_front -= my_t

        gt_front = front_r.squeeze()
        gt_theta = torch.argmax(rot_bins) / opt.num_rot_bins * 2 * np.pi
        gt_t = t.squeeze()

        #pts_gt = get_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        #pts_pred = get_points(model_points, front_orig, my_front, my_theta, my_t)

        #dist = dist2(pts_gt, pts_pred)
        #dists.append(np.mean(np.min(dist, axis=1)))

        visualize_points(model_points, front_orig, gt_front, gt_theta, gt_t, "{0}_gt".format(i))
        visualize_points(model_points, front_orig, my_front, gt_theta, my_t, "{0}_pred".format(i))
        visualize_pointcloud(points, "{0}_projected_depth".format(i))
    
    """

    def get_vis_models(self, model_points, points, pred_front, pred_rot_bins, pred_t, 
                    emb, front_r, rot_bins, front_orig, t):

        #vote clustering
        radius = 0.08
        ms = MeanShiftTorch(bandwidth=radius)

        
        my_front, front_labels = ms.fit(pred_front)
        my_t, t_labels = ms.fit(pred_t)

        #theta vote clustering
        theta_radius = 15 / 180 * np.pi #15 degrees
        ms_theta = MeanShiftTorch(bandwidth=theta_radius)

        pred_thetas = (torch.argmax(pred_rot_bins, dim=1) / self.opt.num_rot_bins * 2 * np.pi).unsqueeze(-1)
        my_theta, theta_labels = ms_theta.fit(pred_thetas)

        my_theta = my_theta.squeeze()

        #switch to vector from center of object
        my_front -= my_t
        
        gt_front = front_r.squeeze()
        gt_theta = torch.argmax(rot_bins) / self.opt.num_rot_bins * 2 * np.pi
        gt_t = t.squeeze()
        
        gt_vis = self.visualize_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        pred_vis = self.visualize_points(model_points, front_orig, my_front, my_theta, my_t)

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