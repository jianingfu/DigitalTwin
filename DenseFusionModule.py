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
import plotly.express as px
from datetime import datetime
from tensorboard.plugins.mesh import summary as mesh_summary
from lib.transformations import rotation_matrix_from_vectors_procedure, rotation_matrix_of_axis_angle


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

        loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = self.criterion(pred_front, 
                        pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, 
                        model_points, points, self.opt.w, self.opt.refine_start)


        if self.opt.profile:
            print("finished loss {0} {1}".format(batch_idx, datetime.now()))
        
        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)
                loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = \
                    self.criterion_refine(pred_front, pred_rot_bins, pred_t, new_front_r, 
                    new_rot_bins, new_front_orig, new_t, idx, new_points)
                loss.backward()

        # self.log_dict({'train_dis':dis, 'loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.opt.profile:
                    print("finished training sample {0} {1}".format(batch_idx, datetime.now()))
        return loss

    def on_validation_epoch_start(self):
        self.test_loss = 0.0

    # default check_val_every_n_epoch=1 by lightning
    def validation_step(self, batch, batch_idx):
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = batch
        pred_front, pred_rot_bins, pred_t, pred_c, emb = self.estimator(img, points, choose, idx)
        loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = self.criterion(pred_front, 
                pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, 
                model_points, points, self.opt.w, self.opt.refine_start)

        if self.opt.refine_start:
            for ite in range(0, self.opt.iteration):
                pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)
                loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = \
                    self.criterion_refine(pred_front, pred_rot_bins, pred_t, new_front_r, 
                                        new_rot_bins, new_front_orig, new_t, idx, new_points)       
        
        # visualize TODO: check pred_r
        if (batch_idx == 0 and self.opt.visualize):
            gt_front_vis, pred_fronts_estimator_vis, selected_pred_front_vis, gt_vis, \
                pred_vis, projected_depth_vi = self.get_vis_points(points, idx, model_points, 
                        pred_front, pred_rot_bins, pred_t, pred_c, emb, front_r, rot_bins, front_orig, 
                        t, new_points)

            summary = mesh_summary.op('point_cloud', vertices=gt_vis)
            self.logger.experiment.add_mesh(summary)    

        self.test_loss += loss.item()
        val_loss = loss.item()
        self.log('val_loss', val_loss, logger=True)
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
            self.opt.sym_list = self.trainer.datamodule.sym_list
            self.opt.num_points_mesh = self.trainer.datamodule.num_points_mesh
            self.criterion = Loss(self.opt.num_points_mesh, self.opt.sym_list)
            self.criterion_refine = Loss_refine(self.opt.num_points_mesh, self.opt.sym_list)
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

    def get_vis_points(self, idx, model_points, points, pred_front, pred_rot_bins, pred_t, pred_c, 
                    emb, front_r, rot_bins, front_orig, t, new_points):
         #immediately shift pred_front to model coordinates
        pred_front = pred_front - (pred_t + points)

        bs, num_p, _ = pred_c.shape

        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)

        gt_front_vis = self.visualize_fronts(front_r, t)
        pred_fronts_estimator_vis = self.visualize_fronts(pred_front, pred_t + points)

        #which_max -> bs
        #pred_t -> bs * num_p * 3
        #points -> bs * num_p * 3

        which_max_3 = which_max.view(bs, 1, 1).repeat(1, 1, 3)
        which_max_rot_bins = which_max.view(bs, 1, 1).repeat(1, 1, self.opt.num_rot_bins)

        #best_c_pred_t -> bs * 1 * 3
        best_c_pred_t = torch.gather(pred_t, 1, which_max_3) + torch.gather(points, 1, which_max_3)

        #we need to calculate the actual transformation that our rotation rep. represents

        best_c_pred_front = torch.gather(pred_front, 1, which_max_3).squeeze(1)
        best_c_rot_bins = torch.gather(pred_rot_bins, 1, which_max_rot_bins).squeeze(1)

        #get the angle in radians based on highest histogram bin
        #angle -> bs * 1
        angle = (torch.argmax(best_c_rot_bins, axis=1) / best_c_rot_bins.shape[1] * 2 * np.pi)

        my_front = best_c_pred_front.squeeze()
        my_theta = angle.squeeze()
        my_t = best_c_pred_t.squeeze()

        gt_front = front_r.squeeze()
        gt_theta = torch.argmax(rot_bins) / self.opt.num_rot_bins * 2 * np.pi
        gt_t = t.squeeze()

        if self.opt.refine_model != "":
            for ite in range(0, self.opt.iteration):
                pred_front, pred_rot_bins, pred_t = self.refiner(new_points, emb, idx)

                #shift to model_coordinates
                pred_front = pred_front - pred_t

                my_front += pred_front.squeeze()

                best_c_rot_bins = pred_rot_bins.squeeze(1)
                angle = (torch.argmax(best_c_rot_bins, axis=1) / best_c_rot_bins.shape[1] * 2 * np.pi)
                my_theta += angle.squeeze()

                my_t += pred_t.squeeze()

                loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = self.criterion_refine(pred_front, pred_rot_bins, pred_t, new_front_r, new_rot_bins, new_front_orig, new_t, idx, new_points)


        selected_pred_front_vis = self.visualize_fronts(my_front, my_t)

        #pts_gt = get_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        #pts_pred = get_points(model_points, front_orig, my_front, my_theta, my_t)

        #dist = dist2(pts_gt, pts_pred)
        #dists.append(np.mean(np.min(dist, axis=1)))

        gt_vis = self.visualize_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        pred_vis = self.visualize_points(model_points, front_orig, my_front, my_theta, my_t)
        projected_depth_vis = self.visualize_pointcloud(points)

        return [gt_front_vis, pred_fronts_estimator_vis, selected_pred_front_vis, 
                gt_vis, pred_vis, projected_depth_vis]

    def visualize_fronts(fronts, t):
        fronts = fronts.cpu().detach().numpy()
        fronts = fronts.reshape((-1, 3))

        t = t.cpu().detach().numpy()
        t = t.reshape((-1, 3))

        front_points = fronts + t
        return front_points

    def visualize_points(model_points, front_orig, front, angle, t):

        model_points = model_points.cpu().detach().numpy()
        front_orig = front_orig.cpu().detach().numpy().squeeze()
        front = front.cpu().detach().numpy()
        angle = angle.cpu().detach().numpy()
        t = t.cpu().detach().numpy()

        Rf = rotation_matrix_from_vectors_procedure(front_orig, front)

        R_axis = rotation_matrix_of_axis_angle(front, angle)

        R_tot = (R_axis @ Rf)

        pts = (model_points @ R_tot.T + t).squeeze()
        return pts

    def visualize_pointcloud(points):

        points = points.cpu().detach().numpy()

        points = points.reshape((-1, 3))
        return points