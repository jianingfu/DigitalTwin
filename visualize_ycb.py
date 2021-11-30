# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

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

from lib.transformations import rotation_matrix_from_vectors_procedure, rotation_matrix_of_axis_angle

import open3d as o3d

def visualize_points(model_points, front_orig, front, angle, t, label):

    model_points = model_points.cpu().detach().numpy()
    front_orig = front_orig.cpu().detach().numpy().squeeze()
    front = front.cpu().detach().numpy()
    angle = angle.cpu().detach().numpy()
    t = t.cpu().detach().numpy()

    Rf = rotation_matrix_from_vectors_procedure(front_orig, front)

    R_axis = rotation_matrix_of_axis_angle(front, angle)

    R_tot = (R_axis @ Rf)

    print("our rot rep", R_tot)

    pts = (model_points @ R_tot.T + t).squeeze()

    pcld = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(pts)

    pcld.points = pts

    o3d.io.write_point_cloud(label + ".ply", pcld)

def visualize_pointcloud(points, label):

    points = points.cpu().detach().numpy()

    points = points.reshape((-1, 3))

    pcld = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)

    pcld.points = points

    o3d.io.write_point_cloud(label + ".ply", pcld)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
    parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
    parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
    parser.add_argument('--model', type=str, default = '',  help='PoseNet model')
    parser.add_argument('--refine_model', type=str, default = '',  help='PoseRefineNet model')
    parser.add_argument('--w', default=0.015, help='regularize confidence')
    parser.add_argument('--w_rate', default=0.3, help='regularize confidence refiner decay')

    parser.add_argument('--num_rot_bins', type=int, default = 18, help='number of bins discretizing the rotation around front')
    parser.add_argument('--num_visualized', type=int, default = 5, help='number of training samples to visualize')
    opt = parser.parse_args()

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
    else:
        print('Unknown dataset')
        return


    if opt.model != '':
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot_bins = opt.num_rot_bins)
        estimator.cuda()
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.model)))

    if opt.refine_model != '':
        refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot_bins = opt.num_rot_bins)
        refiner.cuda()
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.refine_model)))

        #matching train
        opt.w *= opt.w_rate

    if not opt.model and not opt.refine_model:
        raise Exception("this is visualizer code, pls pass in a model lol")

    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_model != '', opt.num_rot_bins)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_model != '', opt.num_rot_bins)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\nnumber of sample points on mesh: {1}\nsymmetry object list: {2}'.format(len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_rot_bins)
    criterion_refine = Loss_refine(opt.num_rot_bins)

    estimator.eval()

    if opt.refine_model != "":
        refiner.eval()

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = data
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(front_r).cuda(), \
                                                            Variable(rot_bins).cuda(), \
                                                            Variable(front_orig).cuda(), \
                                                            Variable(t).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()
        pred_front, pred_rot_bins, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        loss, new_points, new_rot_bins, new_t, _, _, _, _, _ = criterion(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, opt.w, opt.refine_model != "")

        print("here!", pred_front.shape, pred_rot_bins.shape, pred_t.shape, pred_c.shape)

        bs, num_p, _ = pred_c.shape

        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)

        my_front = pred_front.view(bs * num_p, 3)[which_max[0]]
        my_theta = torch.argmax(pred_rot_bins.view(bs*num_p, opt.num_rot_bins)[which_max[0]]) / opt.num_rot_bins * 2 * np.pi
        my_t = (pred_t.contiguous().view(bs*num_p, 1, 3)[which_max[0]] + points.contiguous().view(bs * num_p, 1, 3)[which_max[0]]).squeeze()

        gt_front = front_r.squeeze()
        gt_theta = torch.argmax(rot_bins) / opt.num_rot_bins * 2 * np.pi
        gt_t = t.squeeze()

        print("after first pass, shapes.")
        print(my_front.shape, my_theta.shape, my_t.shape)
        print(gt_front.shape, gt_theta.shape, gt_t.shape)

        if opt.refine_model != "":
            for ite in range(0, opt.iteration):
                pred_front, pred_rot_bins, pred_t = refiner(new_points, emb, idx)
                print("here1", pred_front.shape, pred_rot_bins.shape, pred_t.shape)
                loss, new_points, new_rot_bins, new_t = criterion_refine(pred_front, pred_rot_bins, pred_t, front_r, new_rot_bins, front_orig, new_t, idx, new_points)



        visualize_points(model_points, front_orig, gt_front, gt_theta, gt_t, "{0}_gt".format(i))
        visualize_points(model_points, front_orig, my_front, my_theta, my_t, "{0}_pred".format(i))
        visualize_pointcloud(points, "{0}_projected_depth".format(i))

        if i >= opt.num_visualized:
            print("finished visualizing!")
            exit()


if __name__ == '__main__':
    main()