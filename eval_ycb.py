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
from lib.network import PoseNet
from lib.loss import Loss
from lib.meanshift_pytorch import MeanShiftTorch
from PIL import Image
import cv2

from lib.transformations import rotation_matrix_from_vectors_procedure, rotation_matrix_of_axis_angle

import open3d as o3d

from knn_cuda import KNN

from collections import defaultdict

def dist2(x, y):
    nx, dimx = x.shape
    ny, dimy = y.shape
    assert(dimx == dimy)

    return (np.ones((ny, 1)) * np.sum((x**2).T, axis=0)).T + \
        np.ones((nx, 1)) * np.sum((y**2).T, axis=0) - \
        2 * np.inner(x, y)

def get_model_points(model_points, front_orig, front, angle, t):

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

def visualize_pointcloud(points, label):

    points = points.cpu().detach().numpy()

    points = points.reshape((-1, 3))

    pcld = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)

    pcld.points = points

    o3d.io.write_point_cloud(label + ".ply", pcld)

def cal_auc(add_dis):
        max_dis = 0.1
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default = 'datasets/ycb/YCB_Video_Dataset', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
    parser.add_argument('--model', type=str, default = '',  help='PoseNet model')
    parser.add_argument('--output', type=str, default = 'visualization', help='where to dump output')

    parser.add_argument('--num_rot_bins', type=int, default = 36, help='number of bins discretizing the rotation around front')
    parser.add_argument('--num_visualized', type=int, default = 5, help='number of training samples to visualize')
    parser.add_argument('--image_size', type=int, default=300, help="square side length of cropped image")
    parser.add_argument('--append_depth_to_image', action="store_true", default=False, help='put XYZ of pixel into image')

    opt = parser.parse_args()

    if not os.path.isdir(opt.output):
        os.mkdir(opt.output)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
    else:
        print('ONLY YCB')
        exit(-1)

    
    if opt.model != '':
        num_image_channels = 3
        if opt.append_depth_to_image:
            num_image_channels += 3

        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot_bins = opt.num_rot_bins, num_image_channels=num_image_channels)
        estimator.cuda()
        #estimator = nn.DataParallel(estimator)
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.model)))
    else:
        raise Exception("this is visualizer code, pls pass in a model lol")

    test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.num_rot_bins, opt.image_size, append_depth_to_image=opt.append_depth_to_image)
    
    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\nnumber of sample points on mesh: {1}\nsymmetry object list: {2}'.format(len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_rot_bins)

    estimator.eval()

    adds = defaultdict(list)

    knn = KNN(k=1, transpose_mode=True)

    with torch.no_grad():

        for sample_idx in range(len(test_dataset)):

            torch.cuda.empty_cache()

            print("processing sample", sample_idx)

            data_objs = test_dataset.get_all_objects(sample_idx)

            for obj_idx, data in enumerate(data_objs):

                torch.cuda.empty_cache()

                data, intrinsics = data
                cam_fx, cam_fy, cam_cx, cam_cy = intrinsics

                data = [torch.unsqueeze(d, 0) for d in data]

                points, choose, img, front_r, rot_bins, front_orig, t, model_points, target, idx = data
                

                points, choose, img, front_r, rot_bins, front_orig, t, model_points, target, idx = Variable(points).cuda(), \
                                                                    Variable(choose).cuda(), \
                                                                    Variable(img).cuda(), \
                                                                    Variable(front_r).cuda(), \
                                                                    Variable(rot_bins).cuda(), \
                                                                    Variable(front_orig).cuda(), \
                                                                    Variable(t).cuda(), \
                                                                    Variable(model_points).cuda(), \
                                                                    Variable(target).cuda(), \
                                                                    Variable(idx).cuda()

                #pred_front and pred_t are now absolute positions for front and center keypoint
                pred_front, pred_rot_bins, pred_t, emb = estimator(img, points, choose, idx)
                loss = criterion(pred_front, pred_rot_bins, pred_t, front_r, rot_bins, t)
                
                #vote clustering
                radius = 0.08
                ms = MeanShiftTorch(bandwidth=radius)

                #batch size = 1 always
                my_front, front_labels = ms.fit(pred_front.squeeze(0))
                my_t, t_labels = ms.fit(pred_t.squeeze(0))

                #theta vote clustering
                theta_radius = 15 / 180 * np.pi #15 degrees
                ms_theta = MeanShiftTorch(bandwidth=theta_radius)

                pred_thetas = (torch.argmax(pred_rot_bins.squeeze(0), dim=1) / opt.num_rot_bins * 2 * np.pi).unsqueeze(-1)

                my_theta, theta_labels = ms_theta.fit(pred_thetas)

                #switch to vector from center of object
                my_front -= my_t

                gt_theta = torch.argmax(rot_bins) / opt.num_rot_bins * 2 * np.pi

                pts = get_model_points(model_points, front_orig, my_front, gt_theta, my_t)
                pts = torch.from_numpy(pts.astype(np.float32)).unsqueeze(0).cuda()

                dists, inds = knn(pts, target)
                dist = torch.mean(dists).detach().cpu().item()
                idx = idx.detach().cpu().item()

                print("idx, adds", idx, dist)

                adds[idx].append(dist)

                for d in data:
                    del d

                del pts
                del pred_front
                del rot_bins
                del pred_t
                del emb
                del loss
                del my_front
                del my_t
                del ms
                del ms_theta
                del my_theta
                del gt_theta


        adds_aucs = {}

        for idx, dists in adds.items():
            adds_aucs[idx] = cal_auc(dists)

        print(adds_aucs)



if __name__ == '__main__':
    main()
