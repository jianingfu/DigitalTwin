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

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts


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

    dists = []

    #vote clustering
    radius = 0.08
    ms = MeanShiftTorch(bandwidth=radius)

    colors = [(96, 60, 20), (156, 39, 6), (212, 91, 18), (243, 188, 46), (95, 84, 38)]

    start_sample = 0

    for sample_idx in range(start_sample, len(test_dataset)):

        print("processing sample", sample_idx)

        img_filename = '{0}/{1}-color.png'.format(test_dataset.root, test_dataset.list[sample_idx])
        color_img = cv2.imread(img_filename)

        data_objs = test_dataset.get_all_objects(sample_idx)

        for obj_idx, data in enumerate(data_objs):

            torch.cuda.empty_cache()

            data, intrinsics = data
            cam_fx, cam_fy, cam_cx, cam_cy = intrinsics

            data = [torch.unsqueeze(d, 0) for d in data]

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

            pts = get_model_points(model_points, front_orig, my_front, my_theta, my_t)
            projected_pts = project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy)

            r, g, b = colors[obj_idx % len(colors)]

            for (x, y) in projected_pts:
                color_img = cv2.circle(color_img, (int(x), int(y)), radius=1, color=(b,g,r), thickness=-1)



        output_filename = '{0}/{1}.png'.format(opt.output, sample_idx)
        cv2.imwrite(output_filename, color_img)




    #print(np.mean(dists))


if __name__ == '__main__':
    main()
