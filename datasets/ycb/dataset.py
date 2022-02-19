import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import rotation_matrix_from_vectors_procedure, axis_angle_of_rotation_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
from datetime import datetime
#import open3d as o3d
#import cv2

def standardize_image_size(target_image_size, rmin, rmax, cmin, cmax, image_height, image_width):
    height, width = rmax - rmin, cmax - cmin

    if height > target_image_size:
        diff = height - target_image_size
        rmin += int(diff / 2)
        rmax -= int((diff + 1) / 2)
    
    elif height < target_image_size:
        diff = target_image_size - height
        if rmin - int(diff / 2) < 0:
            rmax += diff
        elif rmax + int((diff + 1) / 2) >= image_height:
            rmin -= diff
        else:
            rmin -= int(diff / 2)
            rmax += int((diff + 1) / 2)
    
    if width > target_image_size:
        diff = width - target_image_size
        cmin += int(diff / 2)
        cmax -= int((diff + 1) / 2)
    
    elif width < target_image_size:
        diff = target_image_size - width
        if cmin - int(diff / 2) < 0:
            cmax += diff
        elif cmax + int((diff + 1) / 2) >= image_width:
            cmin -= diff
        else:
            cmin -= int(diff / 2)
            cmax += int((diff + 1) / 2)
    
    return rmin, rmax, cmin, cmax

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, num_rot_bins, image_size, append_depth_to_image=False, add_front_aug=False):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.image_size = image_size

        print("root", self.root)

        self.append_depth_to_image = append_depth_to_image

        #doesn't really make sense the way this is implemented
        self.add_front_aug = add_front_aug

        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)

        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('datasets/ycb/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        
        #front vector for objects
        self.frontd = {}

        #symmetries for objects
        self.symmd = {}

        supported_symm_types = {'radial'}

        self.symmetry_obj_idx = [12, 15, 18, 19, 20]

        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line.rstrip().split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            input_file = open('{0}/models/{1}/front.xyz'.format(self.root, class_input[:-1]))
            self.frontd[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line or len(input_line) <= 1:
                    break
                input_line = input_line.rstrip().split(' ')
                self.frontd[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.frontd[class_id] = np.array(self.frontd[class_id])
            input_file.close()

            #since class_is 1-indexed but self.symmetry_obj_idx is 0-indexed...
            if class_id - 1 in self.symmetry_obj_idx:
                input_file = open('{0}/models/{1}/symm.txt'.format(self.root, class_input[:-1]))
                self.symmd[class_id] = []
                while 1:
                    symm_type = input_file.readline().rstrip()
                    if not symm_type or len(symm_type) == 0:
                        break
                    if symm_type not in supported_symm_types:
                        raise Exception("Invalid symm_type " + symm_type)
                    number_of_symms = input_file.readline().rstrip()
                    self.symmd[class_id].append((symm_type, number_of_symms))
                input_file.close()
            else:
                self.symmd[class_id] = []

            class_id += 1

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2
        self.num_rot_bins = num_rot_bins
        self.rot_bin_width = 2. * np.pi / num_rot_bins


        #preload items
        num_preload = 0

        self.preloaded_img = {}
        self.preloaded_depth = {}
        self.preloaded_label = {}
        self.preloaded_meta = {}

        #don't have enough room for test
        if mode == 'train':
            for index in range(num_preload):
                color_key = '{0}/{1}-color.png'.format(self.root, self.list[index])
                depth_key = '{0}/{1}-depth.png'.format(self.root, self.list[index])
                label_key = '{0}/{1}-label.png'.format(self.root, self.list[index])
                meta_key = '{0}/{1}-meta.mat'.format(self.root, self.list[index])
                
                #keep our io's as close as possible
                color = Image.open(color_key)
                depth = Image.open(depth_key)
                label = Image.open(label_key)
                self.preloaded_meta[meta_key] = scio.loadmat(meta_key)

                keep = color.copy()
                color.close()
                self.preloaded_img[color_key] = keep

                keep = np.copy(np.array(depth))
                depth.close()
                self.preloaded_depth[depth_key] = keep

                keep = np.copy(np.array(label))
                label.close()
                self.preloaded_label[label_key] = keep
                

                if index % 1000 == 0:
                    print(datetime.now())
                    print("pre-loaded {0} data".format(index))

    

    def __getitem__(self, index):

        color = '{0}/{1}-color.png'.format(self.root, self.list[index])
        depth = '{0}/{1}-depth.png'.format(self.root, self.list[index])
        label = '{0}/{1}-label.png'.format(self.root, self.list[index])
        meta = '{0}/{1}-meta.mat'.format(self.root, self.list[index])

        #print(color)

        if color in self.preloaded_img:
            img = self.preloaded_img[color]
            depth = self.preloaded_depth[depth]
            label = self.preloaded_label[label]
            meta = self.preloaded_meta[meta]
        else:
            img = Image.open(color)
            depth = np.array(Image.open(depth))
            label = np.array(Image.open(label))
            meta = scio.loadmat(meta)

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        #used later to add background to synthetic images
        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        #cv2.imwrite("label_orig.png", label)

        #going to use mask_front to add synthetic on top of both img and depth
        add_front = False
        if self.add_noise and self.add_front_aug:
            for k in range(5):
                seed = random.choice(self.syn)
                front_rgb = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front_depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, seed)))
                front_rgb = np.transpose(front_rgb, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk
                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        #select an object and 
        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) <= self.minimum_num_pt:
                continue

            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            h, w, _= np.array(img).shape
            rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)
            rmin, rmax, cmin, cmax = standardize_image_size(self.image_size, rmin, rmax, cmin, cmax, h, w)

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) == 0:
                continue

            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
                break
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
                break
        
        #cv2.imwrite("mask.png", mask.astype(np.uint16) * 65535)

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])


        #calculating our histogram rotation representation
        
        #right now, we are only dealing with one "front" axis
        front = self.frontd[obj[idx]][0]

        front = front / np.linalg.norm(front) * 0.1

        #symmetries
        symm = self.symmd[obj[idx]]

        #calculate front axis in GT pose
        front_r = target_r @ front

        rot_bins = np.zeros(self.num_rot_bins).astype(np.float32)

        #find rotation matrix that goes from front -> front_r
        Rf = rotation_matrix_from_vectors_procedure(front, front_r)

        #find residual rotation
        R_around_front = target_r @ Rf.T

        #axis will be the same as R_around_front
        axis, angle = axis_angle_of_rotation_matrix(R_around_front)

        #negate the axis and the angle
        if np.abs(np.linalg.norm(axis + front_r)) < 0.001:
            axis = -axis
            angle = -angle

        if angle < 0:
            angle += np.pi * 2
        if angle > np.pi * 2:
            angle -= np.pi * 2

        assert (angle >= 0 and angle <= np.pi * 2)
        
        angle_bin = int(angle / 2 / np.pi * self.num_rot_bins)

        #calculate other peaks based on size of symm
        if len(symm) > 0:
            symm_type, num_symm = symm[0]
            if num_symm == 'inf':
                rot_bins[:] = 1.
            else:
                num_symm = int(num_symm)

                symm_bins = (np.arange(0, 2 * np.pi, 2 * np.pi / num_symm) / self.rot_bin_width).astype(np.int)
                symm_bins += angle_bin
                symm_bins = np.mod(symm_bins, self.num_rot_bins)
                rot_bins[symm_bins] = 1.
        
        else:
            rot_bins[angle_bin] = 1. 

        rot_bins /= np.linalg.norm(rot_bins, ord=1)
        
        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)

        img = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        depth_masked = depth[rmin:rmax, cmin:cmax]
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax]
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax]

        #cv2.imwrite("test_no_back.png", np.transpose(img, (1, 2, 0)))
        #cv2.imwrite("depth_no_back.png", depth_masked.astype(np.uint16))

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back_depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, seed)))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            back_depth = back_depth[rmin:rmax, cmin:cmax]

            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img * ~(mask_back[rmin:rmax, cmin:cmax])
            depth_masked = back_depth * mask_back[rmin:rmax, cmin:cmax] + depth_masked * ~(mask_back[rmin:rmax, cmin:cmax])
        else:
            img_masked = img

        #cv2.imwrite("test_no_front.png", np.transpose(img_masked, (1, 2, 0)))
        #cv2.imwrite("depth_no_front.png", depth_masked.astype(np.uint16))

        if self.add_noise and add_front:
            #print("adding front")

            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front_rgb[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])
            depth_masked = depth_masked * mask_front[rmin:rmax, cmin:cmax] + front_depth[rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))

        if self.append_depth_to_image:
            cam_scale = meta['factor_depth'][0][0]
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy

            pt0 = np.expand_dims(pt0, -1)
            pt1 = np.expand_dims(pt1, -1)
            pt2 = np.expand_dims(pt2, -1)

            cloud = np.concatenate((pt0, pt1, pt2), axis=2)
            if self.add_noise:
                cloud = np.add(cloud, add_t)

            cloud = np.transpose(cloud, (2, 0, 1))

            #cv2.imwrite("label.png", label)
            #cv2.imwrite("test.png", np.transpose(img_masked, (1, 2, 0)))
            #cv2.imwrite("depth.png", depth_masked.astype(np.uint16))

            #test_pcld = o3d.geometry.PointCloud()
            #test_pcld.points = o3d.utility.Vector3dVector(np.transpose(cloud, (1, 2, 0)).reshape((-1, 3)))
            #test_pcld.colors = o3d.utility.Vector3dVector(np.transpose(img_masked, (1, 2, 0)).reshape((-1, 3)))
            #o3d.io.write_point_cloud("projected.ply", test_pcld)

            cloud = torch.from_numpy(cloud.astype(np.float32))

            img_masked = torch.cat((img_masked, cloud), axis=0)

        depth_masked = depth_masked.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap_masked.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap_masked.flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        model_points = self.cld[obj[idx]]
        select_list = np.random.choice(len(model_points), self.num_pt_mesh_small, replace=False) # without replacement, so that it won't choice duplicate points
        model_points = model_points[select_list]

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               img_masked, \
               torch.from_numpy(front_r.astype(np.float32)), \
               torch.from_numpy(rot_bins.astype(np.float32)), \
               torch.from_numpy(front.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([int(obj[idx]) - 1])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
    


