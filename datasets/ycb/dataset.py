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
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix, rotation_matrix_from_vectors_procedure, axis_angle_of_rotation_matrix, rotation_matrix_of_axis_angle
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
from datetime import datetime

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine, num_rot_bins, perform_profiling=False):
        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root

        print("root", self.root)

        self.perform_profiling = perform_profiling

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
        self.refine = refine
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

        if self.perform_profiling:
            print("entering get item {0} {1}".format(index, datetime.now()))

        color = '{0}/{1}-color.png'.format(self.root, self.list[index])
        depth = '{0}/{1}-depth.png'.format(self.root, self.list[index])
        label = '{0}/{1}-label.png'.format(self.root, self.list[index])
        meta = '{0}/{1}-meta.mat'.format(self.root, self.list[index])

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

        if self.perform_profiling:
            print("finished loading from disk {0} {1}".format(index, datetime.now()))

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

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
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

        if self.perform_profiling:
            print("finished add front aug {0} {1}".format(index, datetime.now()))

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.perform_profiling:
            print("finished selecting object {0} {1}".format(index, datetime.now()))

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)

        if self.perform_profiling:
            print("finished get_bbox {0} {1}".format(index, datetime.now()))

        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.perform_profiling:
            print("finished first img_masked {0} {1}".format(index, datetime.now()))    

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        if self.perform_profiling:
            print("finished second img_masked {0} {1}".format(index, datetime.now()))

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        if self.perform_profiling:
            print("finished third img_masked {0} {1}".format(index, datetime.now()))

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        if self.perform_profiling:
            print("finished doing densefusion's stuff {0} {1}".format(index, datetime.now()))

        #calculating our histogram rotation representation
        
        #right now, we are only dealing with one "front" axis
        front = self.frontd[obj[idx]][0]

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

        if self.perform_profiling:
            print("finished my rotation stuff {0} {1}".format(index, datetime.now()))

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        if self.perform_profiling:
            print("finished sampling points from roi {0} {1}".format(index, datetime.now()))
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # fw = open('temp/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        if self.perform_profiling:
            print("finished computations {0} {1}".format(index, datetime.now()))
        
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(front_r.astype(np.float32)), \
               torch.from_numpy(rot_bins.astype(np.float32)), \
               torch.from_numpy(front.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([int(obj[idx]) - 1])

    def get_item_debug(self, index):
        color = '{0}/{1}-color.png'.format(self.root, self.list[index])
        depth = '{0}/{1}-depth.png'.format(self.root, self.list[index])
        label = '{0}/{1}-label.png'.format(self.root, self.list[index])
        meta = '{0}/{1}-meta.mat'.format(self.root, self.list[index])

        print("loading sample", self.list[index])

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

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
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

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        if self.list[index][:8] == 'data_syn':
            img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        #calculating our histogram rotation representation
        
        #right now, we are only dealing with one "front" axis
        front = self.frontd[obj[idx]][0]

        #symmetries
        symm = self.symmd[obj[idx]]

        #calculate front axis in GT pose
        front_r = target_r @ front

        rot_bins = np.zeros(self.num_rot_bins).astype(np.float32)

        print("FRONT 1", front)

        #find rotation matrix that goes from front -> front_r
        Rf = rotation_matrix_from_vectors_procedure(front, front_r)

        print("original calculated rf")
        print(Rf)

        print("front_r from gt", front_r)
        print("Rf @ front", Rf @ front)

        #find residual rotation
        R_around_front = target_r @ Rf.T

        #axis will be the same as R_around_front
        axis, angle = axis_angle_of_rotation_matrix(R_around_front)

        #negate the axis and the angle
        if np.abs(np.linalg.norm(axis + front_r)) < 0.001:
            print("axis flipped")
            axis = -axis
            angle = -angle

        print("these should be the same")
        print(front_r, axis)

        print("test here, target_r, then our calculated")

        R_around_front_recovered = rotation_matrix_of_axis_angle(axis, angle)

        print(target_r)
        print(R_around_front @ Rf)
        print(R_around_front_recovered @ Rf)

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

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # fw = open('temp/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)
        
        return cloud.astype(np.float32), \
               choose.astype(np.int32), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))).numpy(), \
               front_r.astype(np.float32), \
               rot_bins.astype(np.float32), \
               angle.astype(np.float32), \
               Rf.astype(np.float32), \
               front.astype(np.float32), \
               target_r.astype(np.float32), \
               target_t.astype(np.float32), \
               model_points.astype(np.float32), \
               [int(obj[idx]) - 1]

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
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
    

