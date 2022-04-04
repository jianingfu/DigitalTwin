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
from lib.transformations import rotation_matrix_of_axis_angle
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import open3d as o3d
from lib.depth_utils import compute_normals
import cv2

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

def get_random_rotation_around_symmetry_axis(axis, symm_type, num_symm):
    if symm_type == "radial":
        if num_symm == "inf":
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            angles = np.arange(0, 2 * np.pi, 2 * np.pi / int(num_symm))
            angle = np.random.choice(angles)
        return rotation_matrix_of_axis_angle(axis, angle).squeeze()
    else:
        raise Exception("Invalid symm_type " + symm_type)
    

class PoseDataset(data.Dataset):
    def __init__(self, mode, cfg):

        self.cfg = cfg

        if mode == 'train':
            self.path = 'datasets/ycb/dataset_config/train_data_list.txt'
            self.add_noise = True
        elif mode == 'test':
            self.path = 'datasets/ycb/dataset_config/test_data_list.txt'
            self.add_noise = False #only add noise to training samples

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

        self.frontd = {}

        #symmetries for objects
        self.symmd = {}
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]

        supported_symm_types = {'radial'}

        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('{0}/models/{1}/points.xyz'.format(self.cfg.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            input_file = open('{0}/models/{1}/front.xyz'.format(self.cfg.root, class_input[:-1]))
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
                input_file = open('{0}/models/{1}/symm.txt'.format(self.cfg.root, class_input[:-1]))
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

        print(len(self.list))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.cfg.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.cfg.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.cfg.root, self.list[index]))

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
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, seed)))
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

        #select an object
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

            if self.cfg.image_size != -1:
                rmin, rmax, cmin, cmax = standardize_image_size(self.cfg.image_size, rmin, rmax, cmin, cmax, h, w)

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) == 0:
                continue

            if len(choose) > self.cfg.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.cfg.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
                break
            else:
                choose = np.pad(choose, (0, self.cfg.num_points - len(choose)), 'wrap')
                break

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3][rmin:rmax, cmin:cmax,:]
        img = np.transpose(img, (2, 0, 1))

        if self.list[index][:8] == 'data_syn':
            seed = random.choice(self.real)
            back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
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
        add_t = np.array([random.uniform(-self.cfg.noise_trans, self.cfg.noise_trans) for i in range(3)])

        #right now, we are only dealing with one "front" axis
        front = np.expand_dims(self.frontd[obj[idx]][0], 0) * .1

        if self.add_noise and self.cfg.symm_rotation_aug:
            #PERFORM SYMMETRY ROTATION AUGMENTATION
            #symmetries
            symm = self.symmd[obj[idx]]

            #calculate other peaks based on size of symm
            if len(symm) > 0:
                symm_type, num_symm = symm[0]
                symmetry_augmentation = get_random_rotation_around_symmetry_axis(front, symm_type, num_symm)
                target_r = target_r @ symmetry_augmentation


        cam_scale = meta['factor_depth'][0][0]

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        #NORMALS
        if self.cfg.use_normals:
            depth_mm = (depth * (1000 / cam_scale)).astype(np.uint16)
            normals = compute_normals(depth_mm, cam_fx, cam_fy)
            normals_masked = normals[rmin:rmax, cmin:cmax].reshape((-1, 3))[choose].astype(np.float32).squeeze(0)

        model_points = self.cld[obj[idx]]
        if self.cfg.refine_start:
            select_list = np.random.choice(len(model_points), self.num_pt_mesh_large, replace=False) # without replacement, so that it won't choice duplicate points
        else:
            select_list = np.random.choice(len(model_points), self.num_pt_mesh_small, replace=False) # without replacement, so that it won't choice duplicate points
        model_points = model_points[select_list]

        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)
            
        target_front = np.dot(front, target_r.T)

        if self.add_noise:
            target_front = np.add(target_front, target_t + add_t)
        else:
            target_front = np.add(target_front, target_t)


        img_normalized = self.norm(torch.from_numpy(img_masked.astype(np.float32)))

        if self.cfg.use_colors:
            cloud_colors = img_normalized.view((3, -1)).transpose(0, 1)[choose]

        end_points = {}

        end_points["cloud"] = torch.from_numpy(cloud.astype(np.float32))

        if self.cfg.use_normals:
            end_points["normals"] = torch.from_numpy(normals_masked.astype(np.float32))

        if self.cfg.use_colors:
            end_points["cloud_colors"] = cloud_colors

        end_points["choose"] = torch.LongTensor(choose.astype(np.int32))
        end_points["img"] = img_normalized
        end_points["target"] = torch.from_numpy(target.astype(np.float32))
        end_points["target_front"] = torch.from_numpy(target_front.astype(np.float32))
        end_points["model_points"] = torch.from_numpy(model_points.astype(np.float32))
        end_points["front"] = torch.from_numpy(front.astype(np.float32))
        end_points["obj_idx"] = torch.LongTensor([int(obj[idx]) - 1])

        return end_points

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.cfg.refine_start:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

class PoseDatasetAllObjects(PoseDataset):
    def __getitem__(self, index):

        color_filename = '{0}/{1}-color.png'.format(self.cfg.root, self.list[index])

        img = Image.open(color_filename)
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.cfg.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.cfg.root, self.list[index]))

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
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, seed)))
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

        data_output = []

        orig_img = img

        for idx in range(len(obj)):
            itemid = obj[idx]
            
            img = orig_img

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth
            
            if len(mask.nonzero()[0]) <= self.minimum_num_pt:
                print("WARNING, NOT ENOUGH POINTS LABELED OBJECT {0} in FRAME {1}".format(itemid, color_filename))
                continue

            if self.add_noise:
                img = self.trancolor(img)

            rmin, rmax, cmin, cmax = get_bbox(mask_label)

            img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

            if self.list[index][:8] == 'data_syn':
                seed = random.choice(self.real)
                back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
                back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
                img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
            else:
                img_masked = img

            if self.add_noise and add_front:
                img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

            if self.list[index][:8] == 'data_syn':
                img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)

            # p_img = np.transpose(img_masked, (1, 2, 0))
            # scipy.misc.imsave('temp/{0}_input.png'.format(index), p_img)
            # scipy.misc.imsave('temp/{0}_label.png'.format(index), mask[rmin:rmax, cmin:cmax].astype(np.int32))

            target_r = meta['poses'][:, :, idx][:, 0:3]
            target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
            add_t = np.array([random.uniform(-self.cfg.noise_trans, self.cfg.noise_trans) for i in range(3)])

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) == 0:
                print("WARNING, NO POINTS LABELED OBJECT {0} in FRAME {1}".format(itemid, color_filename))
                continue

            if len(choose) > self.cfg.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.cfg.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.cfg.num_points - len(choose)), 'wrap')
            
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

            #NORMALS
            if self.cfg.use_normals:
                depth_mm = (depth * (1000 / cam_scale)).astype(np.uint16)
                normals = compute_normals(depth_mm, cam_fx, cam_fy)
                normals_masked = normals[rmin:rmax, cmin:cmax].reshape((-1, 3))[choose].astype(np.float32).squeeze(0)

            img_normalized = self.norm(torch.from_numpy(img_masked.astype(np.float32)))

            if self.cfg.use_colors:
                cloud_colors = img_normalized.view((3, -1)).transpose(0, 1)[choose]

            #return all model_points (no sampling), for evaluation
            model_points = self.cld[itemid]

            target = np.dot(model_points, target_r.T)
            if self.add_noise:
                target = np.add(target, target_t + add_t)
            else:
                target = np.add(target, target_t)

            end_points = {}

            end_points["cloud"] = torch.from_numpy(cloud.astype(np.float32))

            if self.cfg.use_normals:
                end_points["normals"] = torch.from_numpy(normals_masked.astype(np.float32))

            if self.cfg.use_colors:
                end_points["cloud_colors"] = cloud_colors

            end_points["choose"] = torch.LongTensor(choose.astype(np.int32))
            end_points["img"] = img_normalized
            end_points["target"] = torch.from_numpy(target.astype(np.float32))
            end_points["model_points"] = torch.from_numpy(model_points.astype(np.float32))
            end_points["obj_idx"] = torch.LongTensor([int(obj[idx]) - 1])

            end_points["intr"] = (cam_fx, cam_fy, cam_cx, cam_cy)

            data_output.append(end_points)

        return data_output



class PoseDatasetPoseCNNResults(PoseDataset):

    def __getitem__(self, index):

        color_filename = '{0}/{1}-color.png'.format(self.cfg.root, self.list[index])

        img = Image.open(color_filename)
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.cfg.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.cfg.root, self.list[index]))

        posecnn_meta = scio.loadmat('{0}/{1}.mat'.format(self.cfg.posecnn_results, '%06d' % index))
        label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])

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
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.cfg.root, seed)))
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

        old_obj = meta['cls_indexes'].flatten().astype(np.int32)

        data_output = []

        orig_img = img

        lst = posecnn_rois[:, 1:2].flatten()

        lst = [int(x) for x in lst if int(x) in old_obj]

        for idx in range(len(lst)):
            itemid = lst[idx]
            
            img = orig_img

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            if len(mask.nonzero()[0]) <= self.minimum_num_pt:
                print("WARNING, NOT ENOUGH POINTS LABELED OBJECT {0} in FRAME {1}".format(itemid, color_filename))
                continue

            if self.add_noise:
                img = self.trancolor(img)

            try:
                rmin, rmax, cmin, cmax = get_bbox_posecnn(posecnn_rois, idx)
            except:
                print("POSECNN NO ROI for obj {0} in frame {1}".format(itemid, color_filename))
                continue

            img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

            if self.list[index][:8] == 'data_syn':
                seed = random.choice(self.real)
                back = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.cfg.root, seed)).convert("RGB")))
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
            add_t = np.array([random.uniform(-self.cfg.noise_trans, self.cfg.noise_trans) for i in range(3)])

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) == 0:
                print("WARNING, NO POINTS LABELED OBJECT {0} in FRAME {1}".format(itemid, color_filename))
                continue

            if len(choose) > self.cfg.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.cfg.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.cfg.num_points - len(choose)), 'wrap')
            
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

            #NORMALS
            if self.cfg.use_normals:
                depth_mm = (depth * (1000 / cam_scale)).astype(np.uint16)
                normals = compute_normals(depth_mm, cam_fx, cam_fy)
                normals_masked = normals[rmin:rmax, cmin:cmax].reshape((-1, 3))[choose].astype(np.float32).squeeze(0)
                cloud = np.hstack((cloud, normals_masked))

            #return all model_points (no sampling), for evaluation
            model_points = self.cld[itemid]

            target = np.dot(model_points, target_r.T)
            if self.add_noise:
                target = np.add(target, target_t + add_t)
            else:
                target = np.add(target, target_t)

            data_output.append(([torch.from_numpy(cloud.astype(np.float32)), \
                torch.LongTensor(choose.astype(np.int32)), \
                self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                torch.from_numpy(target.astype(np.float32)), \
                torch.from_numpy(model_points.astype(np.float32)), \
                torch.LongTensor([int(itemid) - 1])], (cam_fx, cam_fy, cam_cx, cam_cy)))

        return data_output

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.cfg.refine_start:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox_posecnn(posecnn_rois, idx):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
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
