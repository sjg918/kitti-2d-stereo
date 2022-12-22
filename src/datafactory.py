
import os
import random
import chardet
import re

import torch.utils.data as data
import torch

import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np


class sceneflowdataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg

        if mode == 'train':
            with open(self.cfg.traintxt, 'r') as f:
                lines = [line.rstrip() for line in f.readlines()]
        elif mode == 'val':
            with open(self.cfg.valtxt, 'r') as f:
                lines = [line.rstrip() for line in f.readlines()]
        else:
            assert 'check dataset mode!'
        self.id_list = [i for i in range(len(lines))]
        splits = [line.split() for line in lines]
        self.left = [x[0] for x in splits]
        self.right = [x[1] for x in splits]
        self.disp = [x[2] for x in splits]

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.totensor = transforms.ToTensor()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        # self.normalize = transforms.Normalize(
        #         mean = [0.485, 0.456, 0.406],
        #         std = [0.229, 0.224, 0.225]
        #         )

    def __len__(self):
        return len(self.left)

    def __getitem__(self, id):
        left_path = self.cfg.sceneflow_home + self.left[id]
        right_path = self.cfg.sceneflow_home + self.right[id]
        disp_path = self.cfg.sceneflow_home + self.disp[id]
        left_img, right_img = cv2.imread(left_path), cv2.imread(right_path)

        dataL, scaleL = readPFM(disp_path)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataL = self.totensor(dataL)

        if self.mode == 'train':                        
            h, w, _ = left_img.shape
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
            right_img = right_img[y1:y1 + th, x1:x1 + tw, :]
            dataL = dataL[:, y1:y1 + th, x1:x1 + tw]

            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            left_img = (left_img.astype(np.float32) / 255.)
            right_img = (right_img.astype(np.float32) / 255.)
            color_aug(self._data_rng, left_img, self._eig_val, self._eig_vec, right_img)

            left_img = (left_img - self.mean) / self.std
            #left_img = left_img.transpose(2, 0, 1)
            left_img = self.totensor(left_img)

            right_img = (right_img - self.mean) / self.std
            #right_img = right_img.transpose(2, 0, 1)
            right_img = self.totensor(right_img)

            return left_img, right_img , dataL, left_path
        elif self.mode == 'val':
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_img = (left_img.astype(np.float32) / 255.)
            left_img = (left_img - self.mean) / self.std
            left_img = left_img.transpose(2, 0, 1)
            left_img = self.totensor(left_img)

            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = (right_img.astype(np.float32) / 255.)
            right_img = (right_img - self.mean) / self.std
            right_img = right_img.transpose(2, 0, 1)
            right_img = self.totensor(right_img)

            return left_img, right_img, dataL, left_path


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)  
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def lighting_(data_rng, left_image, alphastd, eigval, eigvec, right_image):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    left_image += np.dot(eigvec, eigval * alpha)
    if right_image is not None:
        right_image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, left_image, left_gs, left_gs_mean, var, right_image, right_gs, right_gs_mean):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, left_image, left_gs[:, :, None])
    if right_image is not None:
        blend_(alpha, right_image, right_gs[:, :, None])

def brightness_(data_rng, left_image, left_gs, left_gs_mean, var, right_image, right_gs, right_gs_mean):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    left_image *= alpha
    if right_image is not None:
        right_image *= alpha

def contrast_(data_rng, left_image, left_gs, left_gs_mean, var, right_image, right_gs, right_gs_mean):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, left_image, left_gs_mean)
    if right_image is not None:
        blend_(alpha, right_image, right_gs_mean)

def color_aug(data_rng, left_image, eig_val, eig_vec, right_image=None):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    left_gs = grayscale(left_image)
    left_gs_mean = left_gs.mean()
    if right_image is not None:
        right_gs = grayscale(right_image)
        right_gs_mean = right_gs.mean()
    else:
        right_gs = None
        right_gs_mean = None

    for f in functions:
        f(data_rng, left_image, left_gs, left_gs_mean, 0.4, right_image, right_gs, right_gs_mean)
    lighting_(data_rng, left_image, 0.1, eig_val, eig_vec, right_image)


class kitti2ddataset(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        if mode == 'train':
            with open(self.cfg.traintxt) as f:
                self.lists = f.readlines()
                self.lists = [i.replace("\n", "") for i in self.lists]
        elif mode == 'val':
            with open(self.cfg.valtxt) as f:
                self.lists = f.readlines()
                self.lists = [i.replace("\n", "") for i in self.lists]
        else:
            assert 'check dataset mode!'

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.totensor = transforms.ToTensor()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.yoloboxinput = cfg.yoloboxinput

    def __len__(self):
        return len(self.lists)

    def bboxloader(self, label_path, h, w):
        with open(label_path, mode='r') as f:
            bbox = f.readlines()
            bbox = [i.replace("\n", "") for i in bbox]

        if len(bbox) == '0':
            return None
        
        bbox = [k.split(" ") for k in bbox]
        # bbox = [list(map(int, k)) for k in bbox]
        # for j, k in enumerate(bbox):
        #     if k[3] - k[1] < 6:
        #         bbox.pop(j)
        newbox = []
        for box in bbox:
            cls, xmin, ymin, xmax, ymax, depth = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), float(box[5])
            newbox.append([xmin, ymin, xmax, ymax, cls, depth])
        boxlen = len(newbox)

        newbox = torch.tensor(newbox, dtype=torch.float64)
        if boxlen == 1:
            newbox.unsqueeze(0)
        # newbox[:, 0] = newbox[:, 0] * (self.cfg.img_width / w)
        # newbox[:, 1] = newbox[:, 1] * (self.cfg.img_height / h)
        # newbox[:, 2] = newbox[:, 2] * (self.cfg.img_width / w)
        # newbox[:, 3] = newbox[:, 3] * (self.cfg.img_height / h)
        #newbox = newbox.roll(-1, dims=1)
        return newbox


    def __getitem__(self, id):
        if self.mode == 'val':
            left_path = self.cfg.kitti_home + 'image_2/' + self.lists[id] + '.png'
            right_path = self.cfg.kitti_home + 'image_3/' + self.lists[id] + '.png'
            label_path = self.cfg.gendata_home + '2Dlabel_2/' + self.lists[id] + '.txt'
            disp_path = self.cfg.gendata_home + 'image_2sgm/' + self.lists[id] + '.png'

            left_img, right_img = cv2.imread(left_path), cv2.imread(right_path)
            dataL = cv2.imread(disp_path)
            dataL = cv2.cvtColor(dataL, cv2.COLOR_BGR2GRAY)
            oh, ow, _ = left_img.shape
            if os.path.isfile(label_path):
                bbox = self.bboxloader(label_path, ow, oh)
            else:
                bbox = None

            if self.cfg.img_topcut > 0:
                topcut = self.cfg.img_topcut
                left_img = left_img[topcut:, :, :]
                right_img = right_img[topcut:, :, :]
                dataL = dataL[topcut:, :]
                if bbox is not None: 
                    bbox[:, 1] = torch.clamp(bbox[:, 1] - topcut, min=0)
                    bbox[:, 3] = torch.clamp(bbox[:, 3] - topcut, min=0)
            
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            left_img = cv2.resize(left_img, (self.cfg.img_width, self.cfg.img_height), cv2.INTER_LINEAR)
            right_img = cv2.resize(right_img, (self.cfg.img_width, self.cfg.img_height), cv2.INTER_LINEAR)
            dataL = cv2.resize(dataL, (self.cfg.img_width, self.cfg.img_height), cv2.INTER_NEAREST)

            left_img = (left_img.astype(np.float32) / 255.)
            right_img = (right_img.astype(np.float32) / 255.)
            if self.cfg.normalize:
                left_img = (left_img - self.mean) / self.std
                right_img = (right_img - self.mean) / self.std
            left_img = left_img.transpose(2, 0, 1)
            right_img = right_img.transpose(2, 0, 1)
            left_img = torch.from_numpy(left_img)
            right_img = torch.from_numpy(right_img)

            if bbox is not None:
                bbox[:, 0] = bbox[:, 0] * self.cfg.img_width / ow
                bbox[:, 1] = bbox[:, 1] * self.cfg.img_height / (oh - self.cfg.img_topcut)
                bbox[:, 2] = bbox[:, 2] * self.cfg.img_width / ow
                bbox[:, 3] = bbox[:, 3] * self.cfg.img_height / (oh - self.cfg.img_topcut)
            else:
                bbox = np.array([10, 10, 20, 20, 0], dtype=np.float32)
                bbox = np.expand_dims(bbox, 0)
                bbox = torch.from_numpy(bbox)

            dataL = torch.from_numpy(dataL).unsqueeze(0).to(torch.float64)
            target = bbox
            return left_img, right_img , dataL, target, left_path, oh, ow

        if torch.rand(1) < 0.5:
            while True:
                if os.path.isfile(self.cfg.gendata_home + '2Dlabel_2/' + self.lists[id] + '.txt'):
                    break
                else:
                    id = random.randint(0, len(self.lists) - 1)
                    continue

            left_path = self.cfg.kitti_home + 'image_2/' + self.lists[id] + '.png'
            right_path = self.cfg.kitti_home + 'image_3/' + self.lists[id] + '.png'
            label_path = self.cfg.gendata_home + '2Dlabel_2/' + self.lists[id] + '.txt'
            disp_path = self.cfg.gendata_home + 'image_2sgm/' + self.lists[id] + '.png'

            left_img, right_img = cv2.imread(left_path), cv2.imread(right_path)
        else:
            while True:
                if os.path.isfile(self.cfg.gendata_home + '2Dlabel_3/' + self.lists[id] + '.txt'):
                    break
                else:
                    id = random.randint(0, len(self.lists) - 1)
                    continue

            left_path = self.cfg.kitti_home + 'image_3/' + self.lists[id] + '.png'
            right_path = self.cfg.kitti_home + 'image_2/' + self.lists[id] + '.png'
            label_path = self.cfg.gendata_home + '2Dlabel_3/' + self.lists[id] + '.txt'
            disp_path = self.cfg.gendata_home + 'image_3sgm/' + self.lists[id] + '.png'

            left_img, right_img = cv2.imread(left_path), cv2.imread(right_path)
            left_img = cv2.flip(left_img, 1)
            right_img = cv2.flip(right_img, 1)
        
        dataL = cv2.imread(disp_path)
        dataL = cv2.cvtColor(dataL, cv2.COLOR_BGR2GRAY)
        oh, ow, _ = left_img.shape
        bbox = self.bboxloader(label_path, ow, oh)

        if self.cfg.img_topcut > 0:
            topcut = self.cfg.img_topcut
            left_img = left_img[topcut:, :, :]
            right_img = right_img[topcut:, :, :]
            dataL = dataL[topcut:, :]
            bbox[:, 1] = torch.clamp(bbox[:, 1] - topcut, min=0)
            bbox[:, 3] = torch.clamp(bbox[:, 3] - topcut, min=0)

        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        left_img = cv2.resize(left_img, (self.cfg.img_width, self.cfg.img_height), cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (self.cfg.img_width, self.cfg.img_height), cv2.INTER_LINEAR)
        dataL = cv2.resize(dataL, (self.cfg.img_width, self.cfg.img_height), cv2.INTER_NEAREST)

        left_img = (left_img.astype(np.float32) / 255.)
        right_img = (right_img.astype(np.float32) / 255.)

        color_aug(self._data_rng, left_img, self._eig_val, self._eig_vec, right_img)


        if self.cfg.normalize:
            left_img = (left_img - self.mean) / self.std
            right_img = (right_img - self.mean) / self.std
        left_img = left_img.transpose(2, 0, 1)
        right_img = right_img.transpose(2, 0, 1)
        left_img = torch.from_numpy(left_img)
        right_img = torch.from_numpy(right_img)

        bbox[:, 0] = bbox[:, 0] * self.cfg.img_width / ow
        bbox[:, 1] = bbox[:, 1] * self.cfg.img_height / (oh - self.cfg.img_topcut)
        bbox[:, 2] = bbox[:, 2] * self.cfg.img_width / ow
        bbox[:, 3] = bbox[:, 3] * self.cfg.img_height / (oh - self.cfg.img_topcut)

        mask = bbox[:, 5] > self.cfg.maxdepth
        bbox[mask, 5] = self.cfg.maxdepth

        dataL = torch.from_numpy(dataL).unsqueeze(0).to(torch.float64)
        if self.yoloboxinput:
            target = torch.zeros([self.cfg.boxes, 6])
            target[:min(len(bbox), self.cfg.boxes)] = bbox[:min(len(bbox), self.cfg.boxes)]
            return left_img, right_img , dataL, target, left_path
        else:
            return left_img, right_img , dataL, bbox, left_path

    # def collate_fn_cpu_yolo(self, batch):
    #     left_img_list = []
    #     right_img_list = []
    #     dataL_list = []
    #     target_list = []
    #     left_path_list = []
    #     for left_img, right_img , dataL, target, left_path in batch:
    #         left_img_list.append(left_img.unsqueeze(0))
    #         right_img_list.append(right_img.unsqueeze(0))
    #         dataL_list.append(dataL.unsqueeze(0))
    #         target_list.append(target.unsqueeze(0))
    #         left_path_list.append(left_path)
    #     left_img_list = torch.cat(left_img_list, dim=0)
    #     right_img_list = torch.cat(right_img_list, dim=0)
    #     dataL_list = torch.cat(dataL_list, dim=0)
    #     target_list = torch.cat(target_list, dim=0)
    #     return left_img_list, right_img_list, dataL_list, target_list, left_path_list

    @staticmethod
    def collate_fn_cpu(batch):
        left_img_list = []
        right_img_list = []
        dataL_list = []
        target_list = []
        left_path_list = []
        for left_img, right_img , dataL, target, left_path in batch:
            left_img_list.append(left_img.unsqueeze(0))
            right_img_list.append(right_img.unsqueeze(0))
            dataL_list.append(dataL.unsqueeze(0))
            target_list.append(target)
            left_path_list.append(left_path)
        left_img_list = torch.cat(left_img_list, dim=0)
        right_img_list = torch.cat(right_img_list, dim=0)
        dataL_list = torch.cat(dataL_list, dim=0)
        return left_img_list, right_img_list, dataL_list, target_list, left_path_list
