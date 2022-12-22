
import os
import cv2
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from CFG.CFGkitti2D import cfg
from torchvision.ops import box_iou as torchiou
import torch
import math

from src.utils import *


def GenResultTxt_fromkitti(kittiresultdir, resultpath):
    txt_list = os.listdir(kittiresultdir)

    for txt in txt_list:
        #print(txt)
        label_list = []
        imgnum = txt.split('.')[0]
        left_img = cv2.imread(cfg.kitti_home + 'image_2/' + imgnum + '.png')
        imgh, imgw, _ = left_img.shape
        calib = read_calib_file(cfg.kitti_home + 'calib/' + imgnum + '.txt')
        K = calib['P2']

        with open(kittiresultdir + txt, mode='r') as f:
            bbox = f.readlines()
            if len(bbox) == 0:
                pass
            else:
                bbox = [i.replace("\n", "") for i in bbox]
                for box in bbox:
                    box = box.split(' ')
                    cls, xmin_, ymin_, xmax_, ymax_ = box[0], float(box[4]), float(box[5]), float(box[6]), float(box[7])
                    h, w, l = float(box[8]), float(box[9]), float(box[10])
                    x, y, z = float(box[11]), float(box[12]), float(box[13])
                    ry = float(box[14])
                    score = float(box[15]) #if box.__len__() == 16 else 1.00

                    #score = 1.00

                    # KM3D error. (ex. 001699.txt, 002586.txt)
                    if xmax_ < 0 or xmin_ >= imgw:
                        continue

                    xmin, ymin, xmax, ymax = compute_2d_bbox_from_3d_bbox(l, w, h, x, y, z, ry, K)
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax >= imgw:
                        xmax = imgw -1
                    if ymax >= imgh:
                        ymax = imgh -1

                    # if (abs(xmin_ - xmin) > 2) or (abs(xmax_ - xmax) > 2) or (abs(ymin_ - ymin) > 2) or (abs(ymax_ - ymax) > 2):
                    #     print('ssib')
                    #     print(txt)
                    #     print(xmin, ymin, xmax, ymax)
                    #     print(xmin_, ymin_, xmax_, ymax_)
                    #     df=df


                    w_ = xmax - xmin
                    h_ = ymax - ymin

                    # KM3D error. (ex. 002621.txt)
                    if w_ < 3:
                        continue
                    #depth = compute_shortest_distance(l, w, h, x, y, z, ry)
                    
                    label = '%d %.6f %d %d %d %d\n'% (convert_cls(cls), score, int(xmin), int(ymin), int(w_), int(h_))
                    label_list.append(label)
                    continue

        with open(resultpath + txt, mode='w') as f:
            if len(label_list) == 0:
                label = '%d %.2f %d %d %d %d\n'\
                        % (
                            0, 1.0, 10, 10, 10, 10,
                            )
                f.write(label)
            else:
                for label in label_list:
                    f.write(label)
                    continue
        #df=df
        continue



if __name__ == '__main__':
    print('hi')
