import os
from easydict import EasyDict
import torch

cfg = EasyDict()

# device (multi-core train is not supported)
cfg.device = torch.device("cuda:0")
#cfg.device = torch.device("cpu")
cfg.num_cpu = 8

cfg.base_ = 'bnrnet34-cls100'
cfg.log_base_ = 'bnrnet34-cls100'

# directory
cfg.logdir = '/home/user/kew/KITTI-2D-stereo/checkpoint/' + cfg.log_base_ + '/'
cfg.pretrain_res = '/home/user/kew/KITTI-2D-stereo/checkpoint/pretrain/' + cfg.base_ + '.pth'
cfg.pretrain_neck = '/home/user/kew/KITTI-2D-stereo/checkpoint/pretrain/necknetw_16.pth'

cfg.kitti_home = '/home/user/Dataset/kitti_3dobject/KITTI/object/training/'
cfg.gendata_home = '/home/user/kew/KITTI-2D-stereo/gendata/'
cfg.result_home = '/home/user/kew/KITTI-2D-stereo/result/'
cfg.traintxt = '/home/user/kew/KITTI-2D-stereo/gendata/train.txt'
cfg.valtxt = '/home/user/kew/KITTI-2D-stereo/gendata/val.txt'

# image
cfg.img_width = 768
cfg.img_height = 256
cfg.img_topcut = 100
cfg.normalize = True

# training
cfg.batchsize = 8
cfg.classes = 4
cfg.maxdisp = 128
cfg.maxdepth = 70

cfg.learning_rate = 0.001
cfg.maxepoch = 100
cfg.milestones0 = 1000
cfg.milestones1 = 3712 * cfg.maxepoch / cfg.batchsize * 0.8
cfg.milestones2 = 3712 * cfg.maxepoch / cfg.batchsize * 0.95
def yolo_burnin_schedule(i):
    if i < cfg.milestones0:
        factor = pow(i / cfg.milestones0, 4)
    elif i < cfg.milestones1:
        factor = 1
    elif i < cfg.milestones2:
        factor = 0.1
    else:
        factor = 0.01
    return factor
cfg.burnin_schedule = yolo_burnin_schedule
cfg.use_disploss = False
cfg.depth_lambda = 1.0
cfg.use_decoder = True

# anchors
def make_anchors(size):
    if size == 416:
        anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]
    elif size == 608:
        anchors = [12,16,  19,36,  40,28,  36,75,  76,55,  72,146,  142,110, 192,243,  459,401]
    #wfactor = cfg.yolo_width / size
    hfactor = cfg.img_height / size
    wfactor = 1
    for i in range(18):
        if i%2 ==0:
            anchors[i] = anchors[i] * wfactor
        else :
            anchors[i] = anchors[i] * hfactor
    return anchors

cfg.anchors = make_anchors(608)
cfg.boxes = 60
cfg.yoloboxinput = True

# validation
cfg.testepoch = str(100)
cfg.test_back = '/home/user/kew/KITTI-2D-stereo/checkpoint/' + cfg.log_base_ + '/backnetw_' + cfg.testepoch + '.pth'
cfg.test_neck = '/home/user/kew/KITTI-2D-stereo/checkpoint/' + cfg.log_base_ + '/necknetw_' + cfg.testepoch + '.pth'
cfg.test_head = '/home/user/kew/KITTI-2D-stereo/checkpoint/' + cfg.log_base_ + '/headnetw_' + cfg.testepoch + '.pth'
cfg.resultpath = cfg.result_home + cfg.log_base_ + '/'
cfg.resultimg = False
cfg.resultimgdir = '/home/user/kew/KITTI-2D-stereo/result/image/'
