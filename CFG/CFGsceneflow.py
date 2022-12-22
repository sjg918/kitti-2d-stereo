
from easydict import EasyDict
import torch

cfg = EasyDict()

# device (multi-core train is not supported)
cfg.device = torch.device("cuda:0")
cfg.num_cpu = 4

cfg.base_ = 'bnrnet34-cls100'

# directory
cfg.sceneflow_home = '/home/user/Dataset/sceneflow/'
cfg.logdir = '/home/user/kew/KITTI-2D-stereo/checkpoint/' + cfg.base_ + '-sceneflowpre/'
cfg.traintxt = '/home/user/kew/KITTI-2D-stereo/gendata/sceneflow_train.txt'
cfg.valtxt = '/home/user/kew/KITTI-2D-stereo/gendata/sceneflow_test.txt'

cfg.pretrain_res = '/home/user/kew/KITTI-2D-stereo/checkpoint/pretrain/' + cfg.base_ + '.pth'

# training
cfg.maxdisp = 128
cfg.freeze_back = True


cfg.batchsize = 16
cfg.learning_rate = 0.001
cfg.epoch_milestones_0 = 0
cfg.maxepoch = 16
cfg.epoch_milestones_1 = 10
cfg.epoch_milestones_2 = 14
def burnin_schedule_(i):
    if i < cfg.epoch_milestones_0:
        factor = pow(i / cfg.stereo_milestones_0, 4)
    elif i < cfg.epoch_milestones_1:
        factor = 1
    elif i < cfg.epoch_milestones_2:
        factor = 0.5
    else:
        factor = 0.25
    return factor
cfg.burnin_schedule = burnin_schedule_
