
import logging
import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFGsceneflow import cfg
from src.models.BNRNet34 import BNRNet34
from src.models.StereoNeck import StereoNeck_pretrain
from src.datafactory import *
from src.utils import *

trainlogger = logging.getLogger("train")

formatter = logging.Formatter('(%(asctime)s) %(message)s',"%Y-%m-%d %H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(cfg.logdir + 'train.log')
file_handler.setFormatter(formatter)
trainlogger.addHandler(file_handler)
trainlogger.addHandler(stream_handler)
trainlogger.setLevel(logging.INFO)

def train():
    
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(7)

    # start
    trainlogger.info('-start-')

    # define model
    back = BNRNet34().cuda(cfg.device)
    load_model(back, cfg.pretrain_res, cfg.device)


    neck = StereoNeck_pretrain(cfg.maxdisp).cuda(cfg.device)

    # define dataloader
    kitti_dataset = sceneflowdataset(cfg)
    kitti_loader = DataLoader(kitti_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_cpu,\
       pin_memory=True, drop_last=True)

    # define optimizer and scheduler
    if cfg.freeze_back:
        pass
    else:
        back_optimizer = optim.Adam(back.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        back_scheduler = optim.lr_scheduler.LambdaLR(back_optimizer, cfg.burnin_schedule)
    neck_optimizer = optim.Adam(neck.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    neck_scheduler = optim.lr_scheduler.LambdaLR(neck_optimizer, cfg.burnin_schedule)

    if cfg.freeze_back:
        back.eval()
    else:
        back.train()
    neck.train()
    lossrecord = []

    for epoch in range(1, cfg.maxepoch+1):
        # print milestones
        trainlogger.info('({} / {}) epoch'.format(epoch, cfg.maxepoch))
        
        for cnt, (left_img, right_img , dataL, left_path) in enumerate(kitti_loader):
            left_img, right_img, dataL = left_img.cuda(cfg.device), right_img.cuda(cfg.device), dataL.cuda(cfg.device)
            mask1 = dataL < (int)(cfg.maxdisp)
            mask2 = dataL > (int)(0)
            mask = mask1 & mask2
            mask.detach_()

            # forward
            if cfg.freeze_back:
                with torch.no_grad():
                    _, L4, L8, _, _ = back(left_img)
                    _, R4, R8, _, _ = back(right_img)
            else:
                _, L4, L8, _, _ = back(left_img)
                _, R4, R8, _, _ = back(right_img)
            output = neck(L8, R8)

            # loss calculate
            loss = F.smooth_l1_loss(output[mask], dataL[mask], reduction='mean')
            if loss > 200:
                print(left_path)
                assert 'wtf'
            
            # backward
            loss.backward()
            if cfg.freeze_back:
                pass
            else:
                back_optimizer.step()
                back.zero_grad()
            neck_optimizer.step()
            neck.zero_grad()
            lossrecord.append(loss.item())

            # print steploss
            #print("{}/{} steploss:{:.6f}".format(cnt, len(kitti_loader), loss.item()), end="\r")
            print("{}/{} {}/{} loss: {:.2f}".format(epoch, cfg.maxepoch, cnt, len(kitti_loader), sum(lossrecord) / len(lossrecord)), end="\r")
            continue
        
        # learning rate scheduling
        if cfg.freeze_back:
            pass
        else:
            back_scheduler.step()
        neck_scheduler.step()
        trainlogger.info("{}/{} {}/{} loss: {:.2f}".format(epoch, cfg.maxepoch, cnt, len(kitti_loader), sum(lossrecord) / len(lossrecord)))
        lossrecord = []

        # save model
        if epoch % 1 == 0:
            if cfg.freeze_back:
                pass
            else:
                torch.save(back.state_dict(), cfg.logdir + 'backnetw_' + str(epoch) + '.pth')
            torch.save(neck.state_dict(), cfg.logdir + 'necknetw_' + str(epoch) + '.pth')
            #torch.save(neck_optimizer.state_dict(), './weights/' + cfg.saveplace + '/neckopti_' + str(epoch) + '.pth')
            #torch.save(neck_optimizer.state_dict(), './weights/' + cfg.saveplace + '/headopti_' + str(epoch) + '.pth')
            trainlogger.info('{} epoch model saved !'.format(epoch))

        continue
    # end.
    trainlogger.info('-end-')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()
