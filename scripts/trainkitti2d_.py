
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging
import datetime
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFGkitti2D import cfg
from src.models.BNRNet34 import BNRNet34
from src.models.StereoNeck import StereoNeck
from src.models.DetectHead import DetectHead
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
    trainlogger.info(cfg.base_)

    back = BNRNet34().cuda(cfg.device)
    load_model(back, cfg.pretrain_res, cfg.device)

    back_optimizer = optim.Adam(back.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    back_scheduler = optim.lr_scheduler.LambdaLR(back_optimizer, cfg.burnin_schedule)
    back.train()

    neck = StereoNeck(cfg.maxdisp).to(cfg.device)
    load_model(neck, cfg.pretrain_neck, cfg.device)
    neck_optimizer = optim.Adam(neck.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    neck_scheduler = optim.lr_scheduler.LambdaLR(neck_optimizer, cfg.burnin_schedule)
    neck.train()

    head = DetectHead(cfg.anchors, cfg.classes, cfg.use_decoder).to(cfg.device)
    head_optimizer = optim.Adam(head.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    head_scheduler = optim.lr_scheduler.LambdaLR(head_optimizer, cfg.burnin_schedule)
    head.train()
    # define dataloader
    kitti_dataset = kitti2ddataset(cfg)
    kitti_loader = DataLoader(kitti_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_cpu,\
       pin_memory=True, drop_last=True)

    # define loss func
    lossfunc = Yololoss(cfg.anchors, cfg.img_width, cfg.img_height, cfg.classes, 3, cfg.batchsize, cfg.device, cfg.depth_lambda)

    lossrecord = []
    disploss_rec = []
    depthloss_rec = []

    for epoch in range(1, cfg.maxepoch+1):
        # print milestones
        trainlogger.info('({} / {}) epoch'.format(epoch, cfg.maxepoch))
        
        for cnt, (left_img, right_img , dataL, target, left_path) in enumerate(kitti_loader):
            left_img, right_img, dataL = left_img.to(cfg.device), right_img.to(cfg.device), dataL.to(cfg.device)
            if cfg.use_disploss:
                mask1 = dataL < (int)(cfg.maxdisp)
                mask2 = dataL > (int)(0)
                mask = mask1 & mask2
                mask.detach_()
                disptarget = {
                    'dataL': dataL,
                    'mask': mask
                }
            target = target.to(dtype=torch.float64).to(cfg.device)

            # forward
            _, _, L8, L16, L32 = back(left_img)
            R8 = back(right_img, mode='right')

            if cfg.use_disploss:
                s8, disploss = neck(L8, R8, disptarget)
            else:
                s8 = neck(L8, R8)

            if cfg.use_decoder:
                output = head(s8, L32)
            else:
                output = head(s8, None)

            # loss calculate
            loss, depthloss = lossfunc(output, target)
            if cfg.use_disploss:
                loss = loss + disploss
            
            # backward
            loss.backward()

            back_optimizer.step()
            back.zero_grad()
            back_scheduler.step()
            neck_optimizer.step()
            neck.zero_grad()
            neck_scheduler.step()
            head_optimizer.step()
            head.zero_grad()
            head_scheduler.step()

            lossrecord.append(loss.item())
            if cfg.use_disploss:
                disploss_rec.append(disploss.item())
            else:
                disploss_rec.append(1)
            depthloss_rec.append(depthloss.item())

            # print steploss
            #print("{}/{} steploss:{:.6f}".format(cnt, len(kitti_loader), loss.item()), end="\r")
            print("{}/{} {}/{} loss: {:.2f} disploss: {:.2f} depthloss: {:.2f}".format(
                epoch, cfg.maxepoch, cnt, len(kitti_loader),
                sum(lossrecord) / len(lossrecord), sum(disploss_rec) / len(disploss_rec), sum(depthloss_rec) / len(depthloss_rec)), end="\r")
            continue
        
        trainlogger.info("{}/{} {}/{} loss: {:.2f} disploss: {:.2f} depthloss: {:.2f}".format(
                epoch, cfg.maxepoch, cnt, len(kitti_loader),
                sum(lossrecord) / len(lossrecord), sum(disploss_rec) / len(disploss_rec), sum(depthloss_rec) / len(depthloss_rec)))
        lossrecord = []
        disploss_rec = []
        depthloss_rec = []

        # save model
        if epoch % 5 == 0:
            torch.save(back.state_dict(), cfg.logdir + 'backnetw_' + str(epoch) + '.pth')
            torch.save(neck.state_dict(), cfg.logdir + 'necknetw_' + str(epoch) + '.pth')
            torch.save(head.state_dict(), cfg.logdir + 'headnetw_' + str(epoch) + '.pth')
            trainlogger.info('{} epoch model saved !'.format(epoch))

        continue
    # end.
    trainlogger.info('-end-')


def write_pred_result_only2dbbox(pred, left_path, oh, ow):
    imgnum = left_path.split('/')[-1].split('.')[0]
    label_list = []

    with open(cfg.resultpath + imgnum + '.txt', mode='w') as f:

        for i in range(cfg.classes):
            if str(i) in pred:
                pred_ = pred[str(i)]
                for box in pred_:
                    score, xmin, ymin, xmax, ymax, depth = box
                    #score = 1.00
                    xmin = xmin * ow
                    ymin = ymin * (oh - cfg.img_topcut) + cfg.img_topcut
                    xmax = xmax * ow
                    ymax = ymax * (oh - cfg.img_topcut) + cfg.img_topcut
                    # label = '%d %d %d %d %d %.2f %.2f %d %.2f %.2f\n'\
                    # % (
                    #     int(i), int(xmin), int(ymin), int(xmax), int(ymax), depth, 
                    #     0, 0, 0, score
                    #     )

                    w_ = xmax - xmin
                    h_ = ymax - ymin
                    label = '%d %.6f %d %d %d %d\n'\
                    % (
                        int(i), score, int(xmin), int(ymin), int(w_), int(h_),
                        )
                    f.write(label)
                    label_list.append(label)
                    continue
            continue
    return label_list


def GenResult_2dbboxTxt():
    # define model
    back = BNRNet34().cuda(cfg.device)
    load_model(back, cfg.test_back, cfg.device)
    back.eval()
    
    neck = StereoNeck(cfg.maxdisp).to(cfg.device)
    load_model(neck, cfg.test_neck, cfg.device)
    neck.eval()

    head = DetectHead(cfg.anchors, cfg.classes, cfg.use_decoder).to(cfg.device)
    load_model(head, cfg.test_head, cfg.device)
    head.eval()

    # define dataset
    kitti_dataset = kitti2ddataset(cfg, mode='val')

    with torch.no_grad():
        for imgid in range(len(kitti_dataset)):
            left_img, right_img , dataL, target, left_path, oh, ow = kitti_dataset[imgid]
            #print(left_path)
            imgnum = left_path.split('/')[-1].split('.')[0]
            left_img = left_img.to(cfg.device)
            right_img = right_img.to(cfg.device)

            # forward
            _, _, L8, L16, L32 = back(left_img.unsqueeze(0))
            R8 = back(right_img.unsqueeze(0), mode='right')
            s8 = neck(L8, R8)

            if cfg.use_decoder:
                output = head(s8, L32)
            else:
                output = head(s8, None)

            loc = output[0].squeeze(0).squeeze(1).cpu()
            conf = output[1].squeeze(0).cpu()
            depth = output[2].squeeze(0).cpu()
            
            loc, conf, depth = ThresHold_with_depth(loc, conf, depth, cfg.classes, thr=0.3)

            if loc.shape[0] == 0:
                with open(cfg.resultpath + imgnum + '.txt', mode='w') as f:
                    label = '%d %.2f %d %d %d %d\n'\
                        % (
                            0, 1.0, 10, 10, 10, 10,
                            )
                    f.write(label)
                labels = None
            else:
                pred = NmsCls_with_depth(loc, conf, depth, thr=0.5)
                labels = write_pred_result_only2dbbox(pred, left_path, oh, ow)

            if cfg.resultimg:
                left_img = cv2.imread(left_path)

                if labels == None:
                    pass
                else:
                    for obj in labels:
                        obj = obj.replace("\n", "").split(' ')
                        xmin, ymin, w_, h_ = int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])
                        xmax = xmin + w_
                        ymax = ymin + h_
                        cv2.putText(left_img, obj[5], (xmin+1, ymin+1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.rectangle(left_img, (xmin, ymin), (xmax ,ymax), (0, 255, 255), 3)
                        continue

                # if target == None:
                #     pass
                # else:
                #     for obj in target:
                #         xmin, ymin, xmax, ymax = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                #         depth = float(obj[5].item())
                #         xmin = xmin / cfg.img_width * ow
                #         ymin = ymin / cfg.img_height * (oh - cfg.img_topcut) + cfg.img_topcut
                #         xmax = xmax / cfg.img_width * ow
                #         ymax = ymax / cfg.img_height * (oh - cfg.img_topcut) + cfg.img_topcut
                #         xmin = int(xmin)
                #         ymin = int(ymin)
                #         xmax = int(xmax)
                #         ymax = int(ymax)
                #         cv2.putText(left_img, str(depth), (xmin+1, ymin+1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                #         cv2.rectangle(left_img, (xmin, ymin), (xmax ,ymax), (255, 0, 0), 3)
                cv2.imwrite(cfg.resultimgdir + imgnum + '.png', left_img)
            #df=df
            continue



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()
    #GenResult_2dbboxTxt()
