
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision.ops import nms as torchnms
from torchvision.ops import box_iou as torchiou

# refer : https://github.com/Tianxiaomo/pytorch-YOLOv4

def load_model(m, p, device):
    print(p)
    dict = torch.load(p, map_location=device)
    for i, k in zip(m.state_dict(), dict):
        weight = dict[k]
        m.state_dict()[i].copy_(weight)
        
        
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou
  
  
class Yololoss(nn.Module):
    def __init__(self, anchors, width, height, n_classes, n_anchors, batch, device, depth_lambda):
        super(Yololoss, self).__init__()
        self.cuda_ids = device
        self.strides = [8, 16, 32]
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.depth_lambda = depth_lambda

        self.anchors = [[anchors[0], anchors[1]], [anchors[2], anchors[3]], [anchors[4], anchors[5]],
                        [anchors[6], anchors[7]], [anchors[8], anchors[9]], [anchors[10], anchors[11]],
                        [anchors[12], anchors[13]], [anchors[14], anchors[15]], [anchors[16], anchors[17]]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            wsize = width // self.strides[i]
            hsize = height // self.strides[i]
            grid_x = torch.arange(wsize, dtype=torch.float).repeat(batch, 3, hsize, 1).to(self.cuda_ids).contiguous()
            grid_y = torch.arange(hsize, dtype=torch.float).repeat(batch, 3, wsize, 1).permute(0, 1, 3, 2).to(self.cuda_ids).contiguous()
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, hsize, wsize, 1).permute(0, 3, 1, 2).to(
                self.cuda_ids).contiguous()
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, hsize, wsize, 1).permute(0, 3, 1, 2).to(
                self.cuda_ids).contiguous()

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)


    def build_target(self, pred, labels, batchsize, hsize, wsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, hsize, wsize, 4 + self.n_classes + 1).to(self.cuda_ids)
        obj_mask = torch.ones(batchsize, self.n_anchors, hsize, wsize).to(self.cuda_ids)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, hsize, wsize, 2).to(self.cuda_ids)
        target = torch.zeros(batchsize, self.n_anchors, hsize, wsize, n_ch + 1).to(self.cuda_ids)

        #labels = labels.cpu()
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        
        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2)
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2)
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16)
        truth_j_all = truth_y_all.to(torch.int16)

        for b in range(batchsize):
            n = int(nlabel[b])
            if (int)(n) == 0:
                continue
            truth_box = torch.zeros(n, 4)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]
            
            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box, self.ref_anchors[output_id], CIoU=True)

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |
                           (best_n_all == self.anch_masks[output_id][1]) |
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]
            truth_box = truth_box.to(self.cuda_ids)

            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    target[b, a, j, i, -1:] = labels[b, ti, 5]
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / hsize / wsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2, loss_depth = 0, 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            hsize = output.shape[2]
            wsize = output.shape[3]
            n_ch = 5 + self.n_classes
            output = output.view(batchsize, self.n_anchors, n_ch + 1, hsize, wsize)
            output = output.permute(0, 1, 3, 4, 2).contiguous()
            
            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])
            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, hsize, wsize, n_ch, output_id)
            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:(n_ch + 1)]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:(n_ch + 1)]] *= tgt_mask
            target[..., 2:4] *= tgt_scale
            
            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:n_ch], target=target[..., 5:n_ch], reduction='sum')
            loss_depth += F.smooth_l1_loss(input=output[..., n_ch:], target=target[..., n_ch:], reduction='sum') * self.depth_lambda

        loss = loss_xy + loss_wh + loss_obj + loss_cls + loss_depth
        
        return loss, loss_depth
      
      
def ThresHold_with_depth(loc, conf, depth, num_classes, thr=0.3):
    conf_ = torch.max(conf, dim=1)
    mask = (conf_[0] > thr)

    loc_mask = mask.unsqueeze(1).expand_as(loc)
    loc = loc[loc_mask].view([-1, 4])
    conf_mask = mask.unsqueeze(1).expand_as(conf)
    conf = conf[conf_mask].view([-1, num_classes])
    depth_mask = mask.unsqueeze(1).expand_as(depth)
    depth = depth[depth_mask].view([-1, 1])

    return loc, conf, depth


def read_calib_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R_rect': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def compute_2d_bbox_from_3d_bbox(l, w, h, x, y, z, ry, K):
    # compute rotational matrix around yaw axis
    c = np.cos(ry)
    s = np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), K)

    xmin = corners_2d[:, 0].min()
    ymin = corners_2d[:, 1].min()
    xmax = corners_2d[:, 0].max()
    ymax = corners_2d[:, 1].max()

    return xmin, ymin, xmax, ymax


def project_to_image(pts_3d, P):
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


  
def NmsCls_with_depth(loc, conf, depth, thr=0.5):
    conf_ = torch.max(conf, dim=1)
    
    dic = {}
    for i in range(conf.shape[1]):
        mask = (conf_[1] == i)
        if (torch.sum(mask).item() == 0):
            continue
        loc_mask = mask.unsqueeze(1).expand_as(loc)
        depth_mask = mask.unsqueeze(1).expand_as(depth)
        score_mask = mask

        l = loc[loc_mask].view([-1, 4])
        d = depth[depth_mask].view([-1, 1])
        s = conf_[0][score_mask]
        
        a = torchnms(l, s, thr)
        s = s[a[:]].view([-1, 1])
        l = l[a[:], :].view([-1, 4])
        d = d[a[:], :].view([-1, 1])

        #mask = VerifyCord(l)
        #score_mask = mask.unsqueeze(1).expand_as(s)
        #depth_mask = mask.unsqueeze(1).expand_as(d)
        #loc_mask = mask.unsqueeze(1).expand_as(l)
        
        #s = s[score_mask].view([-1, 1])
        #l = l[loc_mask].view([-1, 4])
        #d = d[depth_mask].view([-1, 1])
        
        k = torch.cat([s, l, d], dim=1)
        if k.shape[0] != 0:
            dic[str(i)] = k

    return dic
