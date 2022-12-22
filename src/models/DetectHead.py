
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


def depth_regression(x, maxdepth=70):
    assert len(x.shape) == 4
    dpeth_values = torch.arange(0, maxdepth, dtype=x.dtype).to(x.device)
    dpeth_values = dpeth_values.view(1, maxdepth, 1, 1)
    return torch.sum(x * dpeth_values, 1, keepdim=False)


def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []
    depth_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])
        depth_list.append(item[2])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
    depth = torch.cat(depth_list, dim=1)
        
    return [boxes, confs, depth]


def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []
    depth_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes + 1)
        end = (i + 1) * (5 + num_classes + 1)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4])
        cls_confs_list.append(output[:, begin + 5 : end - 1])
        depth_list.append(output[:, end - 1])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    depth = torch.cat(depth_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    depth = depth.view(batch, num_anchors, 1, H*W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes]
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)
    depth = depth.permute(0, 1, 3, 2).reshape(batch, num_anchors * H* W, 1)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)
    #disp = torch.sigmoid(disp)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= W
    by_bh /= H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    by = by_bh[:, :num_anchors].view(batch, num_anchors * H * W, 1)
    bw = bx_bw[:, num_anchors:].view(batch, num_anchors * H * W, 1)
    bh = by_bh[:, num_anchors:].view(batch, num_anchors * H * W, 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(batch, num_anchors * H * W, 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs
    #disp = disp * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs, depth

class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)


class myConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm=True, bias=False):
        super().__init__()

        pad = (kernel_size - 1) // 2
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=bias))
        if norm:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(Mish())
        if activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1))
        elif activation == 'linear':
            pass
        else:
            assert 'unknown activation function!'

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class DetectHead(nn.Module):
    def __init__(self, anchors, yolo_classes, use_decoder):
        super().__init__()
        self.output_ch = (4 + 1 + yolo_classes)

        self.yolo1 = YoloLayer(anchor_mask=[0, 1, 2], num_classes=yolo_classes,
                                anchors=anchors,
                                num_anchors=9, stride=8)
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=yolo_classes,
                                anchors=anchors,
                                num_anchors=9, stride=16)
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=yolo_classes,
                                anchors=anchors,
                                num_anchors=9, stride=32)

        if use_decoder:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=512,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
            )

        inc = 256
        if use_decoder:
            inc = inc + 128
        #
        self.in_stereo_conv1 = myConv2D(256, 256, 3, 1, 'leaky')
        if use_decoder:
            self.in_decoder_conv1 = myConv2D(128, 128, 3, 1, 'leaky')
        self.conv1 = myConv2D(inc, 128, 1, 1, 'leaky')
        self.conv2 = myConv2D(128, 256, 3, 1, 'leaky')
        self.out_bbox_conv1 = myConv2D(256, self.output_ch * 3, 1, 1, 'linear', norm=False, bias=True)
        self.out_depth_conv1 = myConv2D(256, 70 * 3, 1, 1, 'linear', norm=False, bias=True)

        # conv1
        self.downsample1 = myConv2D(128, 256, 3, 2, 'leaky')
        #
        inc = 512
        if use_decoder:
            inc = inc + 256

        self.in_stereo_conv2 = myConv2D(256, 512, 3, 2, 'leaky')
        if use_decoder:
            self.in_decoder_conv2 = myConv2D(128, 256, 3, 2, 'leaky')

        # x16
        self.conv3 = myConv2D(inc + 256, 256, 1, 1, 'leaky')
        self.conv4 = myConv2D(256, 512, 3, 1, 'leaky')
        self.conv5 = myConv2D(512, 256, 1, 1, 'leaky')
        self.conv6 = myConv2D(256, 512, 3, 1, 'leaky')
        self.conv7 = myConv2D(512, 256, 1, 1, 'leaky')
        self.conv8 = myConv2D(256, 512, 3, 1, 'leaky')
        self.out_bbox_conv2 = myConv2D(512, self.output_ch * 3, 1, 1, 'linear', norm=False, bias=True)
        self.out_depth_conv2 = myConv2D(512, 70 * 3, 1, 1, 'linear', norm=False, bias=True)

        # conv 7
        self.downsample2 = myConv2D(256, 512, 3, 2, 'leaky')
        # stereo neck + decoder
        inc = 1024
        if use_decoder:
            inc = inc + 512

        self.in_stereo_conv3 = nn.Sequential(
            myConv2D(256, 512, 3, 2, 'leaky'),
            myConv2D(512, 1024, 3, 2, 'leaky'),
        )
        if use_decoder:
            self.in_decoder_conv3 = nn.Sequential(
                myConv2D(128, 256, 3, 2, 'leaky'),
                myConv2D(256, 512, 3, 2, 'leaky'),
            )
        
        self.conv9 = myConv2D(inc + 512, 512, 1, 1, 'leaky')
        self.conv10 = myConv2D(512, 1024, 3, 1, 'leaky')
        self.conv11 = myConv2D(1024, 512, 1, 1, 'leaky')
        self.conv12 = myConv2D(512, 1024, 3, 1, 'leaky')
        self.conv13 = myConv2D(1024, 512, 1, 1, 'leaky')
        self.conv14 = myConv2D(512, 1024, 3, 1, 'leaky')
        self.out_bbox_conv3 = myConv2D(1024, self.output_ch * 3, 1, 1, 'linear', norm=False, bias=True)
        self.out_depth_conv3 = myConv2D(1024, 70 * 3, 1, 1, 'linear', norm=False, bias=True)
        
    def forward(self, input, base_fea):
        if base_fea is not None:
            base_fea = self.decoder(base_fea)
            d1 = self.in_decoder_conv1(base_fea)
            s1 = self.in_stereo_conv1(input)
            x0 = torch.cat((s1, d1), dim=1)
        else:
            x0 = self.in_stereo_conv1(input)

        #
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        bbox_out1 = self.out_bbox_conv1(x2)
        depth_out1 = self.out_depth_conv1(x2)

        #
        d1 = self.downsample1(x1)
        if base_fea is not None:
            d2 = self.in_decoder_conv2(base_fea)
            s2 = self.in_stereo_conv2(input)
            x0 = torch.cat((s2, d2), dim=1)
        else:
            x0 = self.in_stereo_conv2(input)


        x3 = self.conv3(torch.cat((x0, d1), dim=1))
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        bbox_out2 = self.out_bbox_conv2(x8)
        depth_out2 = self.out_depth_conv2(x8)

        #
        d2 = self.downsample2(x7)
        if base_fea is not None:
            d3 = self.in_decoder_conv3(base_fea)
            s3 = self.in_stereo_conv3(input)
            x0 = torch.cat((s3, d3), dim=1)
        else:
            x0 = self.in_stereo_conv3(input)

        x9 = self.conv9(torch.cat((x0, d2), dim=1))
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        bbox_out3 = self.out_bbox_conv3(x14)
        depth_out3 = self.out_depth_conv3(x14)

        depth_out1_1 = F.softmax(depth_out1[:, :70, :, :], dim=1)
        depth_out1_2 = F.softmax(depth_out1[:, 70:140, :, :], dim=1)
        depth_out1_3 = F.softmax(depth_out1[:, 140:, :, :], dim=1)
        depth_out1_1 = depth_regression(depth_out1_1).unsqueeze(1)
        depth_out1_2 = depth_regression(depth_out1_2).unsqueeze(1)
        depth_out1_3 = depth_regression(depth_out1_3).unsqueeze(1)

        depth_out2_1 = F.softmax(depth_out2[:, :70, :, :], dim=1)
        depth_out2_2 = F.softmax(depth_out2[:, 70:140, :, :], dim=1)
        depth_out2_3 = F.softmax(depth_out2[:, 140:, :, :], dim=1)
        depth_out2_1 = depth_regression(depth_out2_1).unsqueeze(1)
        depth_out2_2 = depth_regression(depth_out2_2).unsqueeze(1)
        depth_out2_3 = depth_regression(depth_out2_3).unsqueeze(1)

        depth_out3_1 = F.softmax(depth_out3[:, :70, :, :], dim=1)
        depth_out3_2 = F.softmax(depth_out3[:, 70:140, :, :], dim=1)
        depth_out3_3 = F.softmax(depth_out3[:, 140:, :, :], dim=1)
        depth_out3_1 = depth_regression(depth_out3_1).unsqueeze(1)
        depth_out3_2 = depth_regression(depth_out3_2).unsqueeze(1)
        depth_out3_3 = depth_regression(depth_out3_3).unsqueeze(1)

        out1 = torch.cat([
            bbox_out1[:, :self.output_ch, :, :], depth_out1_1,
            bbox_out1[:, self.output_ch:self.output_ch*2, :, :], depth_out1_2,
            bbox_out1[:, self.output_ch*2:, :, :], depth_out1_3,
        ], dim=1)

        out2 = torch.cat([
            bbox_out2[:, :self.output_ch, :, :], depth_out2_1,
            bbox_out2[:, self.output_ch:self.output_ch*2, :, :], depth_out2_2,
            bbox_out2[:, self.output_ch*2:, :, :], depth_out2_3,
        ], dim=1)

        out3 = torch.cat([
            bbox_out3[:, :self.output_ch, :, :], depth_out3_1,
            bbox_out3[:, self.output_ch:self.output_ch*2, :, :], depth_out3_2,
            bbox_out3[:, self.output_ch*2:, :, :], depth_out3_3,
        ], dim=1)

        if self.training:
            return [out1, out2, out3]
        else:
            y1 = self.yolo1(out1)
            y2 = self.yolo2(out2)
            y3 = self.yolo3(out3)

            return get_region_boxes([y1, y2, y3])
