"""YOLOv3 loss function


Author: baiyu
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from conf import settings
from utils import bbox_iou, meshgrid

class YOLOLoss(nn.Module):

    def __init__(self, featuremap=settings.FEATUREMAP, 
                       anchors=settings.ANCHORS,
                       img_size=settings.IMG_SIZE,
                       ignore_thresh=0.7):
        """
        Args:
            featuremap:three detection(yolo) layer feature map size
            anchors: all 9 anchors used in yolov3
        """

        super().__init__()
        self.featuremap = featuremap
        self.sum_squared_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss()
        self.anchors = anchors
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh
    
    def forward(self, x, target):

        #get target mask
        obj_mask = target[:, :, 4] > 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(target)

        #bbox_loss = self.sum_squared_loss(x[obj_mask])
        no_obj_mask = target[:, :, 4] == 0
        no_obj_mask = no_obj_mask.unsqueeze(-1).expand_as(target)

        pred_box = x[obj_mask].view(-1, 85)
        target_box = target[obj_mask].view(-1, 85)

        #"""During training we use sum of squared error loss."""
        loss_bbox_xy = self.bce_loss(pred_box[:, :2], target_box[:, :2])
        loss_bbox_wh = self.bce_loss(pred_box[:, 2:4], target_box[:, 2:4])

        objectness_mask = torch.ones(x.size(0), self.featuremap[0] ** 2 * 3, x.size(2))
        #compute first yolo featuremap objectness score
        first_map = x[:, :self.featuremap[0] * self.featuremap[0] * 3, :]
        offsets = meshgrid(self.featuremap[0])

        print(offsets.shape)

        #for i in range(self.featuremap[0] * self.featuremap[0] * 3):
        #    print(offsets[0, i, :] * 32)

        stride = self.img_size / self.featuremap[0]
        #compute anchors
        anchor_num = len(self.anchors[:3])
        anchors = self._generate_anchors(self.anchors[:3], self.featuremap[0])
        #anchors[:, :, :2] = anchors[:, :, :2] + offsets * stride
        #for i in range(self.featuremap[0] ** 2 * 3):
        #    print(i, anchors[0, i, :])
        #for i in range(self.featuremap[0] ** 2):
        #    print(anchors[0, i, :])
        print(self.img_size)
        anchors[:, :, 2:] = torch.clamp(anchors[:, :, 2:], min=0, max=self.img_size)
        anchors = Variable(anchors.type_as(target.data))
        
        gt_box = target[:, :self.featuremap[0] * self.featuremap[0] * anchor_num, :4].clone()
        gt_box[:, :, :2] = 0
        anchors = anchors.repeat(target.size(0), 1, 1)

        print(anchors.shape, ".....")
        print(gt_box.shape, ".....")


        #for i in range(self.featuremap[0] ** 2 * 3):
        #    print(gt_box[0, i, :])
        ious = bbox_iou(anchors.view(-1, 4), gt_box.view(-1, 4))
        print(ious.shape)

        r = ious.data > self.ignore_thresh
        r = r.long().cuda()
        objectness_mask = objectness_mask.long()
        print(type(objectness_mask))
        print(r.shape)
        
        objectness_mask[:, r, :] = 0



        #for i in range(len(ious)):
        #    print(ious[i])



        #"""During training we use binary cross-entropy loss for the class
        #predictions."""
        loss_classes = self.bce_loss(pred_box[:, 4], target_box[:, 4])


        print(obj_mask.shape)
        print(obj_mask)
        print(target[obj_mask].shape)

        return loss_bbox_xy + loss_bbox_wh 

    @staticmethod
    def _generate_anchors(anchors, grid_size):
        """ generate anchors for computing iou
        Args:
            anchors: anchors for given featuremap needed to compute
            grid_size: grid_size
        
        Returns: a shape (1, grid_size * grid_size * anchors_num, 4)
                 size tensor
        """    
        anchor_num = len(anchors)
        anchors = [(0, 0, aw, ah) for (aw, ah) in anchors]
        anchors = torch.Tensor(anchors)
        anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
        
        return anchors
    
from torch.autograd import Variable

target = Variable(torch.Tensor(3, 22743, 85))
target.fill_(1)
#predict = Variable(torch.Tensor(3, 22743, 85)).cuda()
predict = target.clone()

loss_function = YOLOLoss()

loss_function(predict, target)
