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
                       img_size=settings.IMG_SIZE):
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

        objectness_mask = torch.ones(target.size())
        #compute first yolo featuremap objectness score
        first_map = x[:, :self.featuremap[0] * self.featuremap[0] * 3, :]
        offsets = meshgrid(self.featuremap[0])

        stride = self.img_size / self.featuremap[0]
        #compute anchors
        anchor_num = len(anchors)
        anchors = self._generate_anchors(self.anchors[:3], self.featuremap[0])
        anchors[:, :, :2] = (anchors[:, :, :2] + offsets) * stride
        anchors[:, :, 2:] = torch.clamp(anchors[:, :, 2:], min=0, max=self.img_size)
        anchors = Variable(anchors.as_type(target.data))
        
        gt_box = target[:, self.featuremap[0] * self.featuremap[0] * anchor_num, :4].clone()
        ious = bbox_iou(anchors.view(-1, 2), gt_box)




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
        #assume anchors are in the cell center of the grid
        anchor_num = len(anchors)
        anchors = [(0.5, 0.5, aw, ah) for (aw, ah) in anchors]
        anchors = torch.Tensor(anchors)
        anchors = anchors.repeat(grid_size * grid_size * anchor_num, 1).unsqueeze(0)

        return anchors
    
from torch.autograd import Variable

target = Variable(torch.Tensor(3, 22743, 85)).cuda()
target.fill_(1)
predict = Variable(torch.Tensor(3, 22743, 85)).cuda()

loss_function = YOLOLoss().cuda()

loss_function(predict, target)
