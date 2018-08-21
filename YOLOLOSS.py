"""YOLOv3 loss function


Author: baiyu
"""

import torch
import torch.nn as nn
from conf import settings

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
        print(pred_box.shape)
        loss_bbox = self.sum_squared_loss(pred_box[:, :4], target_box[:, :4])

        #compute first yolo featuremap objectness score


        print(obj_mask.shape)
        print(obj_mask)
        print(target[obj_mask].shape)

        return loss_bbox

    
from torch.autograd import Variable

target = Variable(torch.Tensor(3, 22743, 85)).cuda()
target.fill_(1)
predict = Variable(torch.Tensor(3, 22743, 85)).cuda()

loss_function = YOLOLoss().cuda()

loss_function(predict, target)
