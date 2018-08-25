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
                       anchor_num_per_layer=3,
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
        self.sigmoid = nn.Sigmoid()
        self.anchors = anchors
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh
        self.anchor_num_per_layer = 3
    
    def forward(self, x, target):
        """ compute network loss, x and target input order can't be swithed, since
        we deal x and target differently(x uses sigmoid, target doesnt)

        Args:
            target: (x, y, w, h, c, classprobs......)
            x, y are scaled x, y coordinate x =  x / img_width, y = y / img_height
            w, h are scaled w, h length  w = w / img_width, h = h / img_width

        Returns:
            loss of the network
        """

        #get target mask
        obj_mask = target[:, :, 4] > 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(target)

        #bbox_loss = self.sum_squared_loss(x[obj_mask])
        no_obj_mask = target[:, :, 4] == 0
        no_obj_mask = no_obj_mask.unsqueeze(-1).expand_as(target)

        pred_box = x[obj_mask].view(-1, 85)
        target_box = target[obj_mask].view(-1, 85)

        #"""During training we use sum of squared error loss."""
        loss_bbox_xy = self.bce_loss(self.sigmoid(pred_box[:, :2]), target_box[:, :2])
        loss_bbox_wh = self.bce_loss(pred_box[:, 2:4], target_box[:, 2:4])

        loss_objectness = self._objectness_loss(x, target)

       # objectness_mask = torch.ones(x.size(0), self.featuremap[0] ** 2 * 3, x.size(2))
       # #compute first yolo featuremap objectness score
       # #offsets = meshgrid(self.featuremap[0])

       # #print(offsets.shape)

       # #for i in range(self.featuremap[0] * self.featuremap[0] * 3):
       # #    print(offsets[0, i, :] * 32)

       # #compute anchors
       # anchor_num = len(self.anchors[:3])
       # anchors = self._generate_scaled_anchors(self.anchors[:3], self.featuremap[0])
       # #anchors[:, :, :2] = anchors[:, :, :2] + offsets * stride
       # #for i in range(self.featuremap[0] ** 2 * 3):
       # #    print(i, anchors[0, i, :])
       # #for i in range(self.featuremap[0] ** 2):
       # #    print(anchors[0, i, :])
       # anchors[:, :, 2:] = torch.clamp(anchors[:, :, 2:], min=0, max=self.img_size)
       # anchors = Variable(anchors.type_as(target.data))
       # 
       # gt_box = target[:, :self.featuremap[0] * self.featuremap[0] * anchor_num, :4].clone()
       # gt_box[:, :, :2] = 0
       # anchors = anchors.repeat(target.size(0), 1, 1)



       # #for i in range(self.featuremap[0] ** 2 * 3):
       # #    print(gt_box[0, i, :])
       # ious = bbox_iou(anchors.view(-1, 4), gt_box.view(-1, 4))
       # ious = ious.view(-1, self.featuremap[0] ** 2 * anchor_num)
       # objectness_mask[ious > self.ignore_thresh, :] = 0
 
       # print("-" * 30)
       # for batch_size in range(x.size(0)):
       #     for cell_id in range(self.featuremap[0] ** 2):
       #         best_iou = torch.max(ious[batch_size, cell_id * 3 : cell_id * 3 + 3], 0)[1]
       #         objectness_mask[batch_size, 3 * cell_id + best_iou, :] = 1

       # objectness_target = target[:, :self.featuremap[0] ** 2 * 3, :] * objectness_mask
       # objectness_predict = x[:, :self.featuremap[0] ** 2 * 3, :] * objectness_mask
       # print(objectness_predict.shape)
       # loss_objectness = self.bce_loss(objectness_predict, objectness_target)

        #"""During training we use binary cross-entropy loss for the class
        #predictions."""
        loss_classes = self.bce_loss(self.sigmoid(pred_box[:, 4]), target_box[:, 4])
        #print(self._objectness_loss(x, target))
        return loss_bbox_xy + loss_bbox_wh  + loss_objectness + loss_classes

    def _objectness_loss(self, x, target):
        """compute objectness_loss for all the yolo layer output featuremap

        Args:
            x: concatenated predicted featuremap
            target: gt training label
        (x and target input order cant switch) 
        Returns:
            loss: objectness loss for the whole network
        """

        loss = 0
        first_index = 0
        anchor_num = 3
        for index, feature_size in enumerate(self.featuremap):
            feature_length = feature_size ** 2 * anchor_num
            objectness_mask = torch.ones(x.size(0), feature_length, x.size(2))

            scaled_anchors = self._generate_scaled_anchors(self.anchors[index:(index + 1) * 3], 
                                                           feature_size)
            scaled_anchors = torch.clamp(scaled_anchors, min=0, max=self.img_size)
            scaled_anchors = Variable(scaled_anchors.repeat(x.size(0), 1, 1).type_as(x))

            gt_box = target[:, first_index:feature_length, :4].clone()

            #scale the target, multiply the featuremap size
            gt_box = gt_box * feature_size
            gt_box[:, :, :2] = 0

            ious = bbox_iou(scaled_anchors.view(-1, 4), gt_box.view(-1, 4), align=True)

            #change shape[batch_size * feature_size ** 2 * anchor_num] to 
            #shape [batch_size, feature_size ** 2 * anchor_num]
            ious = ious.view(-1, feature_length)

            #"""If the bounding box prioris not the best but does overlap a ground 
            #truth object by more than some threshold we ignore the prediction, 
            #following [17]."""
            objectness_mask[ious > self.ignore_thresh, :] = 0
            objectness_mask = self._ignore_anchor(x.size(0), feature_size, objectness_mask, ious)

            objectness_pred = x[:, first_index:first_index + feature_length, :] * objectness_mask
            objectness_target = target[: first_index:first_index + feature_length, :] * objectness_mask

            objectness_pred = self.sigmoid(objectness_pred[:, first_index:first_index + feature_length, 4])
            objectness_target = objectness_target[:, first_index:first_index + feature_length, 4]
            loss += self.bce_loss(objectness_pred, objectness_target)

            first_index = feature_length

        return loss

    def _ignore_anchor(self, batch_size, feature_size, objectness_mask, ious):
        """ ignore objectness score of anchors overlap with gt bbx over
            threshold
        
        Args:
            batch_size: input data batch_size
            feature_size: yolo layer output feature size
            objectness: shape(batch_size, feature_size ** 2 * anchor_num, 85)
            ious: shape(batch_size, feature_size ** 2 * anchor_num)
        
        Returns:
            objectness_mask: objectness_mask with best iou value sets to 1
        """

        anchor_num = self.anchor_num_per_layer
        for b_index in range(batch_size):
            for cell_index in range(feature_size ** 2):

                #get the best iou index of 3 anchors
                best_iou = ious[b_index, cell_index * anchor_num : (cell_index + 1) * anchor_num][1]
                objectness_mask[b_index, anchor_num * cell_index + best_iou, :] = 1

        return objectness_mask

    def _generate_scaled_anchors(self, anchors, grid_size):
        """ generate scaled anchors for computing iou
        Args:
            anchors: anchors for given featuremap needed to compute
            grid_size: feature map grid_size
        
        Returns: a shape (1, grid_size * grid_size * anchors_num, 4)
                 size tensor
        """    
        stride = self.img_size / grid_size
        anchors = [(0, 0, aw / stride, ah / stride) for (aw, ah) in anchors]
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
