"""YOLOv3 loss function


Author: baiyu
"""

import torch
import torch.nn as nn

class YOLOLoss(nn.Module):

    def __init__(self, featuremap=[19, 38, 76]):
        """
        Args:
            featuremap:three detection(yolo) layer feature map size
        """

        super().__init__()
        self.featuremap = featuremap
        self.sum_squared_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, x, target):

        #get target mask
        obj_mask = target[:, :, 4] > 0
        print(obj_mask.shape)


    
from torch.autograd import Variable

target = Variable(torch.Tensor(3, 10647, 85))
predict = Variable(torch.Tensor(3, 10647, 85))

loss_function = YOLOLoss()

loss_function(predict, target)
