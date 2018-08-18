
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from conf import settings
#from torch.autograd import Variable


def conv_block(input_channels,
               output_channels,
               stride,
               size=3,
               pad=1,
               use_bn=True,
               activation='leaky'):
    
    """define a convolution block

    Args:
        input_channels(string): input feature map channel
        output_channels(string): filter number 
        stride(string): filter stride
        size(string): filter size
        pad(string): pad value
        use_bn: if use batch_normalization
        activation: activation function type

    Returns:
        a conv block(nn.Sequential)
    """

    conv = nn.Sequential()
    conv.add_module("conv", nn.Conv2d(int(input_channels), 
                                      int(output_channels), 
                                      int(size),
                                      stride=int(stride),
                                      padding=int(pad),
                                      bias=False))
    if bool(use_bn):
        conv.add_module('batch_normalize', nn.BatchNorm2d(int(output_channels)))    
    
    if activation == 'leaky':
        conv.add_module('leaky relu', nn.LeakyReLU(inplace=True))

    return conv

def upsample_block(scale_factor, mode='bilinear'):
    """ upsample the feature map by a factor of scale_factor

    Args:
        sacle_facotr: upsample factor
        mode: one of nearest | linear | bilinear | trilinear
    
    Returns:
        a upsample block(nn.Sequential)
    """

    upsample = nn.Sequential()
    upsample.add_module('upsample', nn.Upsample(scale_factor=int(scale_factor), mode=mode))
    return upsample

def empty_block():
    """a placeholder for route and shortcut block
    just a programming trick

    Returnes:
        an empty block
    """

    return nn.Sequential()

class YOLO(nn.Module):

    """YOLOv3 object detection model"""
    def __init__(self, anchors, classes_num):
        super().__init__()
        self.anchors = anchors
        self.classes_num = classes_num
    
    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        grid_h = x.size(2)
        grid_w = x.size(3)
        anchor_num = len(self.anchors)
        bbox_length = 5 + self.classes_num


        #flatten predected bounding boxes
        #"""In our experiments with COCO [10] we predict 3 boxes at each 
        #scale so the tensor is N × N × [3 ∗ (4 + 1 + 80)] for the 4 
        #bounding box offsets, 1 objectness prediction, and 80 class 
        #predictions."""
        x = x.view(batch_size, channels, grid_h * grid_w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, grid_h * grid_w * anchor_num, bbox_length)

        #"""We predict the center coordinates of the box relative to the 
        #location of filter application using a sigmoid function."""

        #"""YOLOv3 predicts an objectness score for each bounding
        #box using logistic regression."""
        x = F.sigmoid(x[:, :, 0])
        x = F.sigmoid(x[:, :, 1])
        x = F.sigmoid(x[:, :, 4])





def create_modules(blocks):
    """ Constructing network architechture

    Args:
        blocks: cfg file blocks information
    
    Returns:
        yolo modulelist
    """

    #the first block is net information
    net_info = blocks[0]
    pre_output_channels = 3
    output_channels = []

    module_list = nn.ModuleList()

    count = 0
    for index, block in enumerate(blocks):

        #convolutional block
        if block['type'] == 'convolutional':
            module_list.append(conv_block(
                pre_output_channels,
                block['filters'],
                block['stride'],
                size=block['size'],
                pad=block['pad'],
                activation=block['activation'],
                use_bn=block.get('batch_normalize', False)
            ))
            pre_output_channels = int(block['filters'])

        elif block['type'] == 'upsample':
            module_list.append(upsample_block(
                block['stride']          
            ))
        
        elif block['type'] == 'route':
            layer_index = block['layers'].split(',')

            #when length equals to 1, layers attr is always negative
            if(len(layer_index) == 1):
                pre_output_channels = output_channels[int(layer_index[0])]
            
            if(len(layer_index) == 2):
                pre_output_channels = (output_channels[int(layer_index[0])] + 
                                      output_channels[int(layer_index[1])])

            module_list.append(empty_block())

        elif block['type'] == 'shortcut':
            count += 1
            module_list.append(empty_block())
        
        elif block['type'] == 'yolo':
            module_list.append(empty_block())

        output_channels.append(pre_output_channels)

    print(count)

import cProfile

cProfile.runctx("create_modules(utils.parse_cfg(settings.CFG_PATH))", globals(), None)
