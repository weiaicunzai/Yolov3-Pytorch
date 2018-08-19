
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

class YOLOLayer(nn.Module):

    """YOLOv3 object detection model"""
    def __init__(self, anchors, mask, classes_num, image_size):
        super().__init__()
        mask = [int(m) for m in mask.split(',')]
        anchors = anchors.split(",")
        anchors = [(int(anchors[i]), int(anchors[i + 1])) for i in range(0, len(anchors), 2)]
        self.anchors = [anchors[i] for i in mask]
        self.classes_num = int(classes_num)
        self.image_size = int(image_size)
    
    def forward(self, x):
        batch_size, channels, grid_h, grid_w = x.size()
        anchor_num = len(self.anchors)
        bbox_length = 5 + self.classes_num
        stride = int(self.image_size / grid_h)

        #flatten predected bounding boxes
        #"""In our experiments with COCO [10] we predict 3 boxes at each 
        #scale so the tensor is N × N × [3 ∗ (4 + 1 + 80)] for the 4 
        #bounding box offsets, 1 objectness prediction, and 80 class 
        #predictions."""
        x = x.view(batch_size, channels, grid_h * grid_w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, grid_h * grid_w * anchor_num, bbox_length)

        cell_x = torch.linspace(0, grid_w - 1, grid_w)
        cell_y = torch.linspace(0, grid_h - 1, grid_h)
        x_offsets = cell_x.repeat(grid_h, 1).contiguous().view(-1, 1)
        y_offsets = cell_y.repeat(grid_w, 1).t().contiguous().view(-1, 1)
        x_y_offset = torch.cat((x_offsets, y_offsets), 1).repeat(1, anchor_num)
        x_y_offset = x_y_offset.view(-1, 2).unsqueeze(0).type_as(x)
        
        scaled_anchors = [(int(aw / stride), int(ah / stride)) for (aw, ah) in anchors]
        sacled_anchors = scaled_anchors.repeat(grid_w * grid_h, 1).view(-1, 2).unsqueeze(0)
        scaled_anchors = type(x)(scaled_anchors)
        #x_offsets = cell_x.repeat(grid_h, 1).repeat(batch_size * anchor_num, 1, 1).view(x.shape) 
        #y_offsets = cell_y.repeat(grid_w, 1).t().repeat(batch_size * anchor_num, 1, 1).view(x.shape) 
        #image by (c x , c y ) and the bounding box prior has width and
        #height p w , p h , then the predictions correspond to:
        #b x = σ(t x ) + c x
        #b y = σ(t y ) + c y
        #b w = p w e t w
        #b h = p h e t h"""

        #"""We predict the center coordinates of the box relative to the 
        #location of filter application using a sigmoid function."""
        x[:, :, :2] = F.sigmoid(x[:, :, :2]) + x_y_offset
        x[:, :, 2:4] = torch.exp(x[:, :, 2:4]) * scaled_anchors
        x[:, :, :4] *= stride

        #"""YOLOv3 predicts an objectness score for each bounding
        #box using logistic regression."""
        x[:, :, 4] = F.sigmoid(x[:, :, 4])

        #"""We do not use a softmax as we have found it is unnecessary for 
        #good performance, instead we simply use independent logistic 
        #classifiers."""
        x[:, :, 5:] = F.sigmoid(x[:, :, 5:])

        return x

def create_modules(blocks):
    """ Constructing network architechture

    Args:
        blocks: a python list object contains 
                cfg file blocks information
    
    Returns:
        module_list: a module list containing modules
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
            module_list.append(empty_block())
        
        elif block['type'] == 'yolo':
            module_list.append(YOLOLayer(
                block['anchors'],
                block['mask'],
                block['classes'],
                net_info['width']))

        output_channels.append(pre_output_channels)
    print(len(module_list), len(blocks))
    return module_list

class YOLOV3(nn.Module):
    def __init__(self, blocks, module_list):
        self.net_info = blocks[0]
        self.blocks = blocks
        self.module_list = module_list
    
    def forward(self, x):

        outputs = []
        for i in range(1, len(self.blocks)):
            if blocks[i]['type'] == 'convolutional':
                x = self.module_list[i - 1](x)
                outputs[i - 1] 


import cProfile

cProfile.runctx("create_modules(utils.parse_cfg(settings.CFG_PATH))", globals(), None)
