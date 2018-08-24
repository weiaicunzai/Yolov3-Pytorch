
import torch

def bbox_iou(pred_box, target_box):
    """comput iou of pred_box and target_bnox

    Args:
        pred_box: a pytorch tensor, variable, shape(-1, 4),
                  x,y,w,h(center x, y and bbox weight, height, w,h)
        target_box: same as pred_box
    
    Returns:
        iou: intersection of union of these boxes
    """

    print(pred_box.shape, target_box.shape)

#    p_tlx = pred_box[:, 0] - pred_box[:, 2] / 2
#    p_tly = pred_box[:, 1] - pred_box[:, 3] / 2
#    p_brx = pred_box[:, 0] + pred_box[:, 2] / 2
#    p_bry = pred_box[:, 1] + pred_box[:, 3] / 2
#

    print(pred_box)
    p_tlx = pred_box[:, 0]
    p_tly = pred_box[:, 1]
    p_brx = pred_box[:, 2]
    p_bry = pred_box[:, 3]

    t_tlx = target_box[:, 0]
    t_tly = target_box[:, 1]
    t_brx = target_box[:, 2]
    t_bry = target_box[:, 3]


    #get the intersection coordinates of bbox
    inter_tlx = torch.max(p_tlx, t_tlx)
    inter_tly = torch.max(p_tly, t_tly)
    inter_brx = torch.min(p_brx, t_brx)
    inter_bry = torch.min(p_bry, t_bry)

    #get the intersection area, puls 1 means add the border
    inter_area = torch.mul(torch.clamp(inter_brx - inter_tlx + 1, min=0),
                           torch.clamp(inter_bry - inter_tly + 1, min=0))
    
    #get bbox area
    pred_area = (p_brx - p_tlx + 1) * (p_bry - p_tly + 1)
    #for i in range(3249):
    #    print(pred_area[i])
    target_area = (t_brx - t_tlx + 1) * (t_bry - t_tly + 1)

    iou = inter_area / (pred_area + target_area - inter_area + 1e-8)
    return iou

def meshgrid(grid_size, anchor_num=3):
    """similar to numpy or tensorflow  meshgrid
    Args:
        grid_size: feature map grid size
        anchor_num: anchor number per cell, default to 3
    
    Returns:
        x_y_offset: shape size (1, grid_size * grid_size * anchor_num, 2)
    """

    grid_w = grid_size
    grid_h = grid_size
    cell_x = torch.linspace(0, grid_w - 1, grid_w)
    cell_y = torch.linspace(0, grid_h - 1, grid_h)
    x_offsets = cell_x.repeat(grid_h, 1).contiguous().view(-1, 1)
    y_offsets = cell_y.repeat(grid_w, 1).t().contiguous().view(-1, 1)
    x_y_offset = torch.cat((x_offsets, y_offsets), 1).repeat(1, anchor_num)
    x_y_offset = x_y_offset.view(-1, 2).unsqueeze(0)

    return x_y_offset
    #x_y_offset = Variable(x_y_offset)