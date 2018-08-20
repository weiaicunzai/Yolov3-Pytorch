""" this module is the helper functon for data
augmentation

Author: baiyu
"""

import random

import cv2
import numpy as np

#import utils.plot_tools as plot_tools
#from conf import settings

def normalize_box_params(box, image_shape):
    """a helper function to normalize the
    box params

    Args:
        box: a namedtuple contains box parameters in
            the format of cls_num, x, y, w, h
        image_shape: corresponding image shape (h, w, c)
    
    Returnes:
        Box: a namedtuple of (cls_name, x, y, w, h)
    """
    cls_name, x, y, w, h = box
    height, width = image_shape[:2]

    #normalize
    x *= 1. / width
    w *= 1. / width
    y *= 1. / height
    h *= 1. / height

    Box = type(box)
    return Box(cls_name, x, y, w, h)

def unnormalize_box_params(box, image_shape):
    """a helper function to calculate the box
    parameters

    Args:
        box: a namedtuple contains box parameters in
            the format of cls_num, x, y, w, h
        image_shape: corresponding image shape
    
    Returnes:
        Box: a namedtuple of (cls_name, x, y, w, h)
    """

    cls_name, x, y, w, h = box
    height, width = image_shape[:2]

    #unnormalize
    x *= width
    w *= width
    y *= height
    h *= height

    Box = type(box)
    return Box(cls_name, x, y, w, h)

def resize(image, boxes, image_shape, img_size=608):
    """resize image and boxes to certain
    shape

    Args:
        image: a numpy array(BGR)
        boxes: bounding boxes
        image_shape: two element tuple(width, height)
        img_size: network input image size
    
    Returns:
        resized image and boxes
    """

    origin_shape = image.shape
    x_factor = image_shape[1] / float(origin_shape[1])
    y_factor = image_shape[0] / float(origin_shape[0])

    #resize_image
    if (image.shape[1], image.shape[0]) != image_shape:
        image = cv2.resize(image, image_shape)

    #resize_box
    result = []
    for box in boxes:
        cls_id, x, y, w, h = unnormalize_box_params(box, origin_shape)

        x *= x_factor
        w *= x_factor
        y *= y_factor 
        h *= y_factor

        #clamping the box board, make sure box inside the image, 
        #not on the board
        tl_x = x - w / 2
        tl_y = y - h / 2
        br_x = x + w / 2
        br_y = y + h / 2

        tl_x = min(max(0, tl_x), img_size - 1)
        tl_y = min(max(0, tl_y), img_size - 1)
        br_x = max(min(img_size - 1, br_x), 0)
        br_y = max(min(img_size - 1, br_y), 0)

        w = br_x - tl_x
        h = br_y - tl_y
        x = (br_x + tl_x) / 2
        y = (br_y + tl_y) / 2

        Box = type(box)
        box = Box(cls_id, x, y, w, h)
        result.append(normalize_box_params(box, image.shape))

    return image, result 

def random_crop(image, boxes, probs=0.5, img_size=608, jitter=0.2):
    """randomly crop image, resize image's
    shortest side to img_size * (1 + scale_factor)
    while remain the aspect ratio, then crop a 
    img_size * img_size image

    Args:
        image: a image numpy array(BGR)
        boxes: boxes corresponding to image
        probs: random crop probs
        img_size: network input image size
        jitter: random crop jitter ratio

    Returns:
        (image, boxes): possible flipped image
        and boxes
    """

    if random.random() < probs:
        origin_shape = image.shape
        min_side = min(image.shape[:2])

        #resize the image
        resized_side = int(img_size * (1 + jitter))
        scale_ratio = resized_side / float(min_side)
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)
        
        for index, box in enumerate(boxes):
            cls_id, x, y, w, h = unnormalize_box_params(box, origin_shape)
    
            x *= scale_ratio 
            y *= scale_ratio
            w *= scale_ratio
            h *= scale_ratio

            Box = type(box)
            box = Box(cls_id, x, y, w, h)
            boxes[index] = normalize_box_params(box, image.shape)
    
        #crop the image
        mask = [[0, img_size], [0, img_size]]

        #randomly choose a point as the top left corner to the new image
        random_shift_x = random.randint(0, image.shape[1] - img_size)
        random_shift_y = random.randint(0, image.shape[0] - img_size)
        mask[0][0] = random_shift_x
        mask[0][1] = random_shift_x + img_size
        mask[1][0] = random_shift_y
        mask[1][1] = random_shift_y + img_size

        before_cropped = image.shape
        image = image[mask[1][0] : mask[1][1], mask[0][0] : mask[0][1], :]

        #crop boxes
        result = []
        for box in boxes:
            cls_id, x, y, w, h = unnormalize_box_params(box, before_cropped)

            #get old top_left, bottom_right coordinates
            old_tl_x = x - int(w / 2)
            old_tl_y = y - int(h / 2)
            old_br_x = x + int(w / 2)
            old_br_y = y + int(h / 2)

            #clamp the old box coordinates
            new_tl_x = min(max(old_tl_x, mask[0][0]), mask[0][1])
            new_tl_y = min(max(old_tl_y, mask[1][0]), mask[1][1])
            new_br_x = max(min(old_br_x, mask[0][1]), mask[0][0])
            new_br_y = max(min(old_br_y, mask[1][1]), mask[1][0])


            #get new w, h
            if new_br_x - new_tl_x <= 0:
                continue
            w = new_br_x - new_tl_x
            if new_br_y - new_tl_y <= 0:
                continue
            h = new_br_y - new_tl_y

            #get new x, y
            x = (new_br_x + new_tl_x) / 2 - mask[0][0] 
            y = (new_br_y + new_tl_y) / 2 - mask[1][0]

            Box = type(box)
            box = Box(cls_id, x, y, w, h)
            result.append(normalize_box_params(box, image.shape))

        boxes = result
    return image, boxes

def random_affine(image, boxes, probs=0.5, affine_shift_factor=0.2, affine_scale_factor=0.8):
    """randomly apply affine transformation
    to an image

    Args:
        image: an image numpy array(BGR)
        boxes: boxes corresponding to image
        probs: probs to perform random affine transform
        affine_shift_factor: affine shitf factor, shitf image at most 20% of
                             the original image's height and width
        affine_scale_factor: affine scale factor, scale image at least 80% of
                             the original image's height and width
                             
    Returns:
        (image, boxes): possible flipped image
        and boxes
    """

    if random.random() < probs:
        height, width, _ = image.shape

        shift_x = int(width * random.uniform(0, affine_shift_factor))
        shift_y = int(height * random.uniform(0, affine_shift_factor))
        scale_x = float(random.uniform(affine_scale_factor, 1))
        scale_y = float(random.uniform(affine_scale_factor, 1))

        #affine translation matrix
        trans = np.array([[scale_x, 0, shift_x],
                          [0, scale_y, shift_y]], dtype=np.float32)

        image = cv2.warpAffine(image, trans, (width, height))
    
        #change boxes
        result = []
        for index, box in enumerate(boxes):
            cls_id, x, y, w, h = unnormalize_box_params(box, image.shape)
            x *= scale_x
            w *= scale_x
            y *= scale_y
            h *= scale_y

            x += shift_x
            y += shift_y

            Box = type(box)
            box = Box(cls_id, x, y, w, h)

            # if bounding box is still in the image
            # shift might shitf the bounding box
            # outside of the image
            if (width - (x - w / 2)) > 0 and (height - (y - h / 2)) > 0:
                result.append(normalize_box_params(box, image.shape))
        
        boxes = result
    return image, boxes

def random_horizontal_flip(image, boxes, probs):
    """randomly flip an image left to right

    Args:
        image: a numpy array of a BGR image
        boxes: boxes corresponding to image
        probs: probs of randomly horizontal flip the image

    Returns:
        (image, boxes): possible flipped image
        and boxes
    """

    if random.random() < probs:

        #flip image right to left
        image = cv2.flip(image, 1)

        #flip boxes
        image_shape = image.shape
        for index, box in enumerate(boxes):
            cls_num, x, y, w, h = unnormalize_box_params(box, image_shape)
            x = image_shape[1] - x
            Box = type(box)
            box = Box(cls_num, x, y, w, h)
            boxes[index] = normalize_box_params(box, image_shape)

    return image, boxes

def random_bright(image, probs=0.5, brightness=0.7):
    """randomly brightten an image

    Args:
        image: an image numpy array(BGR)
        probs: probs of randomly bright the image
        brightness: randomly set new brightness to [1 - brightness, 1 + brightness]

    Returns:
        image: randomly brightened image
    """

    if random.random() < probs:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        factor = random.uniform(1. - brightness, 1. + brightness)
        v = v * factor
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

def random_hue(image, probs, hue_factor=0.7):
    """randomly change the hue of an image

    Args:
        image: an image numpy array(BGR)
        probs: probs of perform random hue transoform
        hue_factor: set hue to [1 - hue, 1 + hue] of original hue
    
    Returns:
        images: randomly transformed image
    """

    if random.random() < probs:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        factor = random.uniform(1. - hue_factor, 1. + hue_factor)
        h = h * factor
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

def random_saturation(image, probs=0.5, staturation=0.7):
    """randomly change the saturation of an image

    Args:
        image: an image numpy array(BGR)
        probs: probs of perform staturation transform
        staturation: staturation factor, set new staturation to 
                     [1 - staturation, 1 + staturation]
                
    Returns:
        image: transformed image
    """

    if random.random() < probs:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v= cv2.split(hsv)
        factor = random.uniform(1. - staturation,
                                1. + staturation)
        s = s * factor
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image

def random_gaussian_blur(image, probs=0.5):
    """ randomly blurs an image using a Gaussian filter.

    Args:
        image: an image numpy array(BGR)
        probs: probs of perform gaussian_blur
    
    Returnes:
        image: augumentated image
    """

    if random.random() < probs:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return image

def plot_image_bbox(image, boxes):
    """plot an image with its according boxes
    a useful test tool for 

    Args:
        images: an numpy array [r, g, b] format
        boxes: a list contains box
    """

    for box in boxes:
        shape = image.shape
        cls_index, x, y, w, h = unnormalize_box_params(box, shape)

        #draw bbox
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        cv2.rectangle(image, top_left, bottom_right, settings.COLOR[cls_index])

        #draw Text background rectangle
        text_size, baseline = cv2.getTextSize(settings.CLASSES[int(cls_index)], 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (top_left[0], top_left[1] - text_size[1]),
                             (top_left[0] + text_size[0], top_left[1]),
                             settings.COLOR[cls_index],
                             -1)
        
        #draw text
        cv2.putText(image, settings.CLASSES[int(cls_index)], 
                           top_left,
                           cv2.FONT_HERSHEY_DUPLEX,
                           0.4,
                           settings.COLOR[int(cls_index)],
                           1,
                           8,
                           )
    cv2.imshow('test', image)
    cv2.waitKey(0)

#def plot_compare(image, target, boxes):
#    """compare target and boxes on the image
#
#    Args:
#        target: a target is a 7 * 7 * 30 numpy
#        array
#        image: a 448 * 448 * 3 numpy array
#        boxes: a namedtuple
#    """
#
#    plot_image_bbox(image, boxes)
#    row_num, col_num = target.shape[:2]
#    for row in range(row_num):
#        for col in range(col_num):
#            value = target[row, col, :]
#
#            #if this cell does not contain object
#            if not value[9]:
#                continue
#            
#            cls_id = value[10:].tolist().index(1)
#            cv2.rectangle(image, 
#                          (settings.IMG_SIZE // settings.S * col, settings.IMG_SIZE // settings.S * row),
#                          (settings.IMG_SIZE // settings.S * (col + 1), settings.IMG_SIZE // settings.S * (row + 1)),
#                          settings.COLOR[int(cls_id)],
#                          -1)
#
#            x, y, w, h = value[5:9]
#            x *= image.shape[1]
#            w *= image.shape[1]
#            y *= image.shape[0]
#            h *= image.shape[0]
#
#            #draw bbox
#            top_left = (int(x - w / 2), int(y - h / 2))
#            bottom_right = (int(x + w / 2), int(y + h / 2))
#            cv2.rectangle(image, top_left, bottom_right, settings.COLOR[int(cls_id)])
#
#            #draw Text background rectangle
#            text_size, baseline = cv2.getTextSize(settings.CLASSES[int(cls_id)], 
#                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
#            cv2.rectangle(image, (top_left[0], top_left[1] - text_size[1]),
#                                 (top_left[0] + text_size[0], top_left[1]),
#                                 settings.COLOR[int(cls_id)],
#                                 -1)
#
#            #draw text
#            cv2.putText(image, settings.CLASSES[int(cls_id)], 
#                               top_left,
#                               cv2.FONT_HERSHEY_DUPLEX,
#                               0.4,
#                               settings.COLOR[int(cls_id)],
#                               1,
#                               8,
#                               )
#    cv2.imshow('test', image)
#    cv2.waitKey(0)