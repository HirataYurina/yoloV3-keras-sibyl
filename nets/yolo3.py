# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:mish.py
# software: PyCharm

import keras as keras
import tensorflow as tf
from keras import backend as K
from keras.layers import UpSampling2D, Concatenate
from keras.models import Model
from nets.darknet53 import darknet_body
from utils.utils import compose

from nets.darknet53 import DarknetConv2D_BN_Leaky
from nets.darknet53 import DarknetConv2D


# ---------------------------------------------------
#   backbone -> make_last_layers
# ---------------------------------------------------
def make_last_layers(x, num_filters, out_filters):
    """we use a new network for performing feature extraction.
       And this new network is hybrid approach between the network
       used in YOLOv2, DarkNet19 and residual network.

       We use successful 3*3 and 1*1 convolutional layer, also combined with shortcut.
       This network is named DarkNet53 and is much more powerful than DarkNet19.
       It is still more efficient than ResNet-101 and ResNet152.

       That means DarkNet53 better utilizes GPU.
       ResNets have too many layers and some layers are idle.
    """

    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # outputs
    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)

    # do not use activation
    y = DarknetConv2D(out_filters, (1, 1))(y)

    return x, y


# ----------------------------------------------------------------------------------
#   whole YOLOv3 model
# ----------------------------------------------------------------------------------
def yolo_body(inputs, num_anchors, num_classes):

    # Darknet53
    feat1, feat2, feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # fist stage's outputs
    # y1 = (batch_size, 13, 13 , 3, 85)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    # feature fusion
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat2])

    # second stage's outputs
    # y2 = (batch_size, 26, 26, 3, 85)
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat1])

    # third stage's outputs
    # y3 = (batch_size, 52, 52, 3, 85)
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


# --------------------------------------------------------------------------------------------
# decode raw prediction to confidence, class probability and bounding boxes location
# --------------------------------------------------------------------------------------------
def yolo_head(feats,
              anchors,
              num_classes,
              input_shape,
              calc_loss=False):
    """decode prediction"""

    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # (13, 13, 1, 2)
    # feature:(batch, 13, 13, 3, 85)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size, 13, 13, 3, 85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # decode prediction
    # normalize (0, 1)
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # when training, return these results.
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------
#   correct boxes
#   inputs coordinate -> original image coordinate
# ---------------------------------------------------
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# ------------------------------------------------------#
#   boxes_score = boxes_confidence * boxes_class_probs
# ------------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # wrap decoding functions
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# ------------------------------------------------------------------------- #
#   raw_outputs -> decode(yolo_head) -> correct boxes -> nms -> final outputs
# ------------------------------------------------------------------------- #
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):

    num_layers = len(yolo_outputs)
    # stage3 -> [6, 7, 8]
    # stage2 -> [3, 4, 5]
    # stage1 -> [0, 1, 2]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    # perform decode for every layer
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    # discard boxes that have low score
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []

    # perform nms for every class
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # tf.image.non_max_suppression
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


if __name__ == '__main__':
    yolo = yolo_body(keras.Input(shape=(416, 416, 3)), 3, 80)

    # yolo.summary()
    yolo.load_weights('../logs/yolo3_weights.h5')

    from PIL import Image
    from utils.utils import letterbox_image
    import numpy as np

    street = Image.open('../img/street.jpg')
    print(street.size)

    street = letterbox_image(street, (416, 416))
    street = np.expand_dims(np.array(street), axis=0)
    street = street / 255.0
    # street = tf.convert_to_tensor(street, dtype=tf.float32)

    with K.get_session() as sess:
        print(sess.run(yolo.output, {yolo.input: street, K.learning_phase(): 0}))
