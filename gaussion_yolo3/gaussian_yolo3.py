# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:gaussian_yolo3.py
# software: PyCharm

import keras.backend as K
from nets.yolo3 import make_last_layers
from keras.layers import UpSampling2D, Concatenate
from keras.models import Model
from nets.darknet53 import darknet_body
from utils.utils import compose

from nets.darknet53 import DarknetConv2D_BN_Leaky


# #########################
#   gaussian YOLOv3 model
# #########################
def yolo_body(inputs, num_anchors, num_classes):
    """Gaussian YOLOv3
    Gaussian YOLOv3 can predict a bounding box distribution.
    We model the distribution by a gaussian distribution.
    p(y|x) = N(μ, δ)
    We can use δ to represent the uncertainty about bounding box.

    we change [dx, dy, dw, dh] to [μ(dx), μ(dy), μ(dw), μ(dh), δ(dx), δ(dy), δ(dw), δ(dh)]

    Args:
        inputs:      [batch, height, width, channels]
        num_anchors: usually be 3
        num_classes: number of your classes

    Returns:
        model

    """

    # Darknet53
    feat1, feat2, feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # fist stage's outputs
    # y1 = (batch_size, 13, 13 , 3, 89)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 9))

    # feature fusion
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat2])

    # second stage's outputs
    # y2 = (batch_size, 26, 26, 3, 89)
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 9))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, feat1])

    # third stage's outputs
    # y3 = (batch_size, 52, 52, 3, 89)
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 9))

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
    # feature:(batch, 13, 13, 3, 89)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size, 13, 13, 3, 89)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 9])

    # decode prediction
    # normalize (0, 1)
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 8:9])
    box_class_probs = K.sigmoid(feats[..., 9:])

    # when training, return these results.
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs