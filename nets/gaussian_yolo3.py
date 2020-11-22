# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:gaussian_yolo3.py
# software: PyCharm

import keras
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

