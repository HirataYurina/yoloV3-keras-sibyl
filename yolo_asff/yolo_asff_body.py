# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:yolo_asff_body.py
# software: PyCharm

from nets.darknet53 import darknet_asff_body
import keras as keras
from nets.darknet53 import DarknetConv2D_BN_Leaky


def yolo_asff_body(inputs, num_anchors, num_classes):

    # feat1 (52, 52, 128)
    # feat2 (26, 26, 256)
    # feat3 (13, 13, 512)
    feat1, feat2, feat3 = darknet_asff_body(inputs)

    y1 = DarknetConv2D_BN_Leaky(128 * 2, 3, padding='same')(feat1)
    y1 = DarknetConv2D_BN_Leaky(num_anchors * (num_classes + 5), 1)(y1)
    y2 = DarknetConv2D_BN_Leaky(256 * 2, 3, padding='same')(feat2)
    y2 = DarknetConv2D_BN_Leaky(num_anchors * (num_classes + 5), 1)(y2)
    y3 = DarknetConv2D_BN_Leaky(512 * 2, 3, padding='same')(feat3)
    y3 = DarknetConv2D_BN_Leaky(num_anchors * (num_classes + 5), 1)(y3)
    model = keras.Model(inputs, [y3, y2, y1])

    return model


if __name__ == '__main__':
    inputs_ = keras.Input(shape=(416, 416, 3))
    model_ = yolo_asff_body(inputs_, 3, 6)
    model_.summary()
