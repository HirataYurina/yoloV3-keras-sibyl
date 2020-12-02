# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:mish.py
# software: PyCharm

from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose
from nets.mish import Mish
import keras.backend as K
import keras.layers as layers
import tensorflow as tf


# --------------------------------------------------
# convolution without BN and activation
# --------------------------------------------------
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------
# Convolution + BatchNormalization + LeakyReLU
# ---------------------------------------------------
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------
# Convolution + BatchNormalization + Mish
# ---------------------------------------------------
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


# ---------------------------------------------------
# residual block
# ---------------------------------------------------
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1, 1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(y)
        x = Add()([x, y])
    return x


# ---------------------------------------------------
# CSP Block
# https://https://arxiv.org/abs/1911.11929
# yoloV4 uses CSPNet to be the backbone and this can
# reduce computation and enhance backbone performance.
# -----------------------------------------------------
def csp_resblock_body(x, num_filters, num_blocks, all_narrow=True):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN

    Args:
        x: inputs feature [batch_size, m, n, c]
        num_filters: a scalar
        num_blocks: a scalar
        all_narrow: boolean

    Returns:
        results: feature [batch_size, m, n, c]
    """
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    # pre convolution
    preconv = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(x)

    # use 1*1 filter size to group the feature
    main_conv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv)
    short_conv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Mish(num_filters // 2, (1, 1))(main_conv)
        y = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3))(y)
        main_conv = Add()([main_conv, y])

    # post convolution
    post_conv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(main_conv)

    # cross spatial partial concatenate
    csp_concat = Concatenate()([post_conv, short_conv])
    results = DarknetConv2D_BN_Mish(num_filters, (1, 1))(csp_concat)
    return results


# ---------------------------------------------------
#   body of darknet53
# ---------------------------------------------------
def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3


# ---------------------------------------------------
#   body of CSPDarknet53
# ---------------------------------------------------
def csp_darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)
    x = csp_resblock_body(x, 64, 1, False)
    x = csp_resblock_body(x, 128, 2)
    x = csp_resblock_body(x, 256, 8)
    feat1 = x
    x = csp_resblock_body(x, 512, 8)
    feat2 = x
    x = csp_resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3


class WeightsNormalize(layers.Layer):

    def __init__(self, **kwargs):
        super(WeightsNormalize, self).__init__(**kwargs)

        self.conv = layers.Conv2D(3, kernel_size=1)
        self.concat = layers.Concatenate(axis=-1)
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.concat(inputs)
        x = self.conv(x)
        x = self.softmax(x)

        results = [x[..., 0:1],
                   x[..., 1:2],
                   x[..., 2:3]]

        return results


class FeatureFusion(layers.Layer):

    def __init__(self, **kwargs):
        super(FeatureFusion, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        results = tf.multiply(inputs[0], inputs[1]) + \
                  tf.multiply(inputs[2], inputs[3]) + \
                  tf.multiply(inputs[4], inputs[5])

        return results


def make_five_layers(x, num_filters):
    # five convolutional layers
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    return x


# ---------------------------------------------------
#   body of darknet53
#   ASFF + Darknet53
# ---------------------------------------------------
def darknet_asff_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x  # 52*52*256
    x = resblock_body(x, 512, 8)
    feat2 = x  # 26*26*512
    x = resblock_body(x, 1024, 4)
    feat3 = x  # 13*13*1024

    feat3_last = make_five_layers(feat3, num_filters=512)
    feat3_last_up = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                            UpSampling2D(2))(feat3_last)

    feat2_last = Concatenate()([feat3_last_up, feat2])
    feat2_last = make_five_layers(feat2_last, num_filters=256)
    feat2_last_up = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                            UpSampling2D(2))(feat2_last)

    feat1_last = Concatenate()([feat2_last_up, feat1])
    feat1_last = make_five_layers(feat1_last, 128)

    # apply adaptive spatial feature fusion
    feat3_2 = DarknetConv2D_BN_Leaky(256, (1, 1), name='feat3_2')(feat3_last)
    feat3_2 = layers.UpSampling2D(size=(2, 2), name='feat3_2_upsample')(feat3_2)

    feat3_1 = DarknetConv2D_BN_Leaky(128, (1, 1), name='feat3_1')(feat3_last)
    feat3_1 = layers.UpSampling2D(size=(4, 4), name='feat3_1_upsample')(feat3_1)

    feat2_1 = DarknetConv2D_BN_Leaky(128, (1, 1), name='feat2_1')(feat2_last)
    feat2_1 = layers.UpSampling2D(size=(2, 2), name='feat2_1_upsample')(feat2_1)

    feat2_3 = DarknetConv2D_BN_Leaky(512, (3, 3), strides=2, padding='same', name='feat2_3')(feat2_last)

    feat1_2 = DarknetConv2D_BN_Leaky(256, (3, 3), strides=2, padding='same', name='feat1_2')(feat1_last)

    feat1_3 = DarknetConv2D_BN_Leaky(512, (3, 3), strides=2, padding='same', name='feat1_3')(feat1_last)
    feat1_3 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='feat1_3_pool')(feat1_3)

    # compute importance weights
    feat1_weights = DarknetConv2D_BN_Leaky(16, (1, 1), name='feat1_weights')(feat1)
    feat2_1_weights = DarknetConv2D_BN_Leaky(16, (1, 1), name='feat2_1_weights')(feat2_1)
    feat3_1_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat3_1_weights')(feat3_1)

    feat2_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat2_weights')(feat2)
    feat3_2_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat3_2_weights')(feat3_2)
    feat1_2_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat1_2_weights')(feat1_2)

    feat3_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat3_weights')(feat3)
    feat2_3_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat2_3_weights')(feat2_3)
    feat1_3_weights = DarknetConv2D_BN_Leaky(16, 1, name='feat1_3_weights')(feat1_3)

    feat1_weights, feat2_1_weights, feat3_1_weights = WeightsNormalize()([feat1_weights,
                                                                          feat2_1_weights,
                                                                          feat3_1_weights])
    feat2_weights, feat3_2_weights, feat1_2_weights = WeightsNormalize()([feat2_weights,
                                                                          feat3_2_weights,
                                                                          feat1_2_weights])

    feat3_weights, feat2_3_weights, feat1_3_weights = WeightsNormalize()([feat3_weights,
                                                                          feat2_3_weights,
                                                                          feat1_3_weights])
    feat1_fusion = FeatureFusion()([feat1_last, feat1_weights, feat2_1, feat2_1_weights, feat3_1, feat3_1_weights])
    feat2_fusion = FeatureFusion()([feat2_last, feat2_weights, feat1_2, feat1_2_weights, feat3_2, feat3_2_weights])
    feat3_fusion = FeatureFusion()([feat3_last, feat3_weights, feat1_3, feat1_3_weights, feat2_3, feat2_3_weights])

    return feat1_fusion, feat2_fusion, feat3_fusion


if __name__ == '__main__':
    inputs_ = keras.Input(shape=(416, 416, 3))
    results_ = darknet_asff_body(inputs_)

    y1 = DarknetConv2D_BN_Leaky(128 * 2, 3, padding='same')(results_[0])
    y1 = DarknetConv2D_BN_Leaky(3 * 85, 1)(y1)
    y2 = DarknetConv2D_BN_Leaky(256 * 2, 3, padding='same')(results_[1])
    y2 = DarknetConv2D_BN_Leaky(3 * 85, 1)(y2)
    y3 = DarknetConv2D_BN_Leaky(512 * 2, 3, padding='same')(results_[2])
    y3 = DarknetConv2D_BN_Leaky(3 * 85, 1)(y3)
    model = keras.Model(inputs_, [y1, y2, y3])

    model.summary()
