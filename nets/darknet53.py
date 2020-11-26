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
        self.add = layers.Add()

    def call(self, inputs, **kwargs):
        input1 = K.exp(inputs[0])
        input2 = K.exp(inputs[1])
        input3 = K.exp(inputs[2])

        inputs_sum = self.add([input1, input2, input3])

        input1 = input1 / inputs_sum
        input2 = input2 / inputs_sum
        input3 = input3 / inputs_sum

        return [input1, input2, input3]


class FeatureFusion(layers.Layer):

    def __init__(self, **kwargs):
        super(FeatureFusion, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        results = tf.multiply(inputs[0], inputs[1]) + \
                  tf.multiply(inputs[2], inputs[3]) + \
                  tf.multiply(inputs[4], inputs[5])

        return results


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

    # apply adaptive spatial feature fusion
    feat3_2 = Conv2D(512, 1, padding='same', name='feat3_2')(feat3)
    feat3_2 = layers.UpSampling2D(size=(2, 2), name='feat3_2_upsample')(feat3_2)

    feat3_1 = Conv2D(256, 1, padding='same', name='feat3_1')(feat3)
    feat3_1 = layers.UpSampling2D(size=(4, 4), name='feat3_1_upsample')(feat3_1)

    feat2_1 = Conv2D(256, 1, padding='same', name='feat2_1')(feat2)
    feat2_1 = layers.UpSampling2D(size=(2, 2), name='feat2_1_upsample')(feat2_1)

    feat2_3 = Conv2D(1024, 3, strides=2, padding='same', name='feat2_3')(feat2)

    feat1_2 = Conv2D(512, 3, strides=2, padding='same', name='feat1_2')(feat1)

    feat1_3 = Conv2D(1024, 3, strides=2, padding='same', name='feat1_3')(feat1)
    feat1_3 = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='feat1_3_pool')(feat1_3)

    # compute importance weights
    feat1_weights = Conv2D(256, 1, name='feat1_weights')(feat1)
    feat2_1_weights = Conv2D(256, 1, name='feat2_1_weights')(feat2_1)
    feat3_1_weights = Conv2D(256, 1, name='feat3_1_weights')(feat3_1)

    feat2_weights = Conv2D(512, 1, name='feat2_weights')(feat2)
    feat3_2_weights = Conv2D(512, 1, name='feat3_2_weights')(feat3_2)
    feat1_2_weights = Conv2D(512, 1, name='feat1_2_weights')(feat1_2)

    feat3_weights = Conv2D(1024, 1, name='feat3_weights')(feat3)
    feat2_3_weights = Conv2D(1024, 1, name='feat2_3_weights')(feat2_3)
    feat1_3_weights = Conv2D(1024, 1, name='feat1_3_weights')(feat1_3)

    feat1_weights, feat2_1_weights, feat3_1_weights = WeightsNormalize()([feat1_weights,
                                                                          feat2_1_weights,
                                                                          feat3_1_weights])
    feat2_weights, feat3_2_weights, feat1_2_weights = WeightsNormalize()([feat2_weights,
                                                                          feat3_2_weights,
                                                                          feat1_2_weights])

    feat3_weights, feat2_3_weights, feat1_3_weights = WeightsNormalize()([feat3_weights,
                                                                          feat2_3_weights,
                                                                          feat1_3_weights])
    feat1_fusion = FeatureFusion()([feat1, feat1_weights, feat2_1, feat2_1_weights, feat3_1, feat3_1_weights])
    feat2_fusion = FeatureFusion()([feat2, feat2_weights, feat1_2, feat1_2_weights, feat3_2, feat3_2_weights])
    feat3_fusion = FeatureFusion()([feat3, feat3_weights, feat1_3, feat1_3_weights, feat2_3, feat2_3_weights])

    return feat1_fusion, feat2_fusion, feat3_fusion


if __name__ == '__main__':
    import keras

    inputs_ = keras.Input(shape=(416, 416, 3))
    feature1_, feature2_, feature3_ = darknet_asff_body(inputs_)  # Trainable params: 53,897,696
    # feature1_, feature2_, feature3_ = csp_darknet_body(inputs_)  # Trainable params: 26,617,184
    # feature1_, feature2_, feature3_ = darknet_body(inputs_)  # Trainable params: 40,584,928

    # backbone
    backbone = keras.Model(inputs_, [feature1_, feature2_, feature3_])

    # darknet53 trainable params: 40,584,928
    # cspdarknet 53 trainable params: 26,617,184
    # so, the params if cspdarkbet53 is much smaller than draknet53.
    backbone.summary()
