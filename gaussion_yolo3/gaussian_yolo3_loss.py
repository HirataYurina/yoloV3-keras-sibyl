# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:gaussian_yolo3_loss.py
# software: PyCharm

# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:mish.py
# software: PyCharm

import tensorflow as tf
from keras import backend as K
import math
from gaussion_yolo3.gaussian_yolo3 import yolo_head


# ---------------------------------------------------
# get iou
# ---------------------------------------------------
def box_iou(b1, b2):

    # [13, 13, 3, 1, 4]
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # [1, n, 4]
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # calculate iou
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    # TODO:avoid dividing by zero
    iou = intersect_area / (b1_area + b2_area - intersect_area + K.epsilon())

    return iou


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    Args:
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    Returns:
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    p_t = ((y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob)))
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
    # sigmoid_focal_loss = tf.reduce_sum(sigmoid_focal_loss, axis=-1)

    return sigmoid_focal_loss


def gaussian_distribution(mu, sigma, x):
    prob = (1 / (K.sqrt(K.constant(2.0 * math.pi, dtype='float32')) * sigma + K.epsilon())) * \
           K.exp(-1 * K.square(x - mu) / (2 * K.square(sigma) + K.epsilon()))

    return prob


# ---------------------------------------------------------------------------------------
# loss compute
# support:
#   focal confidence loss
#   focal class loss
#   gaussian loss
# ---------------------------------------------------------------------------------------
def gaussian_yolo_loss(args,
                       anchors,
                       num_classes,
                       ignore_thresh=.5,
                       print_loss=False,
                       use_focal_confidence_loss=False,
                       use_focal_class_loss=False):

    # 3 layers
    num_layers = len(anchors)//3

    # args = [*model_body.output, *y_true]
    # y_true: [(m, 13, 13, 3, 85), (m, 26, 26, 3, 85), (m, 52, 52, 3, 85)]
    # yolo_outputs: [(m, 13, 13, 3, 89), (m, 26, 26, 3, 89), (m, 52, 52, 3, 89)]
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # [6, 7, 8]: [(116, 90),  (156, 198),  (373, 326)]
    # [3, 4, 5]: [(30 , 61),  (62,   45),  (59,  119)]
    # [0, 1, 2]: [(10,  13),  (16,   30),  (33,   23)]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # [416, 416]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    # [[13, 13], [26, 26], [52, 52]]
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0

    # cast m to float
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        # confidence
        object_mask = y_true[l][..., 4:5]
        # class probability
        true_class_probs = y_true[l][..., 5:]

        # pred_xy and pred_wh are normalized
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]],
                                                     num_classes,
                                                     input_shape,
                                                     calc_loss=True)

        # (m, 13, 13, 3, 4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # make a dynamic tensor array
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            # (n, 4)
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])

            # calculate iou
            # (13, 13, 3, n)
            iou = box_iou(pred_box[b], true_box)

            # (13, 13, 3, 1)
            best_iou = K.max(iou, axis=-1)

            # if iou < ignore threshold: negative.
            # if iou > ignore threshold and it not positive, it's ignore anchor.
            # And these anchors are closed to positive anchor.
            # yoloV3 uses this trick to maintain number of negative anchors.
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        # repeat loop_body function while condition is true
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        ignore_mask = ignore_mask.stack()
        # (m, 13, 13, 3, 1, 1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # encode the gt bounding boxes
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][:] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])

        # #######################################
        # use switch to exchange -inf to 0
        # 0 * inf = NAN
        # #######################################
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        # TODO: yolo3 uses this scale to penalize errors in small gt bounding boxes.
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        x_loss = (-1) * object_mask * box_loss_scale * \
            K.log(gaussian_distribution(mu=K.sigmoid(raw_pred[..., 0:1]),
                                        sigma=K.sigmoid(raw_pred[..., 4:5]),
                                        x=raw_true_xy[..., 0:1]) + K.epsilon())
        y_loss = (-1) * object_mask * box_loss_scale * \
            K.log(gaussian_distribution(mu=K.sigmoid(raw_pred[..., 1:2]),
                                        sigma=K.sigmoid(raw_pred[..., 5:6]),
                                        x=raw_true_xy[..., 1:2]) + K.epsilon())
        w_loss = (-1) * object_mask * box_loss_scale * \
            K.log(gaussian_distribution(mu=raw_pred[..., 2:3],
                                        sigma=K.sigmoid(raw_pred[..., 6:7]),
                                        x=raw_true_wh[..., 0:1]) + K.epsilon())
        h_loss = (-1) * object_mask * box_loss_scale * \
            K.log(gaussian_distribution(mu=raw_pred[..., 3:4],
                                        sigma=K.sigmoid(raw_pred[..., 7:8]),
                                        x=raw_true_wh[..., 1:2]) + K.epsilon())

        # use focal confidence loss
        if use_focal_confidence_loss:
            confidence_loss = sigmoid_focal_loss(object_mask, raw_pred[..., 8:9])
        else:
            confidence_loss = object_mask * K.binary_crossentropy(object_mask,
                                                                  raw_pred[..., 8:9], from_logits=True) + \
                (1-object_mask) * K.binary_crossentropy(object_mask,
                                                        raw_pred[..., 8:9], from_logits=True) * ignore_mask
        # use focal class loss
        if use_focal_class_loss:
            class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[..., 9:])
        else:
            class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 9:], from_logits=True)

        x_loss = K.sum(x_loss) / mf
        y_loss = K.sum(y_loss) / mf
        w_loss = K.sum(w_loss) / mf
        h_loss = K.sum(h_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += x_loss + y_loss + w_loss + h_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, x_loss, y_loss, w_loss, h_loss,
                                   confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
