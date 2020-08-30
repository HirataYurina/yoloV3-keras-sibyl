# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:2020/4/13 0013 10:35
# filename:yolo1_loss.py
# software: PyCharm

import tensorflow as tf


"""
    the loss function of YOLOv1
    what is YOLOv1 doing to make model be more stable when training?
    Use sqrt to make changes of predict_wh more gentle.
        predict_wh = tf.sqrt(gt_wh)
"""


def loss_layer(self, predicts, labels, scope='loss'):
    """the loss function of YOLOv1

    Args:
        self:
        predicts: raw prediction of YOLOv1
        labels:   gt
        scope:    the name of scope

    """

    with tf.variable_scope(scope):
        # class prob (batch_size, 7, 7, 20)
        # YOLOv1 every grid can only predict one target
        predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                     [self.batch_size, self.cell_size, self.cell_size, self.num_classes])
        # confidence (batch_size, 7, 7, 2)
        # YOLOv1 every grid has only 2 anchor boxes
        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                    [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell])
        # bounding boxes (batch_size, 7, 7, 2, 4)
        predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                   [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        # ground truth
        # the confidence of label
        response = tf.reshape(labels[:, :, :, 0],
                              [self.batch_size, self.cell_size, self.cell_size, 1])
        # the location of label
        boxes = tf.reshape(labels[:, :, :, 1:5],
                           [self.batch_size, self.cell_size, self.cell_size, 1, 4])

        # [batch_szie, cell_size, cell_size, 2, 4]
        # normalize (0, 1)
        boxes = tf.tile(boxes, [1, 1, 1, self.box_per_cell, 1]) / self.image_size
        classes = labels[:, :, :, 5:]

        # the decode of center_xy is same as YOLOv3
        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.box_per_cell])
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])

        # decode raw prediction
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset,
                                                                                    (0, 2, 1, 3))) / self.cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])], axis=-1)
        # iou [batch, 7, 7, 2]
        iou_predict_truth = self.cal_iou(predict_boxes_tran, boxes)
        # best iou of two anchor boxes
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # encode ground truth
        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                               boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])], axis=-1)

        # only positive objects can participate in class loss computing
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                    name='clss_loss') * self.class_scale

        # ----------------------------------------------------------------------------------------------------
        # the confidence of positive objects is not 1 but it is adjusting dynamically by iou_predict_truth
        # TODO: I think it is not sensible because this will
        #       cause the model can not force the confidence of positive grid to be 1.
        # ----------------------------------------------------------------------------------------------------
        object_delta = object_mask * (predict_scales - iou_predict_truth)  # 这是yolo1损失函数一个巧妙的trick
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                     name='object_loss') * self.object_scale
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                       name='noobject_loss') * self.no_object_scale

        # --------------------------------------------------------------------------------------------
        # use squre(sqrt(w) - sqrt(w')) to give small targets more punishment
        # --------------------------------------------------------------------------------------------
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                    name='coord_loss') * self.coord_scale
        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)
