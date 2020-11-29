# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm

import easydict

CONFIG = easydict.EasyDict()

# evaluation
CONFIG.DETECT = easydict.EasyDict()

CONFIG.DETECT.SCORE = 0.3
CONFIG.DETECT.IOU = 0.43
CONFIG.DETECT.RESOLUTION = (416, 416)
CONFIG.DETECT.mAP_THRES = 0.85


# prediction
CONFIG.PREDICT = easydict.EasyDict()
CONFIG.PREDICT.WEIGHTS = '../logs/gaussian_yolo3/ep340-loss-5.042.h5'
CONFIG.PREDICT.ANCHOR_PATH = '../model_data/yolo_anchors.txt'
CONFIG.PREDICT.CLASS_PATH = '../model_data/danger_source_classes.txt'
CONFIG.PREDICT.SCORE = 0.3
CONFIG.PREDICT.IOU = 0.43
CONFIG.PREDICT.RESOLUTION = (416, 416)
CONFIG.PREDICT.MAX_BOXES = 40

# train
CONFIG.TRAIN = easydict.EasyDict()

CONFIG.TRAIN.LR_STAGE = 0.001
CONFIG.TRAIN.BATCH = 4  # it is depending on you GPU memory
CONFIG.TRAIN.EPOCH = 350  # it is enough for transfer training in stage 1
CONFIG.TRAIN.IOU_THRESHOLD = 0.3

CONFIG.TRAIN.COS_INTERVAL = [0.05, 0.15, 0.30, 0.50]  # cosine anneal

CONFIG.TRAIN.ANNO_PATH = '../2088_trainval.txt'
CONFIG.TRAIN.VALID_PATH = '../2088_test.txt'
CONFIG.TRAIN.TEST_PATH = ''
CONFIG.TRAIN.CLASS_PATH = '../model_data/danger_source_classes.txt'
CONFIG.TRAIN.ANCHOR_PATH = '../model_data/yolo_anchors.txt'
CONFIG.TRAIN.PRE_TRAINED_MODEL = '../logs/yolo3_weights.h5'
CONFIG.TRAIN.SAVE_PATH = '../logs/gaussian_yolo3/'
CONFIG.TRAIN.SAVE_PERIOD = 10

CONFIG.TRAIN.RESOLUTION = (416, 416)
CONFIG.TRAIN.IGNORE_THRES = 0.7
CONFIG.TRAIN.CONFIDENCE_FOCAL = False
CONFIG.TRAIN.CLASS_FOCAL = False

# use scale xy to eliminate grid sensitivity
# CONFIG.TRAIN.SCALE_XY = [1.05, 1.1, 1.2]

CONFIG.TRAIN.FREEZE_LAYERS = 249  # freeze 249 layers in YOLOv3

# Augment
CONFIG.AUG = easydict.EasyDict()
CONFIG.AUG.MAX_BOXES = 50

# dataset
CONFIG.DATASET = easydict.EasyDict()
CONFIG.DATASET.MULTIPROCESS = False  # windows can not support multiprocessing in python
CONFIG.DATASET.MOSAIC_AUG = False

CONFIG.DATASET.WORKERS = 1
CONFIG.DATASET.MAX_QUEUE = 128
