#encoding:utf-8
import numpy as np
from easydict import EasyDict as edict
import datetime
#自定义
config = edict()
config.epoch = 50
config.bn_decay=0.999
config.weight_decay = 0.998
config.is_training=True
config.lr_step = [3,10]
config.data_format='NHWC'
config.HEAD_FILTER_NUM = 256
config.USE_CROP=True
config.FACE_LANDMARK = True
config.retrain = False
config.label_name=  ['face_label_stride32', 'face_bbox_target_stride32', 'face_bbox_weight_stride32', 'face_landmark_target_stride32', 'face_landmark_weight_stride32',
            'face_label_stride16', 'face_bbox_target_stride16', 'face_bbox_weight_stride16', 'face_landmark_target_stride16', 'face_landmark_weight_stride16',
            'face_label_stride8', 'face_bbox_target_stride8', 'face_bbox_weight_stride8', 'face_landmark_target_stride8', 'face_landmark_weight_stride8']

config.data_name=['data']

#没用的配置
config.USE_BLUR = False
config.HEAD_BOX = False
config.DENSE_ANCHOR = False
config.USE_3D = False

#图像处理
config.PIXEL_MEANS = np.array([103.939, 116.779, 123.68])
config.PIXEL_STDS = np.array([1.0, 1.0, 1.0])
config.PIXEL_SCALE = 1.0
config.IMAGE_STRIDE = 0
config.COLOR_MODE = 2
config.COLOR_JITTERING = 0.125
config.MIXUP = 0.0

#计算anchor所需
config.SCALES = [(640, 640)]
_ratio = (1.,)
config.RPN_ANCHOR_CFG={
    '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999,'NUM_ANCHORS': 2},
    '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999,'NUM_ANCHORS': 2},
    '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999,'NUM_ANCHORS': 2},
}
config.RPN_FEAT_STRIDE=[32,16,8]

#训练
config.TRAIN = edict()
config.TRAIN.BATCH_IMAGES = 10
config.TRAIN.MIN_BOX_SIZE=0
# rpn anchors batch size
config.TRAIN.RPN_ENABLE_OHEM = 2
config.TRAIN.RPN_BATCH_SIZE = 1024 #TODO 256还是128
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.0625 #正样本64
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.6
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
config.TRAIN.RPN_FORCE_POSITIVE = False
config.TRAIN.FOCALLOSS_alpha=0.5
config.TRAIN.FOCALLOSS_gama=2
config.TRAIN.SMOOTHL1_sigma=3


# default settings
default = edict()
default.flip = False
default.shuffle = True
# default network
default.network = 'resnet'
# default.pretrained = './model/retina11-7-23/loss=13.47.ckpt-13'
# default.pretrained = './model/retina11-8-15/loss=7.76.ckpt-0'
default.pretrained = './model/new/loss=19.39.ckpt-46'
default.pretrained_epoch = 0
# default dataset
default.dataset = 'retinaface'
default.image_set = 'train'
default.test_image_set = 'val'
default.root_path = 'data'
default.dataset_path = 'data/retinaface'
# default.dataset_path = 'data/debug'

# default training
default.frequent = 20
# default.use_ohem = True
default.use_focalLoss = False #当使用focalloss的时候就不使用ohem了
default.use_ohem = False
# default e2e
# time = str(datetime.datetime.now().month)+'-'+str(datetime.datetime.now().day)+'-'+str(datetime.datetime.now().hour)
default.prefix = 'model/new-NOohem'
default.end_epoch = config.epoch
default.lr_step = [3,9,15,30]
default.lr = 1e-3
