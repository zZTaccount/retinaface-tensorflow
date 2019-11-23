#encoding:utf-8
import sys
import numpy as np
import cv2
import random
from rcnn.config import config
from rcnn.io.image import tensor_vstack
from rcnn.io.rpn import get_crop_batch, AA

def get_batch(aa, roidb):
    label_name = config.label_name
    data_name = config.data_name
    # get testing data for multigpu
    data_list = []
    label_list = []
    # TODO：获取单个GPU的rpn_batch，data = {'data': im_array, 'im_info': im_info}，label = {'gt_landmarks'，'gt_boxes'}
    data, label = get_crop_batch(roidb)  # 返回真实label
    # print('core/loader: label.shape = ', np.array(label).shape,'\n')
    data_list += data
    label_list += label  # 每个元素为一张图片
    # print('core/loader: label_list.shape = ',np.array(label_list).shape)
    # print('core/loader: before!! label_list[0] = ',label_list[0])

    select_stride = 0

    for data, label in zip(data_list, label_list):
        # 这里的label是dict
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        im_info = data['im_info']

        gt_boxes = label['gt_boxes']
        gt_label = {'gt_boxes': gt_boxes}
        label_dict = {}
        if config.FACE_LANDMARK:
            gt_landmarks = label['gt_landmarks']
            gt_label['gt_landmarks'] = gt_landmarks
        # TODO 上面把label赋值给gt_label的操作好像是没有意义的，gt_label与label没有区别. but, 后面label好像变了，但是否影响这里呢？

        # ta = datetime.datetime.now()
        # TODO：产生训练label
        face_label_dict = aa.assign_anchor_fpn(gt_label, im_info, config.FACE_LANDMARK, prefix='face',
                                                    select_stride=select_stride)
        # print('face_label_dict.keys = ',face_label_dict.keys())
        # tb = datetime.datetime.now()
        # self._times[0] += (tb-ta).total_seconds()
        label_dict.update(face_label_dict)
        # print('im_info', im_info.shape)
        # print(gt_boxes.shape)
        for k in label_name:
            label[k] = label_dict[k]  # TODO 这里实际上是在更新label_list, 由zip函数返回的label应该是引用关系

    # print('core/loader: after!! label_list[0] = ', label_list[0])

    all_data = dict()
    for key in data_name:
        all_data[key] = tensor_vstack([batch[key] for batch in data_list])

    all_label = dict()
    for key in label_name:
        pad = 0 if key.startswith('bbox_') else -1
        # print('label vstack', key, pad, len(label_list), file=sys.stderr)
        all_label[key] = tensor_vstack([batch[key] for batch in label_list],
                                       pad=pad)  # 这里的batch其实就是dict，这个函数的作用是将所有的图片的相同key的list叠起来

    # print('batch_key_list len = ',len([batch['gt_boxes'] for batch in label_list]))

    labels = {}
    data = [np.array(all_data[key]) for key in data_name]

    # print('label_list len = ', len([np.array(all_label[key])for key in label_name]))
    label_d = {}
    for key in label_name:
        label_d[key] = all_label[key]
        # label = [np.array()for key in label_name] #该list是按照顺序存的，要记住label和本list中的array的对应顺序
    return data, label_d

def label_concat(labels):
    ret_group={}
    ret_group['face_label'] = np.concatenate(
        [labels['face_label_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=1)
    ret_group['face_bbox_target'] = np.concatenate(
        [labels['face_bbox_target_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=1)
    ret_group['face_bbox_weight'] = np.concatenate(
        [labels['face_bbox_weight_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=1)
    ret_group['face_landmark_target'] = np.concatenate(
        [labels['face_landmark_target_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=2)
    ret_group['face_landmark_weight'] = np.concatenate(
        [labels['face_landmark_weight_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=2)
    ind = np.where(ret_group['face_label']==1) #两维，就返回两个array
    # print('core/loader: ind=',ind)
    label_weight = np.zeros_like(ret_group['face_label'])
    label_weight[ind]=1
    # print('label_weight= ',np.where(label_weight<0))
    ret_group['face_label_weight'] = label_weight
    return ret_group

class DataLoader():
    def __init__(self, roidb, batch_size=1, shuffle=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :return: AnchorLoader
        """
        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle

        # infer properties from roidb
        self.size = len(roidb)
        # print('dataloader self.size = ',self.size)
        self.index = np.arange(self.size)

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # infer shape
        feat_shape_list = [[batch_size, 4, 20, 20], [batch_size, 4, 40, 40], [batch_size, 4, 80, 80]] # 这是三个stride的feature map大小
        self.aa = AA(feat_shape_list)

        self._debug = False
        self._debug_id = 0
        self._times = [0.0, 0.0, 0.0, 0.0]

        # get first batch to fill in provide_data and provide_label
        self.reset()


    def __iter__(self):
        return self

    def reset(self):
        self.cur = 0
        if self.shuffle:
          print('core/loader: shuffle roidb!\n')
          random.shuffle(self.roidb)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def get_next(self):
        if self.iter_next():
            batch_roidb = [self.roidb[i] for i in range(self.cur, self.cur+self.batch_size)]
            # print('batch_roidb len=', len(batch_roidb))
            data, label = self.get_batch(batch_roidb)
            label = label_concat(label)
            self.cur += self.batch_size
            return data[0], label #TODO :data的第一维度是gpu数目，目前是单GPU
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def get_batch(self, roidb):
        label_name = config.label_name
        data_name = config.data_name
        # get testing data for multigpu
        data_list = []
        label_list = []
        # TODO：获取单个GPU的rpn_batch，data = {'data': im_array, 'im_info': im_info}，label = {'gt_landmarks'，'gt_boxes'}
        data, label = get_crop_batch(roidb)  # 返回真实label
        # print('core/loader: label.shape = ', np.array(label).shape,'\n')
        data_list += data
        label_list += label  # 每个元素为一张图片
        # print('core/loader: label_list.shape = ',np.array(label_list).shape)
        # print('core/loader: before!! label_list[0] = ',label_list[0])

        select_stride = 0

        for data, label in zip(data_list, label_list):
            # 这里的label是dict
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            im_info = data['im_info']

            gt_boxes = label['gt_boxes']
            # print('loader/get_batch: in the gt_label! gt_bboxes.shape=',gt_boxes.shape)
            gt_label = {'gt_boxes': gt_boxes}
            label_dict = {}
            if config.FACE_LANDMARK:
                gt_landmarks = label['gt_landmarks']
                # print('loader/get_batch: in the gt_label! gt_landmarks.shape=', gt_landmarks.shape)
                gt_label['gt_landmarks'] = gt_landmarks
            # TODO 上面把label赋值给gt_label的操作好像是没有意义的，gt_label与label没有区别. but, 后面label好像变了，但是否影响这里呢？

            # ta = datetime.datetime.now()
            # TODO：产生训练label
            face_label_dict = self.aa.assign_anchor_fpn(gt_label, im_info, config.FACE_LANDMARK, prefix='face',
                                                   select_stride=select_stride)
            # print('face_label_dict.keys = ',face_label_dict.keys())
            # tb = datetime.datetime.now()
            # self._times[0] += (tb-ta).total_seconds()
            label_dict.update(face_label_dict)
            # print('im_info', im_info.shape)
            # print(gt_boxes.shape)
            for k in label_name:
                label[k] = label_dict[k]  # TODO 这里实际上是在更新label_list, 由zip函数返回的label应该是引用关系

        # print('core/loader: after!! label_list[0] = ', label_list[0])

        all_data = dict()
        for key in data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in label_name:
            pad = 0 if key.startswith('bbox_') else -1
            # print('label vstack', key, pad, len(label_list), file=sys.stderr)
            all_label[key] = tensor_vstack([batch[key] for batch in label_list],
                                           pad=pad)  # 这里的batch其实就是dict，这个函数的作用是将所有的图片的相同key的list叠起来

        # print('batch_key_list len = ',len([batch['gt_boxes'] for batch in label_list]))

        labels = {}
        data = [np.array(all_data[key]) for key in data_name]

        # print('label_list len = ', len([np.array(all_label[key])for key in label_name]))
        label_d = {}
        for key in label_name:
            label_d[key] = all_label[key]
            # label = [np.array()for key in label_name] #该list是按照顺序存的，要记住label和本list中的array的对应顺序
        # for key in label_d.keys():
        #     print('{}: {}'.format(key,label_d[key].shape))
        return data, label_d