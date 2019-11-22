#encoding:utf-8
'''
retinaface.py
用于定义一个retinaface类，提供检测接口
'''
from __future__ import print_function
import sys
import os
import datetime
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from rcnn.config import config
import cv2
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from rcnn.model.common import get_pred

class RetinaFace:
  def __init__(self, network, gpu_id=0, nms=0.3, nocrop=False, decay4 = 0.5, vote=False):
    self.network = network
    self.gpu_id = gpu_id
    self.decay4 = decay4
    self.nms_threshold = nms
    self.vote = vote
    self.nocrop = nocrop
    self.debug = False
    self.fpn_keys = []
    self.anchor_cfg = None
    self.use_landmarks = True
    self.sess=None
    self.net=None
    pixel_means=[0.0, 0.0, 0.0]
    pixel_stds=[1.0, 1.0, 1.0]
    pixel_scale = 1.0
    self.preprocess = False
    dense_anchor = False
    _ratio = (1.,)
    image_size = (640, 640)

    self._feat_stride_fpn = config.RPN_FEAT_STRIDE
    self.anchor_cfg = config.RPN_ANCHOR_CFG

    for s in self._feat_stride_fpn:
        self.fpn_keys.append('stride%s'%s)
    self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
    for k in self._anchors_fpn:
      v = self._anchors_fpn[k].astype(np.float32)
      self._anchors_fpn[k] = v
    '''
    {'stride32': array([[-248., -248.,  263.,  263.],[-120., -120.,  135.,  135.]], dtype=float32), 
     'stride16': array([[-56., -56.,  71.,  71.],[-24., -24.,  39.,  39.]], dtype=float32), 
     'stride8': array([[-8., -8., 23., 23.],[ 0.,  0., 15., 15.]], dtype=float32)}
    '''
    # print('Retinaface: self.anchor_fpn=', self._anchors_fpn)

    self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
    #{'stride32': 2, 'stride16': 2, 'stride8': 2}
    # print('Retinaface: self._num_anchors=', self._num_anchors)

    if self.gpu_id>=0:
      self.nms = gpu_nms_wrapper(self.nms_threshold, self.gpu_id)
    else:
      self.nms = cpu_nms_wrapper(self.nms_threshold)

    self.pixel_means = np.array(pixel_means, dtype=np.float32)
    self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
    self.pixel_scale = float(pixel_scale)
    self.build_net()

  def build_net(self):
      tf.reset_default_graph()
      self.data = tf.placeholder(tf.float32, shape=(None,None,None,3), name='data')
      ret = get_pred(self.data)
      saver = tf.train.Saver()  # 创建saver时会去查看现有的图
      for s in config.RPN_FEAT_STRIDE:
          logits = ret['rpn_cls_score_stride%s' % s]
          logits = tf.reshape(logits, shape=(-1, 2))
          logits = tf.Print(logits, [tf.shape(logits)], 'Debug message: logits=', first_n=10,summarize=200)
          prob = tf.nn.softmax(logits, axis=1)
          prob = tf.Print(prob, [tf.shape(prob), prob], 'Debug message: prob',first_n=10, summarize=200)
          ret['rpn_cls_prob_stride%s'%s] = prob
      sess = tf.Session()
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
      sess.run(init)
      saver.restore(sess, self.network)
      self.sess = sess
      self.net = ret

  def get_pred(self, input):
      print('get_pred: after resize: img.shape=',input.shape)
      input = input.transpose(0, 2, 3, 1)  # TODO : very important,change the format to NHWC
      pred = self.sess.run(self.net, feed_dict={self.data: input})
      return pred

  def get_input(self, img):
    im = img.astype(np.float32)
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = (im[:, :, 2 - i]/self.pixel_scale - self.pixel_means[2 - i])/self.pixel_stds[2-i]
    data = np.array(im_tensor)
    return data

  def detect(self, img, threshold=0.5, scales=[1.0], do_flip=False):
    print('get into detect, confi thresold={}, scales={}, do_flip={}'.format(threshold, scales, do_flip))
    proposals_list = []
    scores_list = []
    landmarks_list = []
    timea = datetime.datetime.now()
    flips = [0]
    if do_flip:
      flips = [0, 1]

    #TODO 根据scale给输入的图片做resize
    for im_scale in scales:
      for flip in flips:
        if im_scale!=1.0:
          im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        else:
          im = img.copy()
        # 对图像做翻转
        if flip:
          im = im[:,::-1,:]
        # 对图像做裁剪
        if self.nocrop:
          if im.shape[0]%32==0:
            h = im.shape[0]
          else:
            h = (im.shape[0]//32+1)*32
          if im.shape[1]%32==0:
            w = im.shape[1]
          else:
            w = (im.shape[1]//32+1)*32
          _im = np.zeros( (h, w, 3), dtype=np.float32 )
          _im[0:im.shape[0], 0:im.shape[1], :] = im
          im = _im
        else:
          im = im.astype(np.float32)
        # im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR) #TODO 记得删掉
        im_info = [im.shape[0], im.shape[1]] #h,w
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i]/self.pixel_scale - self.pixel_means[2 - i])/self.pixel_stds[2-i] #TODO 这里好像将Channel顺序倒过来了
        data = np.array(im_tensor)

        # 读入模型进行推理，得到预测值
        net_out = self.get_pred(data)
        if self.debug:
          for key in net_out.keys():
              print('{} = {}\n'.format(key, net_out[key].shape))

        for _idx,s in enumerate(self._feat_stride_fpn):
            # print('begin stride{}-------------------------------------------------\n'.format(s))
            _key = 'stride%s'%s
            stride = int(s)
            # print('getting im_scale={}, stride={}, len(net_out)={}, data.shape={}'.format(im_scale, stride, len(net_out), data.shape))
            scores = net_out['rpn_cls_prob_stride%s'%s] #TODO 要注意这里是nhwc不是nchw
            if self.debug:
              print('get score:',scores.shape)

            # print('stride{}: scores before shape={}, idx={}'.format(stride, scores.shape, self._num_anchors['stride%s' % s]))
            scores = scores[:, 1].reshape((-1, 1))  #TODO: (H*W*A, 1) #这里的1表示正类的概率

            if self.debug:
              print('AAAAstride{}: scores after shape={}'.format(stride, scores.shape))

            bbox_deltas = net_out['rpn_bbox_pred_stride%s'%s] #TODO NHW8

            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
            A = self._num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, stride, anchors_fpn) #获取该特征图上的所有anchor
            #print((height, width), (_height, _width), anchors.shape, bbox_deltas.shape, scores.shape, file=sys.stderr)
            anchors = anchors.reshape((K * A, 4))
            if self.debug:
              print('HW', (height, width))
              print('anchors_fpn', anchors_fpn)
              print('anchors', anchors.shape,'\n')

            # scores = self._clip_pad_NCHW(scores, (height, width))  #(1, 4, H, W)
            # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1)) #(1, H, W, 4)
            # print('scores reshape', scores.shape)
            if self.debug:
              print('before bbox_deltas', bbox_deltas.shape)
            bbox_deltas = self._clip_pad_NHWC(bbox_deltas, (height, width)) #(1, H, W, 8)
            # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))#(1, H, W, 8)
            bbox_pred_len = bbox_deltas.shape[3]//A #4
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len)) #(H*W*2, 4)
            if self.debug:
              print('after bbox_deltas', bbox_deltas.shape, height, width,'\n')

            #print(anchors.shape, bbox_deltas.shape, A, K, file=sys.stderr)
            proposals = self.bbox_pred(anchors, bbox_deltas) #TODO important! 将anchor加上delta进行处理
            proposals = clip_boxes(proposals, im_info[:2])

            scores_ravel = scores.ravel()
            max_score = np.max(scores_ravel)
            print('proposals.shape={}, score_ravel.shape={}'.format(proposals.shape, scores_ravel.shape))
            print('max score',max_score)
            order = np.where(scores_ravel>=threshold)[0]
              #_scores = scores_ravel[order]
              #_order = _scores.argsort()[::-1]
              #order = order[_order]
            proposals = proposals[order, :]
            scores = scores[order]
            if flip:
              oldx1 = proposals[:, 0].copy()
              oldx2 = proposals[:, 2].copy()
              proposals[:, 0] = im.shape[1] - oldx2 - 1
              proposals[:, 2] = im.shape[1] - oldx1 - 1

            proposals[:,0:4] /= im_scale #TODO important 在这里将找到的proposal给映射回原来图像的位置

            proposals_list.append(proposals)
            scores_list.append(scores)

            if not self.vote and self.use_landmarks:
              landmark_deltas = net_out['rpn_landmark_pred_stride%s'%s] #(1,20,H,W)
              if self.debug:
                print('before landmark_deltas', landmark_deltas.shape)
              landmark_deltas = self._clip_pad_NCHW(landmark_deltas, (height, width))
              landmark_pred_len = landmark_deltas.shape[1]//A
              landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
              if self.debug:
                print('after landmark_deltas',landmark_deltas.shape, landmark_deltas)
              landmarks = self.landmark_pred(anchors, landmark_deltas)
              landmarks = landmarks[order, :]

              if flip:
                landmarks[:,:,0] = im.shape[1] - landmarks[:,:,0] - 1
                #for a in range(5):
                #  oldx1 = landmarks[:, a].copy()
                #  landmarks[:,a] = im.shape[1] - oldx1 - 1
                order = [1,0,2,4,3]
                flandmarks = landmarks.copy()
                for idx, a in enumerate(order):
                  flandmarks[:,idx,:] = landmarks[:,a,:]
                  #flandmarks[:, idx*2] = landmarks[:,a*2]
                  #flandmarks[:, idx*2+1] = landmarks[:,a*2+1]
                landmarks = flandmarks
              landmarks[:,:,0:2] /= im_scale
              landmarks_list.append(landmarks)
              if self.debug:
                print('end stride{}-------------------------------------------------\n'.format(s))

    proposals = np.vstack(proposals_list)
    landmarks = None
    if proposals.shape[0]==0:
      if self.use_landmarks:
        landmarks = np.zeros( (0,5,2) )
      return np.zeros( (0,5) ), landmarks
    # for i in range(len(scores_list)):
    #     print('hhhhh score,shape=',scores_list[i].shape)
    scores = np.vstack(scores_list)
    print('finally!!! proposals.shape={}, score.shape={}'.format(proposals.shape, scores.shape))
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1] # 按照score从大到小排序
    proposals = proposals[order, :]
    scores = scores[order]
    if self.debug:
        print('sort score=', scores)
    if not self.vote and self.use_landmarks:
      landmarks = np.vstack(landmarks_list)
      landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
    if not self.vote:
      print('begin to NMS!!\n')
      keep = self.nms(pre_det)
      # print('before hstack: pre_det={}, proposals.shape={}, proposals[:,4:]={}'.format(pre_det.shape, proposals.shape, proposals[:,4:]))
      det = np.hstack( (pre_det, proposals[:,4:]) )
      # print('after hstack: pre_det={}, proposals.shape={}'.format(pre_det.shape, proposals.shape))
      det = det[keep, :]
      if self.use_landmarks:
        landmarks = landmarks[keep]
    else:
      det = np.hstack( (pre_det, proposals[:,4:]) )
      det = self.bbox_vote(det)
    return det, landmarks

  @staticmethod
  def _clip_pad_NHWC(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, H, W, c]
      :param pad_shape: [h, w]
      :return: [n, h, w, c]
      """
      H, W = tensor.shape[1:3]
      h, w = pad_shape

      if h < H or w < W:
          tensor = tensor[:, :h, :w, :].copy()

      return tensor

  @staticmethod
  def _clip_pad_NCHW(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
          print('clip_pad')
          tensor = tensor[:, :, :h, :w].copy()

      return tensor

  @staticmethod
  def bbox_pred(boxes, box_deltas):
      """
      Transform the set of class-agnostic boxes into class-specific boxes
      by applying the predicted offsets (box_deltas)
      :param boxes: !important [N 4]
      :param box_deltas: [N, 4 * num_classes]
      :return: [N 4 * num_classes]
      """
      if boxes.shape[0] == 0:
          return np.zeros((0, box_deltas.shape[1]))

      boxes = boxes.astype(np.float, copy=False)
      widths = boxes[:, 2] - boxes[:, 0] + 1.0
      heights = boxes[:, 3] - boxes[:, 1] + 1.0
      ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
      ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

      dx = box_deltas[:, 0:1]
      dy = box_deltas[:, 1:2]
      dw = box_deltas[:, 2:3]
      dh = box_deltas[:, 3:4]

      pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
      pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
      pred_w = np.exp(dw) * widths[:, np.newaxis]
      pred_h = np.exp(dh) * heights[:, np.newaxis]

      pred_boxes = np.zeros(box_deltas.shape)
      # x1
      pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
      # y1
      pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
      # x2
      pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
      # y2
      pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

      if box_deltas.shape[1] > 4:
          pred_boxes[:, 4:] = box_deltas[:, 4:]

      return pred_boxes

  @staticmethod
  def landmark_pred(boxes, landmark_deltas):
      if boxes.shape[0] == 0:
          return np.zeros((0, landmark_deltas.shape[1]))
      boxes = boxes.astype(np.float, copy=False)
      widths = boxes[:, 2] - boxes[:, 0] + 1.0
      heights = boxes[:, 3] - boxes[:, 1] + 1.0
      ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
      ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
      pred = landmark_deltas.copy()
      for i in range(5):
          pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
          pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
      return pred

  def bbox_vote(self, det):
      if det.shape[0] == 0:
          dets = np.array([[10, 10, 20, 20, 0.002]])
          det = np.empty(shape=[0, 5])
      while det.shape[0] > 0:
          # IOU
          area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
          xx1 = np.maximum(det[0, 0], det[:, 0])
          yy1 = np.maximum(det[0, 1], det[:, 1])
          xx2 = np.minimum(det[0, 2], det[:, 2])
          yy2 = np.minimum(det[0, 3], det[:, 3])
          w = np.maximum(0.0, xx2 - xx1 + 1)
          h = np.maximum(0.0, yy2 - yy1 + 1)
          inter = w * h
          o = inter / (area[0] + area[:] - inter)
          # print('retinaface: bbox_vote o=',o)

          # nms
          merge_index = np.where(o >= self.nms_threshold)[0]
          det_accu = det[merge_index, :]
          det = np.delete(det, merge_index, 0)
          if merge_index.shape[0] <= 1:
              if det.shape[0] == 0:
                  try:
                      dets = np.row_stack((dets, det_accu))
                  except:
                      dets = det_accu
              continue
          det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
          max_score = np.max(det_accu[:, 4])
          det_accu_sum = np.zeros((1, 5))
          det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                        axis=0) / np.sum(det_accu[:, -1:])
          det_accu_sum[:, 4] = max_score
          try:
              dets = np.row_stack((dets, det_accu_sum))
          except:
              dets = det_accu_sum
      dets = dets[0:750, :]
      return dets