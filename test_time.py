import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import argparse
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'
thresh = 0.5
# scales = [640, 640]
# scales = [1280, 1280]

count = 2
gpuid = 0

# imgpath='t3.jpg'
# prefix='./model/focal/loss=117.88.ckpt-2'
prefix='./model/new/loss=47.37.ckpt-10'
# prefix = './model/retina11-7-23/loss=13.47.ckpt-13'

def parse_args():
    parser = argparse.ArgumentParser(description='test the speed of RetinaFace')
    parser.add_argument('--network',default=prefix)
    # parser.add_argument('--img',help='test image',default=imgpath)

    args = parser.parse_args()
    return args

def test(path,imgname):
    scales = [640, 640]
    args = parse_args()
    print('args=', args)
    detector = RetinaFace(args.network, gpuid,nms=0.3)

    img = cv2.imread(path+imgname)
    print(img.shape)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False
    all_time = 0
    for c in range(count):
      start = time.time()
      faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
      end = time.time()
      if c!=0:
        all_time+=end-start
      print('count{}, faces.shape={}, landmarks,shape={}'.format(c, faces.shape, landmarks.shape))
    print("average time:{}".format(all_time/(count-1)))
    if faces is not None:
      print('find', faces.shape[0], 'faces')
      font = cv2.FONT_HERSHEY_SIMPLEX
      for i in range(faces.shape[0]):
        #print('score', faces[i][4])
        box = faces[i]
        ibox = box[0:4].copy().astype(np.int)
        cv2.rectangle(img, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (255, 0, 0), 2)
        score = box[4]
        k = "%.3f" % score
        cv2.putText(img, k, (ibox[0] + 2, ibox[1] + 14), font, 0.6, (0, 255, 0), 2)
        if landmarks is not None:
          landmark5 = landmarks[i].astype(np.int)
          #print(landmark.shape)
          for l in range(landmark5.shape[0]):
            color = (0,0,255)
            if l==0 or l==3:
              color = (0,255,0)
            cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

      filename = '640_'+imgname
      print('writing', filename)
      cv2.imwrite(filename, img)

for i in range(1,5):
    path = './zd/'
    img='t%d.jpg'%i
    test(path,img)