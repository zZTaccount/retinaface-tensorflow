#encoding:utf-8
'''
train.py
训练网络（读入数据-构建网络-计算loss-不断收敛）
'''
import sys
import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
from rcnn.model import resnet_v2
from rcnn.config import config, default
from rcnn.core.loader import label_concat,DataLoader
from rcnn.utils.load_data import load_gt_roidb
from rcnn.io.rpn import AA
from rcnn.model.common import *

os.environ['CUDA_VISIBLE_DEVICES']='1'

def train_net(args, ctx):
    #一些局部变量
    lowest_loss=10000
    tf.reset_default_graph()
    print('len(ctx) = ',len(ctx))
    a = str(datetime.datetime.now().month) + '-' + str(datetime.datetime.now().day) + '-' + str(
        datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)
    if not os.path.exists(args.save_prefix):
        os.mkdir(args.save_prefix)
    log_file = args.save_prefix + '/log.txt'
    f = open(log_file, 'a')

    # 开始构建网络
    print('begin to build net')
    data = tf.placeholder(tf.float32, (None, 640, 640, 3))
    label = tf.placeholder(tf.int32, (None, 16800))
    label_weight = tf.placeholder(tf.int32, (None, 16800))
    bbox_target = tf.placeholder(tf.float32, (None, 8400, 8))
    bbox_weight = tf.placeholder(tf.float32, (None, 8400, 8))
    landmark_target = tf.placeholder(tf.float32, (None, 8400, 20))
    landmark_weight = tf.placeholder(tf.float32, (None, 8400, 20))
    lr = tf.placeholder(tf.float32, name='l_rate')

    #定义resnet
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(data, num_classes=0, is_training=config.is_training, global_pool=False)
    resnet_saver = tf.train.Saver()  # 创建saver时会去查看现有的图

    conv = get_sym_conv(net)
    ret = get_out(conv, 'face', config.FACE_LANDMARK)
    concat = stride_concat(ret)

    with tf.name_scope('losses'):
        # cls_loss,bbox_loss,landmark_loss = get_loss(concat, label, label_weight, bbox_target, bbox_weight, landmark_target, landmark_weight, 'face')
        # pos_loss, neg_loss = get_loss(concat, label, label_weight, bbox_target, bbox_weight, landmark_target, landmark_weight, 'face')
        cls_loss, bbox_loss, landmark_loss = get_loss(concat, label, label_weight, bbox_target, bbox_weight, landmark_target, landmark_weight, 'face')
        # pos_loss = tf.Print(pos_loss, [pos_loss, neg_loss],
        #                         'Debug message: ohem return Loss= ', first_n=50,
        #                                                 summarize=100,
        #                                                 name='PRI_return')
        slim.losses.add_loss(cls_loss)
        reg_loss = tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops), tf.name_scope('optimizer'):
        # myloss = ohem_loss+bbox_loss+landmark_loss
        cls_loss = 2*cls_loss
        myloss = cls_loss + bbox_loss + 0.5*landmark_loss
        total_loss = myloss+0.5*reg_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train = optimizer.minimize(total_loss)
        D = tf.Print(lr,[lr],message='Debug message: ')


    retinaface_saver = tf.train.Saver(max_to_keep=50)  # 创建saver时会去查看现有的图

    with tf.Session() as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        if args.retrain:
            retinaface_saver.restore(sess, args.pretrained)
            print('restore model! ', args.pretrained)
        else:
            resnet_saver.restore(sess, "./model/resnet_v2_50.ckpt")
            print('restore resnet successfully\n')

        # 读取数据
        with tf.device('/cpu:0'), tf.name_scope('input')as scope:
            roidb = load_gt_roidb(args.dataset, 'train', args.root_path, args.dataset_path, flip=args.flip)
            # print('roidb len =', len(roidb))

        batch_size = config.TRAIN.BATCH_IMAGES
        learning_rate = args.lr
        for epoch_count in range(args.begin_epoch, args.end_epoch):
            print('-'*40)
            # print('Epoch: {:d}'.format(epoch_count))

            for step in args.lr_step:
                if epoch_count == step:
                    learning_rate *= 1
                    print('******** learning_rate change to {}\n'.format(learning_rate))

            iter = len(roidb)/config.TRAIN.BATCH_IMAGES
            dataloader = DataLoader(roidb, batch_size, shuffle=True)

            saveloss=0
            for iter_count in range(int(iter)):

                input, labels = dataloader.get_next()

                input = input.transpose(0, 2, 3, 1)  # TODO : very important, change the format to NHWC
                # print('train: (BHWC)input.shape=', input.shape)
                # cls_score = sess.run(cls_score, feed_dict={data: input, landmark_weight: labels['face_landmark_weight']})
                # print('cls_score shape =', np.array(cls_score).shape)

                # real_label = labels['face_bbox_target'][0]
                # print('train: real label:=',real_label.shape)

                c_loss,b_loss,l_loss,r_loss,t_loss, t= sess.run((cls_loss, bbox_loss, landmark_loss, reg_loss, total_loss, train,),
                                                feed_dict={data: input,
                                                label: labels['face_label'],
                                                label_weight: labels['face_label_weight'],
                                                bbox_target: labels['face_bbox_target'],
                                                bbox_weight: labels['face_bbox_weight'],
                                                landmark_target: labels['face_landmark_target'],
                                                landmark_weight: labels['face_landmark_weight'],
                                                lr: learning_rate})
                saveloss=t_loss
                # if epoch_count == 0:
                s ='Epoch{}, Iter{}, Lr{}, cls_loss:{:.2f}, bbox_loss:{:.2f}, landmark_loss:{:.2f}, reg_loss:{:.2f}, total_loss:{:.2f}\n'.format(epoch_count,iter_count,learning_rate,c_loss, b_loss, 0.5*l_loss, r_loss, t_loss)
                print(s)
                f.write(s)
                # else:
                    # if iter_count%args.frequent ==0 :
                    # print(
                    #     'Epoch{} Iter{},, Lr{}, pos_loss:{:.2f}, neg_loss:{:.2f}, bbox_loss:{:.2f}, landmark_loss:{:.2f}, total_loss:{:.2f}'.format(
                    #             epoch_count,iter_count, learning_rate,p_loss,n_loss, b_loss, l_loss, t_loss))
                # if epoch_count == 0:
                #     print('Epoch{}, Lr{}, cls_loss:{:.2f}'.format(epoch_count,learning_rate,c_loss))
                # else:
                #     if iter_count%args.frequent ==0 :
                #         print(
                #             'Epoch{}, Lr{}, cls_loss:{:.2f}'.format(epoch_count, learning_rate, c_loss))

            f.write("epoch:{},loss={}\n".format(epoch_count,saveloss))

            # if saveloss <= lowest_loss+15:
            #     if saveloss<lowest_loss:
            #         lowest_loss = saveloss
            ckpt_file = '{}/loss={:.2f}.ckpt'.format(args.save_prefix,saveloss)
            retinaface_saver.save(sess,ckpt_file,global_step=epoch_count)
            print('save model!\n')
    f.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaFace')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    # generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=default.frequent, type=int)
    # parser.add_argument('--kvstore', help='the kv-store type', default=default.kvstore, type=str)
    # parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--flip', help='flip images', default=default.flip)
    parser.add_argument('--shuffle', help='random shuffle', default=default.shuffle)
    # e2e
    #parser.add_argument('--gpus', help='GPU device to train with', default='0,1,2,3', type=str)
    parser.add_argument('--retrain', help='retrain mode', default=config.retrain, type=bool)
    parser.add_argument('--pretrained', help='pretrained model prefix', default=default.pretrained, type=str)
    # parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=default.pretrained_epoch, type=int)
    parser.add_argument('--save_prefix', help='new model prefix', default=default.prefix, type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=default.end_epoch, type=int)
    parser.add_argument('--lr', help='base learning rate', default=default.lr, type=str)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default=default.lr_step, type=str)
    parser.add_argument('--use_ohem', help='use online hard mining', default=default.use_ohem, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Called with argument: %s' % args)

    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    train_net(args, ctx)

if __name__ == '__main__':
    main()
