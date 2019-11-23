#encoding:utf-8
from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from distutils.util import strtobool
from ..config import config
import pdb
FIRSTN = None

def precision(score,label,pos_ind, neg_ind):
    # score = tf.Print(score, [tf.shape(score)], 'INFO: score shape=')
    # label = tf.Print(label, [tf.shape(label)], 'INFO: label shape=')
    prob = tf.nn.softmax(score)
    pre_label = tf.argmax(prob,axis=-1)
    pre_label = tf.to_int32(pre_label)
    # a = tf.constant(1,tf.int32)
    #计算正样本精度
    pos_real_label = tf.gather(label,pos_ind)
    pos_pre_label = tf.gather(pre_label, pos_ind)
    pos = tf.to_float(tf.equal(pos_pre_label,pos_real_label))
    pos_precision = tf.reduce_mean(pos)

    #计算负样本精度
    neg_real_label = tf.gather(label, neg_ind)
    neg_pre_label = tf.gather(pre_label, neg_ind)
    neg = tf.to_float(tf.equal(neg_pre_label, neg_real_label))
    neg_precision = tf.reduce_mean(neg)
    return pos_precision, neg_precision


def no_ohem(cls_scores, labels, cls_losses):
    pos_loss_sum=0
    neg_loss_sum=0
    pos_precision_sum = 0.
    neg_precision_sum = 0.

    for ibatch in range(config.TRAIN.BATCH_IMAGES):
        label = tf.slice(labels, begin=[ibatch, 0], size=[1, -1], name='label_slice')
        label = tf.squeeze(label)
        cls_loss = tf.slice(cls_losses, begin=[ibatch, 0], size=[1, -1], name='loss_slice')
        cls_loss = tf.squeeze(cls_loss)
        cls_score = tf.slice(cls_scores, begin=[ibatch, 0, 0], size=[1, -1, -1], name='score_slice')
        cls_score = tf.squeeze(cls_score)
        # cls_score = tf.Print(cls_score, [tf.shape(cls_score)], 'INFO: score shape=')
        score = cls_score[:, 1] - cls_score[:, 0]

        pos_ind = tf.where(label>0)
        pos_ind = tf.Print(pos_ind, [tf.shape(pos_ind),pos_ind], 'Debug message: A pos_ind=', first_n=FIRSTN, summarize=400)
        pos_score = tf.gather(score,pos_ind)
        pos_ind = tf.Print(pos_ind, [tf.shape(pos_score), pos_score],'Debug message: POS: score=', first_n=None, summarize=400, name='PRI_pos_score')
        pos_loss = tf.gather(cls_loss, pos_ind, name='gather_pos_loss')

        neg_ind = tf.where(tf.equal(label,0))
        shape = tf.shape(neg_ind)[0]
        random_ind = tf.random_uniform(shape=(128,),minval=0,maxval=shape,dtype=tf.int32, name='random_negind')
        neg_ind = tf.gather(neg_ind, random_ind)
        neg_score = tf.gather(score, neg_ind)
        neg_ind = tf.Print(neg_ind, [tf.shape(neg_score), neg_score], 'Debug message: NEG: score=', first_n=None, summarize=15, name='PRI_neg_score')
        neg_ind = tf.Print(neg_ind, [tf.shape(neg_ind) , neg_ind], 'Debug message: B neg_ind=', first_n=FIRSTN, summarize=400)
        neg_loss = tf.gather(cls_loss, neg_ind)

        #计算精度
        pos_pre, neg_pre = precision(cls_score, label, pos_ind, neg_ind)
        pos_precision_sum = pos_precision_sum + pos_pre
        neg_precision_sum = neg_precision_sum + neg_pre

        pos_loss = tf.Print(pos_loss, [ibatch,tf.shape(cls_loss), tf.shape(pos_loss), tf.shape(neg_loss)], 'Debug message: softmax_loss_loss=', first_n=FIRSTN, summarize=400)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
        pos_loss_sum =pos_loss_sum+pos_loss
        neg_loss_sum = neg_loss_sum +neg_loss

    pos_loss_sum = pos_loss_sum/config.TRAIN.BATCH_IMAGES
    neg_loss_sum = neg_loss_sum/config.TRAIN.BATCH_IMAGES
    pos_precision_sum = pos_precision_sum/config.TRAIN.BATCH_IMAGES
    neg_precision_sum = neg_precision_sum/config.TRAIN.BATCH_IMAGES

    pos_loss_sum = tf.Print(pos_loss_sum, [pos_precision_sum, neg_precision_sum],'INFO: print once per batch! one batch precision= ', summarize=100, name='PRI_final_Loss')
    # precision_sum = tf.Print(precision_sum, [precision_sum], 'ONE batch precision = ',)
    return pos_loss_sum, neg_loss_sum

def ohem(cls_scores, labels, cls_losses):
    '''
    :param cls_score: # BS, ANCHORS, 2
    :param labels: # BS, ANCHORS
    :param is_training:
    :return:
    '''
    print('*'*30,'get in ohem !!!!!!!!\n\n')

    pos_loss_sum=0
    neg_loss_sum=0
    pos_precision_sum = 0.
    neg_precision_sum = 0.

    with tf.name_scope('ohem'):
        for ibatch in range(config.TRAIN.BATCH_IMAGES):
        # for ibatch in range(1):
        #     print('ibatch = ',ibatch)
            label = tf.slice(labels, begin=[ibatch,0],size=[1,-1],name='label_slice')
            label = tf.squeeze(label)
            cls_score = tf.slice(cls_scores, begin=[ibatch, 0, 0], size=[1, -1, -1],name='score_slice')
            cls_score = tf.squeeze(cls_score)
            cls_loss = tf.slice(cls_losses, begin=[ibatch, 0], size=[1, -1],name='loss_slice')
            cls_loss = tf.squeeze(cls_loss)

            # label = tf.Print(label, [ibatch,tf.shape(label), label], 'Debug message: label =', first_n=10, summarize=200)
            # cls_score = tf.Print(cls_score, [ibatch,tf.shape(cls_score), cls_score], 'Debug message: cls_score =',first_n=10, summarize=200)
            score = cls_score[:, 1] - cls_score[:, 0]
            # score = tf.Print(score, [ibatch,tf.shape(score), score], 'Debug message: after_sub_score =',first_n=10, summarize=200)
            # cls_loss = tf.Print(cls_loss, [ibatch,tf.shape(cls_loss), cls_loss], 'Debug message: cls_loss =',first_n=10, summarize=200)

            num_pos_label = tf.reduce_sum(label,name='label_reduce_sum')
            # num_pos_label = tf.Print(num_pos_label,[num_pos_label],'Debug message: num_pos_label= ')
            num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
            # num_fg=1
            ans = tf.greater(num_pos_label, num_fg, name='pos_greater')

            pos_inds = tf.where(label > 0, name='pos_where')
            # pos_inds = tf.Print(pos_inds, [ibatch, tf.shape(pos_inds), pos_inds], 'AAAAAAA POS_INDS =',first_n=FIRSTN, summarize=15,name='PRI_pos_inds')
            neg_inds = tf.where(tf.equal(label, 0),name='neg_where')

            # pdb.set_trace()

            def pos_top(k):
                # print('begin to compute pos loss')
                pos_ohem_scores = tf.gather(score, pos_inds,name='pos_ohem_score')
                pos_ohem_scores = tf.squeeze(pos_ohem_scores)
                # pos_ohem_scores = tf.Print(pos_ohem_scores, [tf.shape(pos_ohem_scores),pos_ohem_scores], 'Debug message: pos_ohem_scores= ', first_n=100, summarize=7)
                # pdb.set_trace()
                # TODO 对正样本，要选择sub_score越小的，表示分类越错误
                pos_ohem_scores = -pos_ohem_scores
                topk_res = tf.nn.top_k([pos_ohem_scores], k, sorted=True, name='pos_top_k')  # (64)
                pri_topk_score = -topk_res[0]
                pos_order_inds = topk_res[1]
                pos_order_inds = tf.Print(pos_order_inds, [tf.shape(pri_topk_score),pri_topk_score], 'Debug message: POS: topk_score=',
                                          first_n=FIRSTN, summarize=20, name='PRI_pos_top_score')
                # pos_order_inds = tf.Print(pos_order_inds, [tf.shape(pos_order_inds),pos_order_inds], '1111111111Debug message: pos_order_inds= ', first_n=100, summarize=15)
                pos_sampled_inds = tf.gather(pos_inds, pos_order_inds,name='pos_sampled_inds')
                pos_sampled_inds = tf.squeeze(pos_sampled_inds)
                # pos_sampled_inds = tf.Print(pos_sampled_inds, [ibatch,tf.shape(pos_sampled_inds),pos_sampled_inds], '2222222222Debug message: pos_sampled_inds= ', first_n=100, summarize=15)
                pos_loss = tf.gather(cls_loss, pos_sampled_inds,name='gather_pos_loss')
                # pos_loss = tf.Print(pos_loss, [ibatch, tf.shape(pos_loss),pos_loss], '3333333333Debug message: pos_loss= ', first_n=FIRSTN, summarize=15,name='PRI_pos_1_loss')
                pos_loss = tf.reduce_sum(pos_loss,name='pos_reducesum')
                return pos_loss, pos_sampled_inds

            def pos0():
                return 0. , tf.to_int64(0)

            def true_fun():
                # print('\nget into true_fun, pos_inds > num_fg(64)')
                pos_loss, pos_sampled_inds = tf.cond(tf.equal(num_pos_label, 1), pos0, lambda:pos_top(num_fg),name='cond_equal_true_fun')

                # print('begin to compute neg loss')
                num_bg = int(config.TRAIN.RPN_BATCH_SIZE) - num_fg
                neg_ohem_scores = tf.gather(score, neg_inds,name='Tgather_neg_score')
                neg_ohem_scores = tf.squeeze(neg_ohem_scores)
                #TODO 对负样本，要选择sub_score越大的，表示分类越错误
                neg_order_inds = tf.nn.top_k(neg_ohem_scores, num_bg, sorted=True, name='Tneg_top_k')[1]  # (256-num_fg)
                neg_sampled_inds = tf.gather(neg_inds, neg_order_inds,name='Tgather_neg_sample')
                # neg_sampled_inds = tf.Print(neg_sampled_inds, [ibatch, tf.shape(neg_sampled_inds)],
                #                             'Debug message: in true fun neg_sampled_inds= ', first_n=FIRSTN, summarize=15,name='11TPRI_neg_sample')
                neg_loss = tf.gather(cls_loss, neg_sampled_inds,name='Tgather_negLoss')
                # neg_loss = tf.reduce_sum(neg_loss,name='Tneg_recudesum')
                return pos_loss, neg_loss, pos_sampled_inds, neg_sampled_inds

            def false_fun():
                # print('get into false_fun, pos_inds <= num_fg(64)')
                pos_loss, pos_sampled_inds = tf.cond(tf.equal(num_pos_label, 0), pos0, lambda:pos_top(num_pos_label),name='cond_equal_false_fun')

                # print('begin to compute neg loss')
                num_bg = int(config.TRAIN.RPN_BATCH_SIZE) - num_pos_label
                neg_ohem_scores = tf.gather(score, neg_inds, name='Fgather_neg_score')
                neg_ohem_scores = tf.squeeze(neg_ohem_scores)
                #TODO 对负样本，要选择sub_score越大的，表示分类越错误
                topk_res = tf.nn.top_k(neg_ohem_scores, num_bg, sorted=True, name='Fneg_top_k')  # (256-num_fg)
                pri_topk_score = topk_res[0]
                neg_order_inds = topk_res[1]
                neg_order_inds = tf.Print(neg_order_inds,[tf.shape(pri_topk_score), pri_topk_score],'Debug message: NEG: topk_score=', first_n=FIRSTN, summarize=12,name='PRI_neg_top_score')
                neg_sampled_inds = tf.gather(neg_inds, neg_order_inds,name='Fgather_neg_sample')
                # neg_sampled_inds = tf.Print(neg_sampled_inds, [ibatch, tf.shape(neg_sampled_inds)],
                #                             'Debug message: in false fun neg_sampled_inds= ', first_n=FIRSTN, summarize=100,name='22FPRI_neg_sample')
                neg_loss = tf.gather(cls_loss, neg_sampled_inds,name='Fgather_negLoss')
                # neg_loss = tf.reduce_sum(neg_loss,name='Fneg_recudesum')
                return pos_loss, neg_loss, pos_sampled_inds, neg_sampled_inds

            # ans = tf.Print(ans, [ans], 'Debug message: greater_op answer= ', first_n=FIRSTN, summarize=100,name='PRI_greater_op')
            posLoss, negLoss, posinds, neginds = tf.cond(ans, true_fun, false_fun,name='Cond_greater')
            # negLoss = tf.Print(negLoss, [tf.shape(negLoss)], '55555555555Debug message: negLoss= ', first_n=FIRSTN, summarize=100,
            #                name='PRI_negLoss')

            negLoss = tf.reduce_sum(negLoss, name='ngeLoss_recudesum')
            # posLoss = tf.Print(posLoss, [ibatch, posLoss, negLoss], 'Debug message: one image Loss= ', first_n=FIRSTN, summarize=100,
            #                name='PRI_posLoss')
            # 计算精度
            pos_pre, neg_pre = precision(cls_score, label, posinds, neginds)
            pos_precision_sum = pos_precision_sum + pos_pre
            neg_precision_sum = neg_precision_sum + neg_pre

            pos_loss_sum = pos_loss_sum+posLoss
            neg_loss_sum = neg_loss_sum+negLoss
            # pos_loss_sum = tf.Print(pos_loss_sum, [ibatch, pos_loss_sum, neg_loss_sum], 'Debug message: final sum Loss= ', first_n=FIRSTN,
            #                summarize=100,
            #                name='PRI_final_Loss')
        pos_precision_sum = pos_precision_sum / config.TRAIN.BATCH_IMAGES
        neg_precision_sum = neg_precision_sum / config.TRAIN.BATCH_IMAGES
        pos_loss_sum = pos_loss_sum/config.TRAIN.BATCH_IMAGES
        neg_loss_sum = neg_loss_sum/config.TRAIN.BATCH_IMAGES
        pos_loss_sum = tf.Print(pos_loss_sum, [pos_loss_sum, neg_loss_sum], 'only print once! Debug message: one batch Loss= ',summarize=100,name='PRI_final_Loss')
        pos_loss_sum = tf.Print(pos_loss_sum, [pos_precision_sum, neg_precision_sum],
                                'INFO: print once per batch! one batch precision= ', summarize=100,
                                name='PRI_final_Loss')
    # return pos_loss_sum, neg_loss_sum, pos_inds, neg_inds
    return pos_loss_sum, neg_loss_sum

