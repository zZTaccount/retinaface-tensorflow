'''
common.py
用于定义主干网络之后接的fpn+ssh混合网络
'''
import tensorflow as tf
from tensorflow.contrib import slim
from rcnn.config import config,default
from rcnn.model.loss import smoothL1,softmax_loss,focal_loss
from rcnn.model.ohem import no_ohem, ohem, full_compute
from rcnn.model import resnet_v2

def my_arg_scope():
    conv2d_scope = slim.arg_scope([slim.conv2d],padding='SAME',activation_fn=None,data_format=config.data_format,weights_regularizer=slim.l2_regularizer(scale=0.0001),biases_initializer=None)
    bn_scope=slim.arg_scope([slim.batch_norm],decay=config.bn_decay,activation_fn=tf.nn.relu,is_training=config.is_training,data_format=config.data_format)
    with conv2d_scope,bn_scope as scope:
        return scope

def conv_act_layer(from_layer, name, num_filter, kernel_size=(1,1), stride=(1,1),act_type='relu'):
    with slim.arg_scope(my_arg_scope()):
        # print('get in the conv_act_layer func! data_format=',config.data_format)
        conv2d = slim.conv2d(from_layer, num_filter, kernel_size, stride, data_format=config.data_format, scope=name)
        if len(act_type)>0:
            bn_relu = slim.batch_norm(conv2d,scope=name+'hhhhhbn_relu')
            return bn_relu
        else :
            bn = slim.batch_norm(conv2d,scope=name+'bn',activation_fn=None)
            return bn

def ssh_context_module(body, num_filter, name):
  conv_dimred = conv_act_layer(body, name+'conv1', num_filter, kernel_size=(3, 3), stride=(1, 1),act_type='relu')
  conv5x5 = conv_act_layer(conv_dimred, name+'conv2',num_filter, kernel_size=(3, 3), stride=(1, 1),act_type='')
  conv7x7_1 = conv_act_layer(conv_dimred, name+'_conv3_1',num_filter, kernel_size=(3, 3), stride=(1, 1),act_type='relu')
  conv7x7 = conv_act_layer(conv7x7_1, name+'_conv3_2',num_filter, kernel_size=(3, 3), stride=(1, 1),act_type='')
  return (conv5x5, conv7x7)

def ssh_detection_module(body, num_filter, name):
  assert num_filter%4==0
  conv3x3 = conv_act_layer(body, name+'_conv1',num_filter//2, kernel_size=(3, 3), stride=(1, 1), act_type='')
  _filter = num_filter//4
  conv5x5, conv7x7 = ssh_context_module(body, _filter, name+'_context')
  if config.data_format=='NHWC':
      concat_dim=3
  else:
      concat_dim=1
  ret = tf.concat([conv3x3,conv5x5,conv7x7],axis=concat_dim,name=name+'_concat')
  ret = tf.nn.relu(ret,name=name+'_concat_relu')
  return ret

def upsampling(data):
    # ret = slim.conv2d_transpose(data,num_outputs=num_filter,stride=(2,2),kernel_size=(2,2),data_format=config.data_format, padding='SAME',scope=name)
    shape = tf.shape(data)
    h = shape[1]
    w = shape[2]
    ret = tf.image.resize_images(data,size=(h*2,w*2),method=1) #最近邻插值
    return ret

def crop(template, tensor):
    '''按照模板的形状对tensor进行裁剪'''
    shape = tf.shape(template)
    h=shape[1]
    w=shape[2]
    ret = tensor[:,:h,:w,:]
    return ret

def get_sym_conv(end_points):

    F1 = config.HEAD_FILTER_NUM
    F2 = F1
    # c1 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_1/bottleneck_v2/conv1/Relu:0")  # stride8 80*80
    # c2 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_1/bottleneck_v2/conv1/Relu:0")  # stride16 40*40
    # c3 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/Relu:0")  #stride32 20*20

    c1 = end_points['resnet_v2_50/block1']
    c2 = end_points['resnet_v2_50/block2']
    c3 = end_points['resnet_v2_50/block4']

    # c3 = tf.Print(c3,[tf.shape(sym)],'------------------------------------INFO: resnet output = ',first_n=10,summarize=100)

    c3 = conv_act_layer(c3, 'rf_c3_lateral',F2, kernel_size=(1, 1), stride=(1, 1), act_type='relu') #i,20,20,256
    c3_up = upsampling(c3)  #i,40,40,256
    c2_lateral = conv_act_layer(c2, 'rf_c2_lateral', F2, kernel_size=(1, 1), stride=(1, 1), act_type='relu') #i,40,40,256
    if config.USE_CROP:
        c3_up = crop(c3_up, c2_lateral) #TODO 输入大小不一样的图片，可能需要裁剪
    c2 = c2_lateral + c3_up #i,40,40,256
    c2 = conv_act_layer(c2, 'rf_c2_aggr',F2, kernel_size=(3, 3),stride=(1, 1), act_type='relu')
    c1_lateral = conv_act_layer(c1, 'rf_c1_red_conv', F2, kernel_size=(1, 1), stride=(1, 1), act_type='relu')
    c2_up = upsampling(c2) #(i, 80, 80, 256)
    if config.USE_CROP:
        c2_up = crop(c2_up, c1_lateral)
    c1 = c1_lateral + c2_up #(i, 80, 80, 256)
    c1 = conv_act_layer(c1, 'rf_c1_aggr',F2, kernel_size=(3, 3), stride=(1, 1), act_type='relu')
    m1 = ssh_detection_module(c1, F2, 'rf_c1_det')
    m2 = ssh_detection_module(c2, F1, 'rf_c2_det')
    m3 = ssh_detection_module(c3, F1, 'rf_c3_det')
    assert len(config.RPN_ANCHOR_CFG)==3
    ret = {8: m1, 16:m2, 32: m3}
    return ret


def get_out(conv_fpn_feat, prefix, landmark=False):
    out_group = {}
    # imgs = tf.shape(conv_fpn_feat[32])[0]
    # imgs = tf.Print(imgs,[imgs],'Debug message: common/get_out: imgs=',first_n=3,summarize=100)
    for stride in config.RPN_FEAT_STRIDE:
        bbox_pred_len = 4
        landmark_pred_len = 10
        num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
        # print('model/common: num_anchor = ',num_anchors)
        rpn_relu = conv_fpn_feat[stride]
        with slim.arg_scope([slim.conv2d],kernel_size=(1,1), padding='SAME', weights_regularizer=slim.l2_regularizer(scale=0.0001), stride=(1,1)):
            rpn_cls_score = slim.conv2d(rpn_relu, num_outputs=2*num_anchors, scope='%s_rpn_cls_score_stride%d'%(prefix,stride),data_format=config.data_format) #i,h,w,4
            rpn_bbox_pred = slim.conv2d(rpn_relu, num_outputs=bbox_pred_len*num_anchors, scope='%s_rpn_bbox_pred_stride%d'%(prefix,stride),data_format=config.data_format)#i,h,w,8
            # rpn_cls_score = tf.Print(rpn_cls_score, [tf.shape(rpn_cls_score)], 'Debug message: rpn_cls_score output shape=', summarize=15)
            # rpn_bbox_pred = tf.Print(rpn_bbox_pred, [tf.shape(rpn_bbox_pred)], 'Debug message: rpn_bbox_pred output shape=', summarize=15)

        if landmark:
            with slim.arg_scope([slim.conv2d], kernel_size=(1, 1), padding='SAME', weights_regularizer=slim.l2_regularizer(scale=0.0001), stride=(1, 1)):
                rpn_landmark_pred = slim.conv2d(rpn_relu, num_outputs=landmark_pred_len*num_anchors, scope='%s_rpn_landmark_pred_stride%d' % (prefix, stride),data_format=config.data_format)#i,h,w,20

        # if config.data_format == 'NHWC':
        #     with tf.name_scope('transpose'):
                # rpn_cls_score = tf.transpose(rpn_cls_score, perm=(0, 3, 1, 2))
                # rpn_bbox_pred = tf.transpose(rpn_bbox_pred, perm=(0, 3, 1, 2),name='bbox_transpose')
                # rpn_landmark_pred = tf.transpose(rpn_landmark_pred, perm=(0, 3, 1, 2),name='landmark_transpose')

        out_group['rpn_cls_score_stride%s'%stride] =rpn_cls_score #(N,H,W,4)
        out_group['rpn_bbox_pred_stride%s'%stride] =rpn_bbox_pred #(N,H,W,8)
        out_group['rpn_landmark_pred_stride%s'%stride] =rpn_landmark_pred #(N,H,W,20)
    return out_group

def stride_concat(out_group):
    reshape_group={}
    imgs = tf.shape(out_group['rpn_cls_score_stride32'])[0]
    # imgs = tf.Print(imgs, [imgs], 'Debug message: common/stride_concat: imgs=', first_n=3, summarize=100)
    for s in config.RPN_FEAT_STRIDE:
        reshape_group['rpn_cls_score_reshape_stride%s'%s] = tf.reshape(out_group['rpn_cls_score_stride%s'%s], shape=(imgs, -1, 2),
                                           name="face_rpn_cls_score_reshape_stride%s" %s)  # i,-1,2
        reshape_group['rpn_bbox_pred_reshape_stride%s'%s] = tf.reshape(out_group['rpn_bbox_pred_stride%s'%s], shape=(imgs, -1, 8),
                                           name="face_rpn_bbox_pred_reshape_stride%s"%s)  # i,-1,8
        reshape_group['rpn_landmark_pred_reshape_stride%s'%s] = tf.reshape(out_group['rpn_landmark_pred_stride%s'%s], shape=(imgs, -1, 20),
                                               name="face_rpn_landmark_pred_reshape_stride%s"%s)  # i,-1,20

    ret_group = {}
    ret_group['rpn_cls_score_reshape'] = tf.concat([reshape_group['rpn_cls_score_reshape_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=1)
    ret_group['rpn_bbox_pred_reshape'] = tf.concat([reshape_group['rpn_bbox_pred_reshape_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE], axis=1)
    ret_group['rpn_landmark_pred_reshape'] = tf.concat([reshape_group['rpn_landmark_pred_reshape_stride%s' % stride] for stride in config.RPN_FEAT_STRIDE],axis=1)
    return ret_group

def get_loss(ret_group, label, label_weight, bbox_target, bbox_weight, landmark_target ,landmark_weight,prefix):
    cls_score = ret_group['rpn_cls_score_reshape'] # i,16800,2
    bbox_pred = ret_group['rpn_bbox_pred_reshape'] # i,8400,8
    landmark_pred = ret_group['rpn_landmark_pred_reshape'] # i,8400,20
    # cls_score = tf.Print(cls_score, [tf.shape(cls_score)], 'Debug message: cls_score output shape=',summarize=15)
    # bbox_pred = tf.Print(bbox_pred, [tf.shape(bbox_pred)], 'Debug message: bbox_pred output shape=',summarize=15)
    # landmark_pred = tf.Print(landmark_pred, [tf.shape(landmark_pred)], 'Debug message: landmark_pred output shape=', summarize=15)

    # 计算cls loss
    if default.use_focalLoss:
        cls_loss = focal_loss(cls_score, label, alpha=config.TRAIN.FOCALLOSS_alpha, gama=config.TRAIN.FOCALLOSS_gama)

    else:
        label_out = label*label_weight #所有正样本为1，其他都为0
        cls_losses = softmax_loss(cls_score, label_out)

        if default.use_ohem:
            pos_loss, neg_loss = ohem(cls_score, label, cls_losses) # 使用label, 0.3-0.5抛弃
            # pos_loss, neg_loss = full_compute(cls_score, label, cls_losses)
        else:
            pos_loss, neg_loss = no_ohem(cls_score, label, cls_losses) # 使用label, 0.3-0.5抛弃
        cls_loss = pos_loss* config.TRAIN.posloss_weight + neg_loss * config.TRAIN.negloss_weight

    # 计算bbox loss
    # bbox_pred = tf.Print(bbox_pred, [tf.shape(bbox_pred), bbox_pred], 'INFO: bbox_pred =', summarize=10000)
    # bbox_target = tf.Print(bbox_target, [tf.shape(bbox_target), bbox_target], 'INFO: bbox_target =', summarize=500)
    bbox_diff = bbox_pred - bbox_target

    # bbox_weight = tf.Print(bbox_weight, [tf.shape(bbox_weight)], 'INFO: bbox_weight =', summarize=400)
    bbox_diff = tf.where(tf.equal(bbox_weight, 1), bbox_diff, bbox_diff*0) #TODO important
    # target = tf.Print(target, [tf.shape(target)], 'INFO: after select target =', summarize=500)
    # bbox_diff = tf.Print(bbox_diff, [tf.shape(bbox_diff), bbox_diff], 'INFO: before: bbox_diff =', summarize=10000)

    # bbox_diff = bbox_diff * bbox_weight
    num = tf.reduce_sum(bbox_weight)/4

    bbox_diff = tf.Print(bbox_diff, [num], 'INFO: valid bbox num =', summarize=500)
    bbox_loss = smoothL1(bbox_diff, sigma=config.TRAIN.SMOOTHL1_sigma)
    # bbox_loss = bbox_loss / num
    slim.losses.add_loss(bbox_loss) #使用slim来管理loss

    # ret_group.append(mx.sym.BlockGrad(bbox_weight))

    # 计算landmark loss
    if config.FACE_LANDMARK:
        # landmark_pred = tf.Print(landmark_pred, [tf.shape(landmark_pred), landmark_pred], 'Debug message: landmark_pred =',
        #                      summarize=20)
        # landmark_target = tf.Print(landmark_target, [tf.shape(landmark_target), landmark_target],
        #                        'Debug message: landmark_target =', summarize=20)
        ld_num = tf.reduce_sum(landmark_weight) / 10
        landmark_diff = landmark_pred - landmark_target
        landmark_diff = landmark_diff * landmark_weight
        landmark_diff = tf.Print(landmark_diff, [ld_num], 'INFO: valid landmark num =', summarize=500)
        landmark_loss = smoothL1(landmark_diff,sigma=config.TRAIN.SMOOTHL1_sigma)
        slim.losses.add_loss(landmark_loss)  # 使用slim来管理loss

    # loss_group.append(mx.sym.BlockGrad(landmark_weight))
    return cls_loss, bbox_loss, landmark_loss

def get_pred(data):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(data, num_classes=0, is_training=False, global_pool=False)
    conv = get_sym_conv(net)
    ret = get_out(conv, 'face', config.FACE_LANDMARK)
    return ret