# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg

class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes, [self._gt_image, self._gt_boxes, self._im_info], tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction', tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(rpn_cls_prob,rpn_bbox_pred,self._im_info,self._feat_stride,
                    self._anchors,self._num_anchors)
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,[rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal_top")

            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_layer_tf(rpn_cls_prob,rpn_bbox_pred,self._im_info,self._mode,self._feat_stride,
                    self._anchors,self._num_anchors)  #筛选超越边界的候选框，进行nms处理，得到前6k个候选框
            else:
                rois, rpn_scores = tf.py_func(proposal_layer,[rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                               self._feat_stride, self._anchors, self._num_anchors],
                                              [tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])  #[w*h*9,5]
            rpn_scores.set_shape([None, 1])  #[w*h*9,1]

        return rois, rpn_scores

    # Only use it if you have roi_pooling op written in tf.image
    def _roi_pool_layer(self, bootom, rois, name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bootom, rois,pooled_height=cfg.POOLING_SIZE,pooled_width=cfg.POOLING_SIZE,spatial_scale=1. / 16.)[0]

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME') #最大池化层

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        #根据候选框计算实际偏移量,用于计算误差
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")
            # 剔除越出边界的roi,计算边界偏移值,选出前后景,初始化权重

            rpn_labels.set_shape([1, 1, None, None]) #[1,1,h,w]
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4]) #[1,h,w,9*4]
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])#[1,h,w,9*4]
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])#[1,h,w,9*4]

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        #选出batchsize的候选框,计算真实偏移量
        #根据预测边界计算的实际偏移量
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, roi_scores

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))  #blob图像高除以16
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))   #blob图像宽除以16
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(height,width,self._feat_stride,self._anchor_scales,self._anchor_ratios)
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,[height, width,self._feat_stride, self._anchor_scales, self._anchor_ratios],
                                                    [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            # 截断的正态分布中输出随机值
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else: #标准正态分布的随机数
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            self._anchor_component()    #得到9*w*h/256个候选框
            # region proposal network
            rois = self._region_proposal(net_conv, is_training, initializer) #得到batchsize的roi,最多一半的前景
            # region of interest pooling
            if cfg.POOLING_MODE == 'crop':
                pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
            else:
                raise NotImplementedError

        fc7 = self._head_to_tail(pool5, is_training)
        with tf.variable_scope(self._scope, self._scope):
            # region classification
            cls_prob, bbox_pred = self._region_classification(fc7, is_training, initializer, initializer_bbox)

        self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2]) #前后景分数
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1]) #标签
            rpn_select = tf.where(tf.not_equal(rpn_label, -1)) #不等于-1的小标
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2]) #重整
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)

        return loss

    def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")  #[1,h,w,256]
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score') #[1,h,w,9*2],通过可训练的卷积计算前后景概率
        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')  #[1,9*h,w,2]
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape") #[1,9*h,w,2]
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred") #[1,9*h,w,1],得到窗口是前景还是后景
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob") #[1,h,w,9*2]
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred') #[1,h,w,9*4]
        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois") #根据预测偏移值,计算预测边界，根据nms筛选2k，并重构blob
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")  #剔除越出边界的roi,计算边界偏移值,选出前后景,初始化权重,self._anchor_targets
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):  #控制依赖关系,先运行完rpn_labels,才由下面
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois") #选出batchsize的候选框,计算真实偏移量
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois

        return rois

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')#全连接层,输出[None,7]
        cls_prob = self._softmax_layer(cls_score, "cls_prob") #softmax预测类别
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred") #最大可能性类别
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred') #全连接层,输出[None,4]

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios
        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)   #正则化，避免过拟合
        if cfg.TRAIN.BIAS_DECAY:   #是否使用正则化的参数
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob, bbox_pred = self._build_network(training) #返回候选框,对应标签类别,对应边界偏移值

        layers_to_output = {'rois': rois}

        for var in tf.trainable_variables(): #返回需要训练的变量列表
            self._train_summaries.append(var)

        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}

        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'], self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'], self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            train_op],
                                                                           feed_dict=feed_dict)


        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def createNewData(self,blobs):
        data = np.zeros((2, blobs['data'].shape[1], blobs['data'].shape[2], 3),
                        dtype=np.float32)
        im_info = np.zeros((2, len(blobs['im_info'])),
                           dtype=np.float32)
        boxes = np.zeros((2, blobs['gt_boxes'].shape[0], blobs['gt_boxes'].shape[1]),
                         dtype=np.float32)
        # print(blobs['gt_boxes'])
        for i in range(2):
            im = blobs['data'][0, :, :, :]  # 获取图像
            info = blobs["im_info"]
            box = blobs["gt_boxes"]
            data[i, 0:, 0:, :] = im  # 填充图像，其余为0
            im_info[i, :] = info
            boxes[i, :, :] = box
        # print(im_info.shape,im_info)
        # print(boxes.shape,boxes)
        newBlobs={"data":data}
        newBlobs['im_info']=im_info
        newBlobs["gt_boxes"]=boxes
        return newBlobs
    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'], self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                                     self._losses['rpn_loss_box'],
                                                                                     self._losses['cross_entropy'],
                                                                                     self._losses['loss_box'],
                                                                                     self._losses['total_loss'],
                                                                                     self._summary_op,
                                                                                     train_op],
                                                                                    feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'], self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)

