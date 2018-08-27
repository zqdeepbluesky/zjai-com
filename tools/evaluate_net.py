# -----------------------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 8/9/2018 4:34 PM
# @Author  : sunyonghai
# @Software: ZJ_AI
# -----------------------------------------------------
from lib.model.config import cfg,cal_data_aug_code
import cv2
import numpy as np
import tensorflow as tf
from lib.extra_utils.xml_store import *
from lib.extra_utils import io_utils,cal_acc
from tools import predict
from lib.model.nms_wrapper import nms
from lib.datasets import voc_eval
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_test_images_from_txt(data_dir,txt_name):
    io_utils.mkdir(os.path.join(data_dir,"Annotations_test"))
    jpg_path=os.path.join(data_dir,'JPEGImages')
    xml_path=os.path.join(data_dir,"Annotations_test")
    test_txt_file=os.path.join(data_dir,'ImageSets','Main','{}.txt'.format(txt_name))
    jpg_files=[]
    with open(test_txt_file,'r') as f:
        for line in f.readlines():
            jpg_files.append(os.path.join(jpg_path,line.replace("\n","")+".jpg"))
    if len(jpg_files)==0:
        print("Please load some Images in the {}!".format(jpg_path))
    return jpg_files,xml_path

def write_report(f,prec, recall, tp_sum, fp_sum, fn_sum, d_sum, t_sum):
    f.write("-->precsion: {}\n".format(float(prec)))
    f.write("-->recall: {}\n".format(float(recall)))
    f.write("-->tp_num: {}\n".format(tp_sum))
    f.write("-->fp_num: {}\n".format(fp_sum))
    f.write("-->fn_num: {}\n".format(fn_sum))
    f.write("-->model detect num : {}\n".format(d_sum))
    f.write("-->actual num : {}\n\n".format(t_sum))

def get_model(batch_model,iter,demo_net,CLASSES):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    model_dir = batch_model
    import datetime
    date = str(datetime.datetime.now())
    date_prefix = demo_net + '_' + date[0:date.find(":")].replace(" ", '-')
    model_data = date_prefix + '_iter_{:d}'.format(iter) + '.ckpt'
    tf_model = predict.get_tf_model(model_dir, model_data)
    print(tf_model)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess_test = tf.Session(config=tfconfig)
    # load network
    saver, net_test = predict.load_model(sess_test, demo_net, tf_model, len(CLASSES))
    return sess_test,net_test,model_dir,model_data

def eval_net(batch_model,iter,CLASSES,net_name,predict_dir,test_package,extra_test_package=[]):
    test_g=tf.Graph()
    with test_g.as_default():
        sess_test, net_test, model_dir,model_data=get_model(batch_model,iter,net_name,CLASSES)
        evaluate_model(sess_test, net_test, model_dir, model_data, CLASSES, predict_dir, test_package,
                   extra_test_package)

def evaluate_model(sess_test, net_test, model_dir,model_data,CLASSES,predict_dir,test_package,extra_test_package):
    test_infos = []
    packages = []
    for package in test_package:
        packages.append(package)
        jpg_files, xml_path = load_test_images_from_txt(os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package),'test')
        aps = predict_images(sess_test, net_test, jpg_files, xml_path, CLASSES)
        test_xml_path = os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package, 'Annotations_test')
        true_xml_path = os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package, 'Annotations')
        test_info = cal_acc.cal_model_acc(test_xml_path, true_xml_path)
        test_info_label = cal_acc.cal_label_acc(test_xml_path, true_xml_path, CLASSES)
        for index in np.argsort(aps):
            test_infos.append("{},{},{},{},{}".format(model_data.split(".")[0], package, CLASSES[index + 1],test_info_label[index+1], aps[index]))
        test_infos.append("{},{},total,{},{}".format(model_data.split(".")[0], package, test_info, np.nanmean(aps)))
    for package in extra_test_package:
        packages.append(package)
        jpg_files, xml_path = load_test_images_from_txt(os.path.join(predict_dir, package), 'trainval_test')
        aps = predict_images(sess_test, net_test, jpg_files, xml_path, CLASSES)
        test_xml_path = os.path.join(predict_dir, package, 'Annotations_test')
        true_xml_path = os.path.join(predict_dir, package, 'Annotations')
        test_info = cal_acc.cal_model_acc(test_xml_path, true_xml_path)
        test_info_label = cal_acc.cal_label_acc(test_xml_path, true_xml_path, CLASSES)
        for index in np.argsort(aps):
            test_infos.append("{},{},{},{},{}".format(model_data.split(".")[0], package, CLASSES[index + 1], test_info_label[index+1],aps[index]))
        test_infos.append("{},{},total,{},{}".format(model_data.split(".")[0], package, test_info, np.nanmean(aps)))
    tb = cal_acc.get_tabs(test_infos)
    tb = cal_acc.summary_tb(tb, test_infos)
    print(tb)
    aug_code=os.path.split(model_dir)[1][:8]
    txt_save_path = os.path.abspath(os.path.join(model_dir,'..','{}_batch_test_result'.format(aug_code)))
    if not os.path.exists(txt_save_path):
        io_utils.mkdir(txt_save_path)
    txt_save_path = os.path.join(txt_save_path, model_data.split(".")[0] + "_test_result")
    cal_acc.save_tb_in_txt(txt_save_path, tb)
    cal_acc.save_tb_in_xml(txt_save_path, tb)

def set_all_boxes(all_boxes,index,scores, boxes,CLASSES,max_per_image=100,thresh=0.):
    for j in range(1, len(CLASSES)):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        all_boxes[j][index] = cls_dets  # 3维

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][index][:, -1] for j in range(1, len(CLASSES))])
        if len(image_scores) > max_per_image:  # 若果超过100个，那么就选取置信度top100的
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, len(CLASSES)):
                keep = np.where(all_boxes[j][index][:, -1] >= image_thresh)[0]
                all_boxes[j][index] = all_boxes[j][index][keep, :]
    return all_boxes

def _init_rec(jpg_files,classname):
    recs={}
    for i, imagename in enumerate(jpg_files):
        recs[imagename] = voc_eval.parse_rec(imagename.replace("JPEGImages", "Annotations").replace(".jpg", ".xml"))
    class_recs = {}
    npos = 0
    for imagename in jpg_files:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}
    return npos,class_recs

def cal_tp_and_fp(image_ids,confidence,BB,class_recs,ovthresh):
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                ovmax, jmax = cal_acc.cal_iou(BBGT, bb)
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
    return tp,fp

def do_ap_eval(jpg_files,CLASSES,all_boxes,ovthresh=0.5):
    result_list=get_result_list(jpg_files, CLASSES, all_boxes)
    aps=[]
    for cls_ind,classname in enumerate(CLASSES):
        if classname == '__background__':
            continue
        npos, class_recs=_init_rec(jpg_files, classname)
        image_ids = [x.split(' ')[0] for x in result_list[cls_ind]]
        confidence = np.array([float(x.split(" ")[1]) for x in result_list[cls_ind]])
        BB = np.array([[float(z) for z in x.split(" ")[2:]] for x in result_list[cls_ind]])

        tp,fp=cal_tp_and_fp(image_ids, confidence, BB, class_recs, ovthresh)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = cal_acc.voc_ap(rec, prec, use_07_metric=False)
        aps+=[ap]
    return aps





def get_result_list(jpg_files,CLASSES,all_boxes):
    result_list=[[] for i in range(len(CLASSES))]
    for cls_ind, cls in enumerate(CLASSES):
        if cls == '__background__':
            continue

        for im_ind, im in enumerate(jpg_files):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            for k in range(dets.shape[0]):
                result_list[cls_ind].append('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(im, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,
                                                                   dets[k, 2] + 1, dets[k, 3] + 1))
    return result_list

def predict_images(sess,net,jpg_files,xml_path,CLASSES):
    all_boxes = [[[] for _ in range(len(jpg_files))] for _ in range(len(CLASSES))]
    index=0
    for image in jpg_files:
        sys.stdout.write("\r")
        sys.stdout.write('Demo for {}'.format(image))
        sys.stdout.flush()
        im = cv2.imread(image)
        scores, boxes = predict._detect_image(sess, net, im)
        result_data= predict.predict_image(scores, boxes,CLASSES)
        save_data_into_xml(image,im,xml_path,result_data)

        all_boxes=set_all_boxes(all_boxes, index, scores, boxes, CLASSES)
        index+=1
    aps=do_ap_eval(jpg_files, CLASSES, all_boxes, ovthresh=0.5)
    return aps




# imdb_name='voc_2007_trainval'
# iter='160020'
# demo_net='vgg16'
# predict_dir='/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data'
# package_data=['test_data-2018-07-24']
# test_model(imdb_name,iter,demo_net,predict_dir,package_data)