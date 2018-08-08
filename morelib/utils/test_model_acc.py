from lib.model.config import cfg

import tensorflow as tf
import os.path as osp
from morelib.utils.xml_store import *
from lib.datasets import pascal_voc
from zjai_createData import zjai_6_comparison
from morelib.utils import io_utils,prepare_model
from morelib.bin import predict_batch

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

def get_model(batch_model,iter,demo_net):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    model_dir = batch_model
    model_data = "{}_faster_rcnn_iter_{}.ckpt".format(demo_net, iter)
    tf_model = prepare_model.get_tf_model(model_dir, model_data)
    print(tf_model)
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess_test = tf.Session(config=tfconfig)
    # load network
    CLASSES = pascal_voc.read_classes(osp.join(cfg.ROOT_DIR, 'data', 'cfgs', 'com_classes.txt'))
    print(len(CLASSES))
    saver, net_test = prepare_model.load_model(sess_test, demo_net, tf_model, len(CLASSES))
    return sess_test,net_test,model_dir,model_data,CLASSES

def test_model(batch_model,iter,net_name,predict_dir,test_package,extra_test_package=[]):
    test_g=tf.Graph()
    with test_g.as_default():
        sess_test, net_test, model_dir,model_data,CLASSES=get_model(batch_model,iter,net_name)
        acc_list=[]
        for package in test_package:
            jpg_files, xml_path = load_test_images_from_txt(os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package),'test')
            predict_batch.predict_images(sess_test, net_test, jpg_files, xml_path,CLASSES)
            acc=zjai_6_comparison.compare_from_xml(xml_path,xml_path[:-5])
            acc_list.append(acc)
        for package in extra_test_package:
            jpg_files, xml_path = load_test_images_from_txt(os.path.join(predict_dir, package),'trainval')
            predict_batch.predict_images(sess_test, net_test, jpg_files, xml_path,CLASSES)
            acc=zjai_6_comparison.compare_from_xml(xml_path,xml_path[:-5])
            acc_list.append(acc)
        with open(osp.join(model_dir,'{}_model_test.log'.format(net_name)),'a+') as f:
            f.write("-----------{} model test result------------\n\n".format(model_data))
            fp,tp,fn,act_num,detect_num=0,0,0,0,0
            for i in range(len(acc_list)):
                f.write('test data : {}\n'.format(test_package[i]))
                prec, recall, tp_sum, fp_sum, fn_sum, d_sum, t_sum=acc_list[i].split(",")
                write_report(f, prec, recall, tp_sum, fp_sum, fn_sum, d_sum, t_sum)
                tp+=int(tp_sum)
                fp+=int(fp_sum)
                fn+=int(fn_sum)
                act_num+=int(t_sum)
                detect_num+=int(d_sum)
            precsion=tp/(fp+tp)
            recall=tp/(tp+fn)
            f.write('total :\n')
            write_report(f,precsion,recall,tp,fp,fn,detect_num,act_num)
        print("finish test model {}".format(model_data))

# imdb_name='voc_2007_trainval'
# iter='160020'
# demo_net='vgg16'
# predict_dir='/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data'
# package_data=['test_data-2018-07-24']
# test_model(imdb_name,iter,demo_net,predict_dir,package_data)