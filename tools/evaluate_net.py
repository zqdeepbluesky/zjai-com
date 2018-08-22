from lib.model.config import cfg,cal_data_aug_code
import cv2
import tensorflow as tf
from lib.extra_utils.xml_store import *
from lib.extra_utils import io_utils,cal_acc
from tools import predict

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
    date_prefix = cfg.TRAIN.SNAPSHOT_PREFIX[0:cfg.TRAIN.SNAPSHOT_PREFIX.find("_")] + '_' + date[0:date.find(":")].replace(" ", '-')
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
        predict_images(sess_test, net_test, jpg_files, xml_path, CLASSES)
        test_xml_path = os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package, 'Annotations_test')
        true_xml_path = os.path.join(cfg.ROOT_DIR, 'data', 'train_data', package, 'Annotations')
        test_info = cal_acc.cal_model_acc(test_xml_path, true_xml_path)
        test_info = "{},{},".format(model_data.split(".")[0], package) + test_info
        test_infos.append(test_info)
    for package in extra_test_package:
        packages.append(package)
        jpg_files, xml_path = load_test_images_from_txt(os.path.join(predict_dir, package), 'trainval_test')
        predict_images(sess_test, net_test, jpg_files, xml_path, CLASSES)
        test_xml_path = os.path.join(predict_dir, package, 'Annotations_test')
        true_xml_path = os.path.join(predict_dir, package, 'Annotations')
        test_info = cal_acc.cal_model_acc(test_xml_path, true_xml_path)
        test_info = "{},{},".format(model_data.split(".")[0], package) + test_info
        test_infos.append(test_info)
    tb = cal_acc.get_tabs(test_infos)
    tb = cal_acc.summary_tb(tb, test_infos)
    aug_code=cal_data_aug_code(cfg)
    txt_save_path = os.path.abspath(os.path.join(model_dir,'..','..','{}_batch_test_result'.format(aug_code)))
    if not os.path.exists(txt_save_path):
        io_utils.mkdir(txt_save_path)
    txt_save_path = os.path.join(txt_save_path, model_data.split(".")[0] + "_test_result")
    cal_acc.save_tb_in_txt(txt_save_path, tb)
    cal_acc.save_tb_in_xml(txt_save_path, tb)


def predict_images(sess,net,jpg_files,xml_path,CLASSES):
    for image in jpg_files:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(image))
        im = cv2.imread(image)
        result_data= predict.predict_image(sess, net, im, CLASSES)
        save_data_into_xml(image,im,xml_path,result_data)
# imdb_name='voc_2007_trainval'
# iter='160020'
# demo_net='vgg16'
# predict_dir='/home/hyl/data/ljk/github-pro/zjai-com/data/predict_data'
# package_data=['test_data-2018-07-24']
# test_model(imdb_name,iter,demo_net,predict_dir,package_data)