# tf-faster-rcnn is deprecated:
For a good and more up-to-date implementation for faster/mask RCNN with multi-gpu support, please see the example in TensorPack [here](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN).

# tf-faster-rcnn
A Tensorflow implementation of faster RCNN detection framework by Xinlei Chen (xinleic@cs.cmu.edu). This repository is based on the python Caffe implementation of faster RCNN available [here](https://github.com/rbgirshick/py-faster-rcnn).

# 开发环境配置
1. 硬件设备条件：  
  ```shell
  A.操作系统---ubuntu 16.04
  B.GPU显卡---支持cuda 8.0的显卡  查看网址：https://developer.nvidia.com/cuda-gpus
  ```
2. 软件系统条件：
  ```shell
  A.python3.5版本
  B.显卡驱动
  C.cuda8.0  &  对应cuda 8.0的cudnn v6
  D.tensorflow-gpu 1.4版本
  E.PyCharm 代码编写IDE
  F.g++
  G.git
  ```

3. 必备工具包：
  ```shell
  A.python3-dev
  B.python3-tk
  C.bazel
  D.Cython
  ```

4. 环境变量设置：
  ```shell
  A.将python路径加入环境变量， PYTHONPATH
  B.将cuda路径加入环境变量，CUDA_HOME
  '''

# 安装
1. 下载代码 
  ```shell
  git clone git@github.com:FirminSun/zjai-com.git
  ```

2. 修改部分代码
  打开./lib/setup.py更新-arch参数为当前GPU设备型号对应数值
  ```shell
  cd tf-faster-rcnn/lib
  vim setup.py
  设置-arch
  ```
  设备参数表如下：

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **注意**:目前操作都是针对GPU运行环境下，若只运用CPU进行训练，则需要设置USE_GPU_NMS=False

3. 构建Cython模块
  ```shell
  在./lib路径下，进行编译相关文件，得到so等编译文件
  make clean
  make
  cd ..
  ```

# 训练VOC数据集
1. 下载并设置VOC_2007数据
  ```shell
  A.下载VOC_2007数据集，
  """
  VOCtrainval  :http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  VOCtest      :http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  VOCdevkit    :http:/host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  """
  B.将tar文件解压到同一个文件夹
  """
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_08-Jun-2007.tar
  """
  C.将数据放到./data文件夹下且创建软连接
  cd ./data
  In -s $VOCdevkit VOCdevkit2007
  ```
2. 下载预训练模型
  ```shell
  预训练模型下载网址  res101：http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
		      vgg16:http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
  ./data路径下创建imagenet_weights文件夹放置解压后的预训练模型文件
  修改模型文件名：res101模型改为 res101.ckpt  vgg16模型改为 vgg16.ckpt
  ```

3. 相关工具包下载
  运行程序需要较多python工具包，如numpy、pandas、matplotlib、easydict等，遇到未安装的，可通过pycharm安装下载相应工具包

4. 修改部分代码
  打开./lib/datasets/tools/pascal_voc.py,设置第25行的DEBUG=False； 

5. 运行
  用pycharm打开./tools/train_net.py查看args参数是否正确，如使用的网络架构是否与下载的预训练模型名称一致；<br>
  检查完成后，使用python3.5 运行程序；<br>
  若出现类似如下报告，则代表成功运行：<br>
  ```shell
  iter: 20 / 50000, total loss: 0.359652
  >>> rpn_loss_cls: 0.000063
  >>> rpn_loss_box: 0.000821
  >>> loss_cls: 0.024140
  >>> loss_box: 0.007290
  >>> lr: 0.000100
  speed: 0.440s / iter
  ```


# 训练自己的数据集
1. 放置数据集
  将对应VOC数据集文件结构的数据文件放置到./data/train_data文件夹下，若无train_data文件夹，则自行创建

2. 创建训练集和测试集
  使用./zjai_createData/zjai_2_create_main.py文件，将数据集按9:1的比例分割成训练集和测试集，并生成trainval.txt和test.txt；<br>
  使用./zjai_createData/zjai_4_get_all_label.py文件，将统计所数据集中的所有label，并存放于./data/cfgs/com_classes.txt文件下，用于训练时读取classes；<br>

3. 修改模型配置文件：
  打开 ./experiments/cfgs/{net}.yml 文件修改相应参数，其中{net}代表你需要使用的模型，如vgg16或resnet<br>
  所需 数据增强方式 是在此文件进行设置的，如USE_HOR_FLIPPED、USE_VER_FLIPPED、BRIGHT_ADJUEST、ROTATE_ADJUEST、SHIFT_ADJUEST；<br>

4. 修改程序参数配置文件：
  打开 ./experiments/cfgs/args_setting.cfg 文件修改相应参数
  * load_args_json:是否加载指定args_json文件中的参数值；其中文件中的参数与args_setting.cfg参数类型一致；为正时，指定文件中的参数将覆盖当前加载的参数，为负时，继续加载args_setting.cfg的参数；<br>
  * args_json_dir:所需要加载指定args_json文件的路径，load_args_json为true才生效；<br>
  * cfg_file:是模型配置文件，若为vgg模型则为 vgg16.yml在根目录下的路径，如experiments/cfgs/vgg16.yml；<br>
  * weight:是模型权重文件，若为vgg模型则为vgg16.ckpt 在根目录下的路径，如data/imagenet_weights/vgg16.ckpt；<br>
  * imdb_name:是训练集数据的名称，代表某一批次数据名称；<br>
  * imdbval_name:是测试集数据的名称，<br>
  * epochs:是训练循环轮数的意义；<br>
  * max_iters:是一轮最大iter数的意义，最终程序训练到epochs*max_iters将会终止；<br>
  * tag:是存放模型文件的文件夹名，默认为空时，模型文件夹名为default;<br>
  * net:分类网络的名称；<br>
  * set_cfgs:重设config.py文件中的部分参数；<br>
  * package_name:是本次所需加载的数据包名，需要放置在./data/train_data文件夹下，支持多个数据包，如：['fusion_2018-08-06','fusion_2018-08-07']<br>
  * use_extra_test_data:含义为是否使用额外的数据包进行测试模型，程序每训练10W iter将会另外保存模型文件，并对其进行数据集测试，若参数为False，则使用默认的测试集进行测试，否则在原本的测试集基础上加载额外指定的数据包<br>
  * test_dir:是使用额外测试集在程序根目录下的路径，如:data/predict_data代表在xxxx/zjai-com/data/predict_data路径下；<br>
  * test_package:是指定test_dir路径下的测试数据包的文件名，同样支持多个测试数据包<br>
  **注意**:额外指定的数据包中，必须在数据包./ImageSets/Main/路径下存在test.txt，用于指定测试数据；<br>

5. 运行程序开始训练
  **注意**:目前代码只支持单GPU训练，若需要多GPU训练，请看https://github.com/endernewton/tf-faster-rcnn/issues/121<br>
  **注意**:./data/cache文件下放置的是加载数据过程中产生的roidb数据，若在训练前，roidb数据已存在并能够正常读取，程序将直接开始训练，减少重复数据准备工作；roidb文件的命名方式为：数据包+数据增强操作<br>

6. 使用tensorbroad可视化
  ```shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

7. 程序运行结果文件：
  * 测试报告:程序每迭代10万 iter 会对测试集进行测试，并检测结果返回到./output/{net}/{imdb_name}/{tag}/{aug_code}_batch_model/{net}_model_test.log,其中log是表格，表头意义如下：
model_name-->测试模型名称；
test_data-->测试数据名称；
precsion -->精确率；
recall   -->召回率；
detect_num -->模型识别到的物体总数；
actual_num -->测试数据集中实际存在物体总数；
tp_num -->正确识别的物体总数；
fp_num -->错误识别的物体总数；
fn_num -->漏识别的物体总数；
  * 中间模型文件：程序每迭代5000 iter 会保存一次模型文件，并返回到./output/{net}/{imdb_name}/{tag}/{aug_code}/路径下，一次共4个文件；
  * 每轮模型文件：程序每迭代10万 iter 会保存一次模型文件，并返回到./output/{net}/{imdb_name}/{tag}/{aug_code}_batch_model/路径下，一次共4个文件；
  * 执行参数文件：程序执行的参数，将被保存至./data/args_parse/{net}/{tag}/{package_name}_{aug_code}.json中；
  **注意**:aug_code为数据增强操作码，初始化为00000000，代表不做任何操作，根据[cfg.TRAIN.USE_HOR_FLIPPED,cfg.TRAIN.USE_VER_FLIPPED,cfg.TRAIN.BRIGHT_ADJUEST,cfg.TRAIN.ROTATE_ADJUEST,cfg.TRAIN.SHIFT_ADJUEST]来重构，每位代表每个数据增强操作状态，0代表不操作，1为操作，若进行水平翻转，代码为10000000。

# 预测测试集
1. 预测单张图像测试
  将测试数据放置到./data/predcit_data文件夹下，并将模型文件数据防止到./data/model文件夹下；<br>
  修改参数，使程序能够读取指定的测试图像和模型文件；<br>
  运行predict_test.py即可进行测试，并将其识别物体类别和位置信息，存放于相同路径的同名xml文件中；<br>

2. 批量预测测试集
  将测试数据文件夹放置到./data/predcit_data文件夹下，并将模型文件数据防止到./data/model文件夹下；<br>
  修改参数，使程序能够读取指定的测试图像和模型文件；<br>
  运行predict_batch.py即可进行批量测试，并将其识别物体类别和位置信息，存放于数据文件夹路径下的Annotations_test文件中；<br>

# 数据增强
1. 水平翻转
  * 参数意义：在config.py文件中参数USE_HOR_FLIPPED为设定是否执行水平翻转操作的参数；<br>
  * 参数设定：USE_HOR_FLIPPED=TRUE时数据集加倍，否则不做操作；<br>

2. 竖直翻转
  * 参数意义：在config.py文件中参数USE_VER_FLIPPED为设定是否执行竖直翻转操作的参数；<br>
  * 参数设定：USE_VER_FLIPPED=TRUE时数据集加倍，否则不做操作；<br>

3. 调整亮度
  * 参数意义：在config.py文件中参数BRIGHT_ADJUEST为设定是否执行调整亮度操作的参数；参数BRIGHT_ADJUEST_SCALE为设定调整亮度程度的参数；<br>
  * 参数设定：BRIGHT_ADJUEST=TRUE时数据集执行亮度调整操作；参数BRIGHT_ADJUEST_SCALE=[0.8,1.2]为对图像调整亮度是原本的0.8和1.2倍，为1或为空时不做操作；<br>

4. 旋转
  * 参数意义：在config.py文件中参数ROTATE_ADJUEST为设定是否执行旋转操作的参数；参数ROTATE_ADJUEST_ANGLE为设定旋转角度的参数；<br>
  * 参数设定：ROTATE_ADJUEST=TRUE时数据集执行旋转操作；参数ROTATE_ADJUEST_ANGLE=[90,180]为对图像旋转90度和180度，为0或360的倍数时不做操作；<br>

5. 水平移动
  * 参数意义：在config.py文件中参数SHIFT_ADJUEST为设定是否执行旋转操作的参数；参数SHIFT_ADJUEST_X、SHIFT_ADJUEST_Y为设定平移沿x、y平移距离的参数；<br>
  * 参数设定：SHIFT_ADJUEST=TRUE时数据集执行旋转操作；参数SHIFT_ADJUEST_X=100,SHIFT_ADJUEST_Y=100为图像向右平移100个像素点、向下平移100个像素点；<br>

## 关于config.py文件使用说明
config.py文件不允许修改赋值，如需添加函数，变量，需通过审核。<br>
如程序需要修改config中的值，可通过修改*.yml文件，覆盖config中的默认值。<br>
