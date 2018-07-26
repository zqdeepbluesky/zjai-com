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
  用pycharm打开./tools/train_net.py查看args参数是否正确，如使用的网络架构是否与下载的预训练模型名称一致；
  检查完成后，使用python3.5 运行程序；
  若出现类似如下报告，则代表成功运行：
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
  使用./zjai_createData/zjai_2_create_main.py文件，将数据集按9:1的比例分割成训练集和测试集，并生成trainval.txt和test.txt；
  使用./zjai_createData/zjai_4_get_all_label.py文件，将统计所数据集中的所有label，并存放于./data/cfgs/com_classes.txt文件下，用于训练时读取classes；

3. 修改部分代码
  设置./lib/datasets/pascal_voc.py的DEBUG=True；核对self._devkit_path和self._classes的路径是否正确；
  核对./tools/trainval_net.py的输入参数

4. 运行程序开始训练
  **注意**:目前代码只支持单GPU训练，若需要多GPU训练，请看https://github.com/endernewton/tf-faster-rcnn/issues/121
  **注意**:如果更替数据集或者进行不一样的数据增强操作后，需要删掉./data/cache文件夹下的pkl文件

5. 使用tensorbroad可视化
  ```shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

# 预测测试集
1. 预测单张图像测试
  将测试数据放置到./data/predcit_data文件夹下，并将模型文件数据防止到./data/model文件夹下；
  修改参数，使程序能够读取指定的测试图像和模型文件；
  运行predict_test.py即可进行测试，并将其识别物体类别和位置信息，存放于相同路径的同名xml文件中；

2. 批量预测测试集
  将测试数据文件夹放置到./data/predcit_data文件夹下，并将模型文件数据防止到./data/model文件夹下；
  修改参数，使程序能够读取指定的测试图像和模型文件；
  运行predict_batch.py即可进行批量测试，并将其识别物体类别和位置信息，存放于数据文件夹路径下的Annotations文件中；

# 数据增强
1. 水平翻转
  参数意义：在config.py文件中参数USE_HOR_FLIPPED为设定是否执行水平翻转操作的参数；
  参数设定：USE_HOR_FLIPPED=TRUE时数据集加倍，否则不做操作；

2. 竖直翻转
  参数意义：在config.py文件中参数USE_VER_FLIPPED为设定是否执行竖直翻转操作的参数；
  参数设定：USE_VER_FLIPPED=TRUE时数据集加倍，否则不做操作；

3. 调整亮度
  参数意义：在config.py文件中参数BRIGHT_ADJUEST为设定是否执行调整亮度操作的参数；参数BRIGHT_ADJUEST_SCALE为设定调整亮度程度的参数；
  参数设定：BRIGHT_ADJUEST=TRUE时数据集执行亮度调整操作；参数BRIGHT_ADJUEST_SCALE=[0.8,1.2]为对图像调整亮度是原本的0.8和1.2倍，为1或为空时不做操作；



