# ___2018 - 04 - 19 Tensorflow 实战 Google 深度学习框架___
***

# TensorFlow 环境搭建
## Protocol Buffer
  - Protocot Buffer 是谷歌开发的处理结构化数据的工具，是 TensorFlow 系统中使用到的重要工具，TensorFlow 中的数据基本都是通过 Protocol Buffer 来组织的
  - **序列化** 将结构化的数据变成数据流的格式，简单地说就是变为一个字符串。如何将结构化的数据序列化，并从序列化之后的数据流中还原出原来的结构化数据，统称为 **处理结构化数据**
  - 除 Protocol Buffer 之外， **XML** 和 **JSON** 是两种比较常用的结构化数据处理工具
  - Protocol Buffer 格式的数据和 XML 或者 JSON 格式的数据有比较大的区别
    - Protocol Buffer 序列化之后得到的数据不是可读的字符串，而是二进制流
    - XML 或 JSON 格式的数据信息都包含在了序列化之后的数据中，不需要任何其他信息就能还原序列化之后的数据
    - 使用 Protocol Buffer 时需要先定义数据的格式 **schema**，还原一个序列化之后的数据将需要使用到这个定义好的数据格式
  - Protocol Buffer 定义数据格式的文件一般保存在 **.proto文件** 中，每一个message代表了一类结构化的数据
  - **message** 里面定义了每一个属性的类型和名字，Protocol Buffer里属性的类型可以是像布尔型、整数型、实数型、字符型这样的基本类型，也可以是另外一个message
  - **message** 中的属性
    - **required 必须的** 所有这个message的实例都需要有这个属性
    - **optional 可选的** 这个属性的取值可以为空
    - **repeated 可重复的** 这个属性的取值可以是一个列表
## Bazel
  - Bazel 是从谷歌开源的自动化构建工具，相比传统的Makefile、Ant或者Maven，Bazel在速度、可伸缩性、灵活性以及对不同程序语言和平台的支持上都要更加出色
  - **安装**
    ```shell
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
    sudo apt-get update && sudo apt-get install bazel  |
    ```
  - **项目空间 workspace** 是Bazel的一个基本概念
    - 一个项目空间可以简单地理解为一个文件夹
    - 在这个文件夹中包含了编译一个软件所需要的源代码以及输出编译结果的 **软连接 symbolic link** 地址
    - 一个项目空间内可以只包含一个应用（比如TensorFlow），一个项目空间也可以包含多个应用
  - **WORKSPACE 文件** 定义了项目空间对外部资源的依赖关系，空文件表示没有外部依赖
  - **BUILD 文件** 定义需要编译的目标
    - BUILD文件采用一种类似于Python的语法来指定每一个编译目标的输入、输出以及编译方式
    - 与 Makefile 这种比较开放式的编译工具不同，Bazel的编译方式是事先定义好的
  - Bazel 对 Python 支持的编译方式
    - **py_binary** 将 Python 程序编译为可执行文件
    - **py_test** 编译 Python 测试程序
    - **py_library** 将 Python 程序编译成库函数供其他 py_binary 或 py_test 调用
  - **示例**
    - **WORKSPACE** 空文件
    - **hello_lib.py**
      ```python
      def print_hello_world():
          print("hello world")
      ```
    - **hello_main.py**
      ```python
      import hello_lib

      hello_lib.print_hello_world()
      ```
    - **BUILD** 定义两个编译目标
      ```python
      py_library (                  # 编译目标
          name = "hello_lib",       # 目标名称
          srcs = ["hello_lib.py",]  # 编译需要的源代码
      )

      py_binary (
              name = "hello_main",
              srcs = ["hello_main.py"],
              deps = [":hello_lib",] # 编译的依赖关系
      )
      ```
    - **编译**
      ```shell
      bazel build hello_main
      # or
      bazel build :hello_main
      ```
    - 编译完生成目标的软链接
      ```shell
      bazel-bin  bazel-genfiles  bazel-hello_bazel  bazel-out  bazel-testlogs  BUILD  hello_lib.py  hello_main.py  WORKSPACE
      ```
      - **bazel-bin** 目录下存放了编译产生的二进制文件以及运行该二进制文件所需要的所有依赖关系
      - 实际的编译结果文件保存到 **~/.cache/bazel 目录**，可以通过 **output_user_root** 或者 **output_base** 参数来改变
    - **执行** 在当前目录下运行
      ```shell
      bazel-bin/hello_main
      # [Output] hello world
      ```
## 通过 docker 安装 tensorflow
  - [tensorflow dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/)
  - Start CPU only container
    ```shell
    docker run -it -p 8888:8888 tensorflow/tensorflow
    ```
  - Start GPU (CUDA) container. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and run
    ```shell
    nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu
    ```
  - Docker Pull Command
    ```shell
    docker pull tensorflow/tensorflow
    ```
  - Go to your browser on http://localhost:8888/
  - [才云科技镜像 0.12.0](cargo.caicloud.io/tensorflow/tensorflow:0.12.0)
  - 启动命令，第一次会自动下载镜像
    ```shell
    sudo docker run --name=tensorflow -it -p 8888:8888 -p 6006:6006 cargo.caicloud.io/tensorflow/tensorflow:0.12.0
    # GPU version
    sudo nvidia-docker run --name=tensorflow-gpu -it -p 8888:8888 -p 6006:6006 cargo.caicloud.io/tensorflow/tensorflow:0.12.0-gpu
    ```
    - **-p 8888:8888** 将容器内运行的 **Jupyter 服务** 映射到本地机器 [Jupyter notebook](http://localhost:8888)
    - **-p 6006:6006** 将容器内运行的 **TensorFlow 可视化工具 TensorBoard** 映射到本地机器 [Tensor Board](http://localhost:6006/)
  - 修改 docker 中 Jupyter Notebook 密码
    ```shell
    # 登录 docker shell
    sudo docker ps
    # [Output] 6574b9090e4d        cargo.caicloud.io/tensorflow/tensorflow:0.12.0   tensorflow
    sudo docker exec -it 6574b9090e4d /bin/bash
    # or
    sudo docker exec -it `sudo docker ps | grep tensorflow | cut -d ' ' -f 1` /bin/bash

    # ipython in docker
    ipython
    In [7]: from notebook.auth import passwd
    In [9]: passwd()
    # or
    In [8]: passwd('[Your password]')
    # [Output] 'sha1:...'

    # docker shell
    vi ~/.jupyter/jupyter_notebook_config.py
    # Add
    c.NotebookApp.password = u'sha1:...'

    # 系统 shell 中重启容器
    sudo docker restart tensorflow
    ```
## pip 安装
  - [Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources)
  - [Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)
  - Install newest ternsorflow version in anaconda
    ```shell
    # pip
    pip install tensorflow
    pip install tensorflow-gpu
    pip install tensorflowonspark

    # conda
    conda install -c conda-forge tensorflow  # to install
    conda upgrade -c conda-forge tensorflow  # to upgrade
    ```
    ```python
    # Newest version including new APIs like
    # tf.feature_column <-- 1.2.0
    # tf.estimator.LinearRegressor  <-- 1.3.0
    tf.__version__
    Out[24]: '1.3.0'
    ```
  - Anaconda update
    ```shell
    conda update -n base conda
    conda update anaconda # update anaconda only
    conda update --all  # update all anaconda related packages
    conda clean --all # clean temporary fills
    ```
## 启用 GPU 支持
  - 安装开启 GPU 支持的 TensorFlow
  - 安装正确的 CUDA sdk 和 CUDNN 版本
  - 设置 LD_LIBRARY_PATH 和 CUDA_HOME 环境变量
    ```shell
    # 假定 CUDA 安装目录为 /usr/local/cuda
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
    export CUDA_HOME=/usr/local/cuda
    ```
## Hello World
  ```python
  import tensorflow as tf
  hello = tf.constant('Hello TensorFlow')
  sess = tf.Session()
  sess.run(hello)
  Out[5]: b'Hello TensorFlow'
  ```
  ```python
  a = tf.constant(10)
  b = tf.constant(32)
  sess.run(a+b)
  Out[9]: 42
  ```
***

# 基本概念
## 张量 Tensor
  - TensorFlow 数据模型，包含一个任意维度的数组，tensor 的 **秩 rank** 表示数组的维度
    ```python
    3 # a rank 0 tensor; this is a scalar with shape []
    [1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
    [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
    [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
    ```
  - 所有的数据都通过张量的形式来表示，从功能的角度上看，张量可以被简单理解为多维数组
    - 张量在 TensorFlow 中的实现并不是直接采用数组的形式，它只是对 TensorFlow 中运算结果的引用
    - 在张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程
  - 张量中主要保存了三个属性
    ```python
    a = tf.constant([1.0, 2.0], name='a')
    b = tf.constant([1.0, 2.0], name='b')
    result = tf.add(a, b, name='add')
    result
    Out[47]: <tf.Tensor 'add_6:0' shape=(2,) dtype=float32>
    ```
    - **名称 name** **node:src_output**，其中 node 为节点的名称，src_output 表示当前张量来自节点的第几个输出，0 表示第一个输出
    - **维度 shape**
    - **类型 type** 每一个张量会有一个唯一的类型，参与运算的所有张量类型不匹配时会报错
      ```python
      c = tf.constant([1, 2], name='c')
      result = c + b
      # [Out] ValueError: Tensor conversion requested dtype int32 for Tensor with dtype float32: 'Tensor("d:0", shape=(2,), dtype=float32)'
      # Define type as tf.fload32
      c = tf.constant([1, 2], name='c', dtype=tf.float32)
      ```
## 计算图 Graph
  - TensorFlow 计算模型，TensorFlow 的程序一般由两部分组成
    - 创建计算图 Building the computational graph
    - 运行计算图 Running the computational graph
  - **计算图 Computational Graph** 由一系列 TensorFlow 操作作为节点组成的图，每个节点使用零个或多个 tensor 作为 **输入**，并产生一个 tensor 作为 **输出**，如 **constant** 没有输入，产生一个内部存储的值
    ```python
    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0)  # also tf.float32 implicitly

    # 作为print参数输出时，并不会输出内部值，只有在运行时才会计算输出
    print(a, b)
    Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
    ```
  - 系统会自动维护一个默认的计算图，通过 **tf.get_default_graph** 函数可以获取当前默认的计算图
  - **a.graph** 可以查看张量所属的计算图，如果没有特意指定，等于当前默认的计算图
    ```python
    a = tf.constant([1.0, 2.0], name='a')
    b = tf.constant([1.0, 2.0], name='b')
    result = a + b
    a.graph is tf.get_default_graph()
    Out[5]: True
    ```
  - **tf.Graph** 函数来生成新的计算图,不同计算图上的张量和运算都不会共享
    ```python
    # g1 中将 v 初始化为 0
    g1 = tf.Graph()
    with g1.as_default():
        v = tf.get_variable('v', initializer=tf.zeros([1]))

    # g2 中将 v 初始化为 1
    g2 = tf.Graph()
    with g2.as_default():
        v = tf.get_variable('v', initializer=tf.ones([1]))

    # g1 中 v 值为 0
    with tf.Session(graph=g1) as sess:
        tf.initialize_all_variables().run()
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable('v')))
    # [Out] [0.]

    # g2 中 v 值为 1
    with tf.Session(graph=g2) as sess:
        tf.initialize_all_variables().run()
        with tf.variable_scope("", reuse=True):
            print(sess.run(tf.get_variable('v')))
    # [Out] [1.]
    ```
  - **tf.Graph.device** 函数来指定运行计算的设备
    ```python
    with g.device('/gpu:0'):
        result = a + b
        print(sess.run(result))
    # [Out] [2. 4.]
    ```
  - **collection 集合** 管理计算图不同类别的资源
    - 通过 **tf.add_to_collection** 函数可以将资源加入一个或多个集合中
    - 通过 **tf.get_collection** 获取一个集合里面的所有资源
  - TensorFlow 中维护的集合列表
    - **tf.GraphKeys.GLOBAL_VARIABLES** 所有变量
    - **tf.GraphKeys.TRAINABLE_VARIABLES** 可学习的变量（一般指神经网络中的参数）
    - **tf.GraphKeys.SUMMARIES** 日志生成相关的张量，用于 TensorFlow 计算可视化
    - **tf.GraphKeys.QUEUE_RUNNERS** 处理输入的QueueRunner
    - **tf.GraphKeys.MOVING_AVERAGE_VARIABLES** 所有计算了滑动平均值的变量
## 会话 Session
  - TensorFlow 运行模型，可使用 **run** 方法运行创建好的图
  - 当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄漏的问题
    ```python
    ''' sess.close '''
    sess = tf.Session()
    sess.run(result)
    sess.close()

    ''' with '''
    with tf.Session() as sess：
        sess.run(result)
    ```
  - **tf.Tensor.eval** 如果没有特殊指定，运算会自动加入默认的计算图中，但默认的会话需要手动指定，当默认的会话被指定之后可以通过 **tf.Tensor.eval** 来计算一个张量的取值
    ```python
    with tf.Session() as sess:
        with sess.as_default():
            print(result.eval())
    # [Out] [2. 4.]
    ```
  - **tf.InteractiveSession** 在交互式环境下直接构建默认会话的函数，自动将生成的会话注册为默认会话
    ```python
    # 可以省去将产生的会话注册为默认会话的过程
    sess = tf.InteractiveSession()
    result.eval()
    Out[60]: array([2., 4.], dtype=float32)

    sess.close()
    ```
  - **ConfigProto** 配置需要生成的会话，可以配置类似 **并行的线程数** / **GPU 分配策略** / **运算超时时间** 等参数
    - **allow_soft_placement** 布尔型参数，默认值为 False，为 True 的时候，在以下任意一个条件成立的时候，GPU上的运算可以放到CPU上进行，当某些运算无法被当前GPU支持时，可以自动调整到CPU上，而不是报错
      - 运算无法在GPU上执行
      - 没有GPU资源（比如运算被指定在第二个GPU上运行，但是机器只有一个GPU）
      - 运算输入包含对CPU计算结果的引用
    - **log_device_placement** 布尔型参数，为 True 时日志中将会记录每个节点被安排在了哪个设备上以方便调试，在生产环境中将这个参数设置为 False 可以减少日志量
    ```python
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=config)
    ```
## 占位符 placeholders
  - **占位符 placeholders** 可以在运行时指定的参数，用于提供输入数据
  - 在定义 placeholder 时，这个位置上的数据类型是需要指定的，数据的维度信息可以根据提供的数据推导得出，所以不一定要给出
    ```python
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b

    # Dict values to initialize a / b
    sess = tf.Session()
    sess.run(adder_node, {a: [1, 2], b: [2, 3]})
    Out[130]: array([3., 5.], dtype=float32)
    ```
## 变量 Variables
  - **变量 tf.Variable** 向图中添加可训练的参数，包含初值与类型
  - 变量的初值可以是 **常量 tf.constant** / **随机数 tf.random_normal** / **其他变量**
  - 调用 **tf.Variable** 不会初始化变量，变量的值在被使用之前，需要调用变量初始化函数 **ww.initializer** / **ww.initialized_value** / **tf.global_variables_initializer**
    ```python
    # 定义交互式会话
    sess = tf.InteractiveSession()
    # 定义变量
    ww = tf.Variable(tf.constant([2, 3], dtype=tf.float32))
    # 运行会话
    sess.run(ww.initialized_value())
    Out[31]: array([2., 3.], dtype=float32)
    ```
  - **TensorFlow 常量声明方法**
    - **tf.zeros** 产生全0的数组
    - **tf.ones** 产生全1的数组
    - **tf.fill** 产生一个全部为给定数字的数组
    - **tf.constant** 产生一个给定值的常量
    ```python
    ff = tf.fill([2, 3], 9)
    sess.run(ff)
    Out[10]:
    array([[9, 9, 9],
           [9, 9, 9]], dtype=int32)
    ```
  - **TensorFlow 随机数生成函数**
    - **tf.random_normal** 正态分布
    - **tf.truncated_normal** 正态分布，但如果随机出来的值偏离平均值超过2个标准差，那么这个数将会被重新随机
    - **tf.random_uniform** 均匀分布
    - **tf.random_gamma** Gamma 分布
    ```python
    # 均值为 2，标准差为 2 正态随机分布的 2 * 3 数组
    rr = tf.random_normal([2, 3], mean=2, stddev=2)
    sess.run(rr)
    Out[8]:
    array([[ 0.48397958,  5.52174   ,  0.3912573 ],
           [ 4.4237976 , -0.6381433 ,  2.838407  ]], dtype=float32)
    ```
  - **ww.initializer 方法初始化变量** 需要先调用 sess.run(ww.initializer)，再调用 sess.run(ww)
    ```python
    ww = tf.Variable([2.0, 3.0], dtype=tf.float32)
    bb = tf.Variable([1.0, 2.0], dtype=tf.float32)
    yy = ww + bb

    sess.run(ww.initializer)
    sess.run(bb.initializer)

    sess.run(ww)
    Out[69]: array([2., 3.], dtype=float32)
    sess.run(bb)
    Out[70]: array([1., 2.], dtype=float32)
    sess.run(yy)
    Out[71]: array([3., 5.], dtype=float32)
    ```
  - **ww.initialized_value() 方法初始化变量** 可以直接调用 sess.run(ww.initialized_value())
    ```python
    ww = tf.Variable([2.0, 3.0], dtype=tf.float32)
    bb = tf.Variable(ww.initialized_value() * 2)
    sess.run(bb.initialized_value())
    Out[84]: array([4., 6.], dtype=float32)

    sess.run(ww.initialized_value())
    Out[85]: array([2., 3.], dtype=float32)
    ```
  - **tf.global_variables_initializer() 方法初始化所有变量** 创建 **子图 sub-graph**，并调用 **sess.run** 运行子图完成变量初始化
    ```python
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)  # b = -0.3
    x = tf.placeholder(tf.float32)

    # linear model
    linear_model = W * x + b
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(b)
    Out[111]: array([-0.3], dtype=float32)
    sess.run(linear_model, {x:[1, 2, 3, 4]})
    Out[112]: array([0.        , 0.3       , 0.6       , 0.90000004], dtype=float32)
    ```
  - **tf.assign() 调整变量值** 变量值可以通过 **tf.assign** 等函数修改
    ```python
    # W = -1, b = 1 is the ideal value for this model
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    Out[124]: [array([-1.], dtype=float32), array([ 1.], dtype=float32)]

    sess.run(linear_model, {x:[1, 2, 3, 4]})
    Out[126]: array([ 0., -1., -2., -3.], dtype=float32)
    ```
  - **tf.global_variables() 方法获取当前计算图上所有变量** 有助于持久化整个计算图的运行状态
  - **tf.trainable_variables() 方法获取当前计算图上所有需要优化的参数** 声明变量时参数 trainable=True(默认为 True) 时加入 GraphKeys.TRAINABLE_VARIABLES
## Tensorflow playground
  - [Tensorflow playground](http://playground.tensorflow.org)
  - 一个小格子代表神经网络中的一个节点，而边代表节点之间的连接
  - 每一条边代表了神经网络中的一个参数，边上的颜色体现了这个参数的取值，颜色越深时表示这个参数取值的绝对值越大，当边的颜色接近白色时，这个参数的取值接近于0
  ![image](images/tf_playground_1.png)
## 通用函数
  - **tf.clip_by_value** 函数将一个张量中的数值限制在一个范围之内
    ```python
    sess = tf.InteractiveSession()
    tt = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    tf.clip_by_value(tt, 2.5, 4.5).eval()
    Out[9]:
    array([[2.5, 2.5, 3. ],
           [4. , 4.5, 4.5]], dtype=float32)
    ```
  - **tf.greater** 返回两个张量的比较结果
    ```python
    sess = tf.InteractiveSession()
    v1 = tf.constant([1, 2, 3, 4])
    v2 = tf.constant([4, 3, 2, 1])
    tf.greater(v1, v2).eval()
    Out[37]: array([False, False,  True,  True])
    ```
  - **tf.where** 类似 np.where 的功能，在条件满足时取第一个值，否则取第二个值
    ```python
    sess = tf.InteractiveSession()
    v1 = tf.constant([1, 2, 3, 4])
    v2 = tf.constant([4, 3, 2, 1])
    tf.where(tf.greater(v1, v2), v1, v2).eval()
    Out[42]: array([4, 3, 3, 4], dtype=int32)
    ```
***

# 三层线性神经网络
## 神经网络分层结构
  - 目前主流的神经网络都是 **分层结构**
    - 第一层 **输入层** 代表特征向量中每一个特征的取值，同一层的节点不会相互连接，而且每一层只和下一层连接
    - 中间层 **隐藏层** 一般一个神经网络的隐藏层越多，这个神经网络越“深”，深度学习中的深度和神经网络的层数也是密切相关的
    - 最后一层 **输出层** 得到计算的结果，通过输出值和一个事先设定的阈值，就可以得到最后的分类结果，一般可以认为当输出值离阈值越远时得到的答案越可靠
  - 一个最简单的神经元结构的输出就是 **所有输入的加权和**，而不同输入的权重就是神经元的参数，神经网络的优化过程就是优化神经元中参数取值的过程
  - **全连接神经网络** 是因为相邻两层之间任意两个节点之间都有连接，区别于 **卷积层** / **LSTM 结构**
  - **训练神经网络的过程** 一般分为三个步骤
    - 定义 **神经网络的结构** 和 **前向传播的输出结果**
    - 定义 **损失函数** 以及选择 **反向传播优化的算法**
    - 生成 **会话 session** 并且在训练数据上反复运行反向传播优化算法

  ![image](images/forward_network.png)
## 前向传播算法 Forward propagation 输出结果
  ```python
  ''' Forward propagation
  输入层 有两个输入
  隐藏层 有三个节点
  输出层 有一个输出
  '''
  train_X = [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8], [0.2, 0.3]]
  # Notice train_y is shape of [4, 1]
  train_y = [[0 if x1+x2 < 0.6 else 1] for x1, x2 in train_X]  # [[1], [0], [1], [0]]

  X = tf.placeholder(tf.float32, name='input')
  w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
  w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))
  b = tf.Variable(tf.constant([.0, .0, .0]))

  a = tf.matmul(X, w1) + b
  linear_model = tf.matmul(a, w2)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)
  sess.run(linear_model, feed_dict={X: train_X})
  Out[6]:
  array([[-1.8375354 ],
         [-0.14922595],
         [-1.2469425 ],
         [-0.5071239 ]], dtype=float32)

  # Close session: sess.close()
  # Run again: sess = tf.Session(); sess.run(init); sess.run(linear_model, feed_dict={X:[[0.7, 0.9]]})
  ```
## 损失函数 loss function 评估当前模型的预测结果与目标值的距离
  ```python
  ''' standard loss model '''
  # Target
  y = tf.placeholder(tf.float32)
  # predict values' distance to target values
  squared_deltas = tf.square(linear_model - y)
  # Sum up total Loss
  loss = tf.reduce_sum(squared_deltas)
  sess.run(loss, {X: train_X, y: train_y})
  Out[11]: 13.379801
  ```
## 反向传播算法 Back propagation 更新神经网络参数的取值
  - 在每次迭代的开始，首先需要选取一小部分训练数据，这一小部分数据叫做一个 **batch**
  - 然后，这个batch的样例会通过 **前向传播算法** 得到神经网络模型的预测结果
  - 因为训练数据都是有正确答案标注的，所以可以计算出当前神经网络模型的预测答案与正确答案之间的差距
  - 最后，基于这预测值和真实值之间的差距，反向传播算法会相应更新神经网络参数的取值，使得在这个batch上神经网络模型的预测结果和真实答案更加接近
  - **优化器 optimizers** 用于缓慢改变变量的值，使得 **损失函数 loss function** 最小
  - **梯度下降算法 gradient descent** 沿损失函数梯度下降最大的方向调整参数，**tf.gradients 方法** 用于计算模型的导数，**tf.train.GradientDescentOptimizer 方法** 用于自动完成参数选取
  ```python
  ''' Back propagation '''
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(loss)
  for i in range(1000):
      sess.run(train, {X: train_X, y: train_y})
  # Trained w1, w2, b
  sess.run([w1, w2, b])
  Out[5]:
  [array([[-0.15459022, -0.58988345,  0.45714408],
          [ 0.2202878 , -0.4982153 ,  1.3863227 ]], dtype=float32),
   array([[ 1.6989982 ],
          [-0.3714918 ],
          [ 0.78323066]], dtype=float32),
   array([-0.28499216, -0.04386206,  0.11725496], dtype=float32)]

  # Loss after trained
  sess.run(loss, {X: train_X, y: train_y})
  Out[16]: 0.03600077

  # Predict value on train data
  sess.run(linear_model, feed_dict={X: train_X})
  Out[17]:
  array([[ 1.0957603 ],
        [ 0.0844568 ],
        [ 0.86833656],
        [-0.04860568]], dtype=float32)

  # Re-init: sess.run(init); sess.run([w1, w2, b])
  ```
***

# 深度学习与深层神经网络
## 激活函数 activation function 实现非线性化
  - **深度学习** 一类通过 **多层** **非线性** 变换对高复杂性数据建模算法的合集，**深层神经网络** 是实现多层非线性变换最常用的一种方法
  - **激活函数** 将每一个神经网络中节点的输出通过一个非线性函数，将整个神经网络模型转化为非线性的
  - **感知机** 先将输入进行加权和，然后再通过激活函数最后得到输出，这个结构就是一个没有隐藏层的神经网络，可以简单地理解为单层的神经网络

    ![image](images/activation_function_on_node.png)
  - **在线性模型中无法处理非线性数据**

    ![image](images/linear_on_nolinear.png)

    通过加入 **激活函数 Tanh** 将线性模型转化为非线性模型

    ![image](images/linear_with_tanh.png)
  - **常用的激活函数** 目前 TensorFlow 提供了 7 种不同的非线性激活函数，**tf.nn.relu** / **tf.sigmoid** / **tf.tanh** 是其中比较常用的几个

    ![image](images/activation_function.png)
  - **前向传播算法中使用激活函数**
    ```python
    a = tf.nn.relu(tf.matmul(X, w1) + b)
    relu_model = tf.nn.relu(tf.matmul(a, w2))
    ```
## 多层网络解决异或运算 exclusive or
  - **单层感知机模型** 无法解决异或数据集的分类问题，如果两个输入的符号相同则输出为0，否则输出为1

    ![image](images/exclusive_or_1.png)

    当加入隐藏层之后，异或问题就可以得到很好地解决

    ![image](images/exclusive_or_2.png)
  - 隐藏层的四个节点中，每个节点都有一个角是黑色的。这四个隐藏节点可以被认为代表了从输入特征中抽取的更高维的特征。比如第一个节点可以大致代表两个输入的逻辑与操作的结果（当两个输入都为正数时该节点输出为正数）。从这个例子中可以看到，深层神经网络实际上有组合特征提取的功能。这个特性对于解决不易提取特征向量的问题（比如图片识别、语音识别等）有很大帮助
## 损失函数 Loss function softmax and cross-entropy
  - 通过神经网络解决 **多分类问题** 最常用的方法是设置 n 个输出节点，其中n为类别的个，即 **n 维的哑变量矩阵 dummy variables**
  - **Softmax 回归** 将神经网络的输出变成一个概率分布，通过交叉熵 **Cross-entropy** 来计算预测的概率分布和目标的概率分布之间的距离

    ![iamge](images/tensorflow_softmax.png)

  - **Softmax Regression**，在分类问题中给出所属类别的概率，首先收集属于某一类别的证据 evidence，随后将其转化为概率
    ```shell
    # i is the index of each class, j is the index of each input data
    evidence(i) = Σ(j) (W(i, j) * x(j)) + b(i)
    # convert evidence into probabilities of our input being in each class
    y = softmax(evidence)
    # softmax is exponentiating its inputs and then normalizing them
    softmax(x) = normalize(exp(x)) = exp(x) / Σ(j) exp(x)
    # Consequently, in a more compactly way
    y = softmax(W x + b)
    ```
  - **python 实现 Softmax 回归**
    ```python
    # Convert a two-dimension array to softmax prob on their own line
    def softmax_prob(prob_array):
      e_prob_array = np.exp(np.array(prob_array))
      if np.all(prob_array == -np.inf):
          return e_prob_array
      else:
          return np.divide(e_prob_array, np.sum(e_prob_array, 1).reshape(-1, 1))

    # Test run
    tt = np.arange(5)
    softmax_prob(np.vstack([tt, tt]))
    Out[27]:
    array([[0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865],
           [0.01165623, 0.03168492, 0.08612854, 0.23412166, 0.63640865]])
    ```
  - **交叉熵 Cross-entropy** 评估模型的预测结果与目标值的差距，刻画了两个概率分布之间的距离，是分类问题中使用比较广的一种损失函数，目标使该值尽量小
    ```python
    # 其中y'表示预测值，y表示目标值
    H(y_, y) = -Σ(i)(y_(i) * log (y(i)))
    ```
    实现
    ```python
    # reduction_indices=[1] 指定在列轴上求和，随后 tf.reduce_mean 计算所有样本的平均值
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    ```
    使用更稳定的内建版本 **softmax_cross_entropy_with_logits_v2**
    ```python
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    ```
  - **tf.nn.sparse_softmax_cross_entropy_with_logits** 在只有一个正确答案的分类问题中，来进一步加速计算过程
  - 解决 **回归问题** 的神经网络一般只有一个输出节点，这个节点的输出值就是预测值，最常用的损失函数是 **均方误差 MSE，mean squared error**
    ```python
    MSE(y, y_) = Σ(i)(y - y_) ** 2 / n
    其中y代表了神经网络的输出答案，y_代表了标准答案
    ```
    **python 实现**
    ```python
    mse = tf.reduce_mean(tf.square(y_ - y))
    ```
## 梯度下降算法 Gradient decent
  - **梯度下降算法** 主要用于优化单个参数的取值，而 **反向传播算法** 给出了一个高效的方式在所有参数上使用梯度下降算法，从而使神经网络模型在训练数据上的损失函数尽可能小
  - **梯度下降算法** 会迭代式更新参数，不断沿着梯度的反方向让参数朝着总损失更小的方向更新，对于参数 w 与损失函数 Loss(w)，其梯度
    ```shell
    grad(w) = δ Loss(w) / δ w
    ```
  - **学习率 learning rate** 定义每次参数更新的幅度，根据参数的梯度和学习率，每次参数更新的公式为
    ```shell
    w(n+1) = w(n) - r * grad(w(n))
    ```
  - **梯度下降算法** 在偏导为 0 时可能只能得到 **局部最优解**，并不能保证达到 **全局最优解**，只有当损失函数为凸函数时，梯度下降算法才能保证达到全局最优解

    ![image](images/gradient_decent_limit.png)
  - **随机梯度下降的算法stochastic gradient descent**
    - **梯度下降算法** 在每次更新回归系数时都需要遍历整个数据集，在数据集变大时，该方法的计算复杂度就会很高
    - **随机梯度下降的算法** 在每一轮迭代中，随机优化某一条训练数据上的损失函数
    - 使用随机梯度下降优化得到的神经网络甚至可能无法达到局部最优
  - **batch** 为了综合梯度下降算法和随机梯度下降算法的优缺点，可以每次计算一小部分训练数据的损失函数，这一小部分数据被称之为一个 batch
## 自定义损失函数预测商品销量计算盈利
  - 自定义损失函数，在预测商品数量 多 / 少 时利润的损失是不同的
    ```python
    Loss(y, y_) = Σ(i)f(y, y_)

    f(x, y) = {'x > y': a(x - y), 'x <= y': b(x - y)}
    其中 a = 10, b = 1
    ```
    **python 实现**
    ```python
    loss = tf.reduce_mean(tf.where(tf.grater(y, y_), (y - y_) * a, (y - y_) * b))
    ```
    由于损失函数中定义的预测少比预测多损失更多，因此训练出的 w1 略大于 1
  ```python
  from numpy.random import RandomState
  import tensorflow as tf
  import matplotlib.pyplot as plt

  # Two input, one output
  X = tf.placeholder(tf.float32, [None, 2], name='Input-X')
  y_ = tf.placeholder(tf.float32, [None, 1], name='Input-y')

  w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
  y = tf.matmul(X, w1)

  loss_less = 10
  loss_more = 1
  # Choose different loss weight if predicted more / less
  loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y-y_) * loss_more, (y_-y) * loss_less))
  # Define train steps
  optimizer = tf.train.AdamOptimizer(0.001)
  train_step = optimizer.minimize(loss)

  # Generate train / test data
  rdm = RandomState(1)
  data_size = 128
  train_X = rdm.rand(data_size, 2)
  # Target with a random -0.05 - 0.05 noise
  train_y = np.array([ [mm + nn + rdm.rand()/10.0 - 0.05] for mm, nn in train_X ])

  # Train model
  with tf.Session() as sess:
      STEPS = 5000
      batch_size = 8
      w1_collection = []

      init = tf.global_variables_initializer()
      sess.run(init)
      for ii in range(STEPS):
          batch = np.random.permutation(data_size)[:batch_size]
          sess.run(train_step, feed_dict={X: train_X[batch], y_:train_y[batch]})
          w1_collection.append(sess.run(w1))

  w1_collection = np.array(w1_collection)
  print('w1 = %s' % w1_collection[-1])  # w1 = [[1.0193471], [1.0428091]]

  # plot w1
  fig, axes = plt.subplots(2, 1)
  for ind in range(w1_collection.shape[1]):
      axes[ind].plot(w1_collection[:, ind, 0])
      axes[ind].set_title('w1[%d]' % ind)
  ```
  ![image](images/user_defined_loss_weight.png)
## 指数衰减学习率 exponential decay learning rate
  - 学习率决定参数每次更新的幅度
    - 如果幅度过大，那么可能导致参数在极优值的两侧来回移动
    - 如果幅度过小，虽然能保证收敛性，但会大大降低优化速度
  - **tf.train.exponential_decay 指数衰减学习率** 可以先使用较大的学习率来快速得到一个比较优的解，然后随着迭代的继续逐步减小学习率，使得模型在训练后期更加稳定
    ```shell
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decaysteps)
    ```
    函数定义
    ```python
    exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
    ```
  - **staircase** 的值被设置为 True 时
    - global_step / decay_steps 会被转化成整数 使得学习率成为一个阶梯函数 staircase function
    - 在这样的设置下，**decay_steps** 一般等于 **总样本数 / batch size**， 通常代表了完整的使用一遍训练数据所需要的迭代轮数
    - 这种设置的常用场景是每完整地过完一遍训练数据，学习率就减小一次，这可以使得训练数据集中的所有数据对模型训练有相等的作用
  - **示例**
    ```python
    TRAIN_STEPS = 100
    global_step = tf.Variable(0)
    LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

    x = tf.Variable(tf.constant(5, dtype=tf.float32), name='x')
    y = tf.square(x)
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

    LEARNING_RATE_list = []
    x_list = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for ii in range(TRAIN_STEPS):
            sess.run(train_op)
            LEARNING_RATE_list.append(sess.run(LEARNING_RATE))
            x_list.append(sess.run(x))

    # plot
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x_list)
    axes[0].set_title('x')
    axes[1].plot(LEARNING_RATE_list, drawstyle='steps-post')
    axes[1].set_title('learning rate')
    ```
    ![image](images/exponential_decay_learning_rate.png)
## 过拟合 与 正则化缩减
  - 为了避免过拟合问题，一个常用的方法是 **正则化 regularization**
  - **正则化** 就是在损失函数 **Loss(w)** 中加入刻画模型复杂程度的指标 **R(w)**，在参数优化时优化正则化的损失函数
    ```shell
    # Loss(w) 损失函数
    # R(w) 模型复杂程度
    # λ 模型复杂损失在总损失中的比例
    Loss(w) + λ * R(w)
    ```
  - **模型复杂度 R(w)** 一般来说只由权重 w 决定，tensorflow 提供两种正则化方法 **L1 正则化 / L2 正则化**，基本的思想都是希望通过限制权重的大小，使得模型不能任意拟合训练数据中的随机噪音
  - **tf.contrib.layers.l1_regularizer L1 正则化** (Lasso 缩减) 会将某些参数缩减到 0，使参数变得更稀疏，可以达到类似特征选取的功能
    ```shell
    R(w) = Σ|w(i)|
    ```
  - **tf.contrib.layers.l2_regularizer L2 正则化** (Ridge regression) 参数不会缩减到 0，公式可导，对含有 L2 正则化损失函数的优化要更加简洁
    ```shell
    R(w) = Σ|w(i) ^ 2|
    ```
  - **tf.contrib.layers.l1_l2_regularizer 同时使用 L1 和 L2 正则化**
    ```shell
    R(w) = Σ(a * w(i) ^ 2 + (1 - a) * |w(i)|)
    ```
  - **示例**
    ```python
    weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

    # L1: out = np.sum(np.abs(sess.run(weights))) * 0.5
    sess.run(tf.contrib.layers.l1_regularizer(.5)(weights))
    Out[24]: 5.0

    # L2: out = np.sum(sess.run(weights) ** 2) / 2 * 0.5
    sess.run(tf.contrib.layers.l2_regularizer(.5)(weights))
    Out[34]: 7.5
    ```
## 计算一个 5 层神经网络带 L2 正则化的损失函数的计算方法
  ```python
  import tensorflow as tf
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd

  ''' Generate random test data, labels and then noised data '''
  TRAINING_STEPS = 4000

  np.random.seed(0)
  x1 = np.random.uniform(-1, 1, size=150)
  x2 = np.random.uniform(0, 2, size=150)
  label = np.array([ 0 if i ** 2 + j ** 2 <= 1 else 1 for i, j in zip(x1, x2) ])

  x1 = np.random.normal(x1, 0.1)
  x2 = np.random.normal(x2, 0.1)
  df = pd.DataFrame({'x1':x1, 'x2': x2, 'y': label})

  # plot original data
  fig, axes = plt.subplots(3, 1)
  grouped = df.groupby(df.iloc[:, -1])
  scatterFunc = lambda x : plt.scatter(x.iloc[:, 0], x.iloc[:, 1], label = x.iloc[0, 2])
  for ind in range(1, 4):
      plt.subplot(3, 1, ind)
      grouped.agg(scatterFunc)

  ''' Define a 5-layer network model '''
  def get_weight(shape, lambda1):
      ww = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
      tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(ww))
      return ww

  graph1 = tf.Graph()
  with graph1.as_default():
      x = tf.placeholder(tf.float32, shape=(None, 2))
      y_ = tf.placeholder(tf.float32, shape=(None, 1))

      # Nodes number for every layer
      layer_dimensions = [2, 10, 5, 3, 1]
      in_dimension = layer_dimensions[0]
      cur_layer = x
      for ll in layer_dimensions[1:]:
          out_dimension = ll
          weight = get_weight([in_dimension, out_dimension], lambda1 = 0.003)
          bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
          cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
          in_dimension = out_dimension

      y = cur_layer

      # Loss function, normal one and l2 regularized one
      mse_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / df.shape[0]
      tf.add_to_collection('losses', mse_loss)
      loss = tf.add_n(tf.get_collection('losses'))

  ''' Train model using mse_loss that without regularization '''
  with tf.Session(graph=graph1) as sess:
      train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss)
      init = tf.global_variables_initializer()
      sess.run(init)
      for i in range(TRAINING_STEPS):
          sess.run(train_op, feed_dict={x: df.values[:, :2], y_: df.values[:, -1:]})
          if i % 2000 == 0:
              print("After %d steps, mse_loss: %f" % (i,sess.run(mse_loss, feed_dict={x: df.values[:, :2], y_: df.values[:, -1:]})))

      xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
      grid = np.c_[xx.ravel(), yy.ravel()]
      probs = sess.run(y, feed_dict={x:grid})
      probs = probs.reshape(xx.shape)

  # Plot
  plt.subplot(3, 1, 2)
  plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)

  ''' Train model using loss that regularized '''
  with tf.Session(graph=graph1) as sess:
      train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
      init = tf.global_variables_initializer()
      sess.run(init)
      for i in range(TRAINING_STEPS):
          sess.run(train_op, feed_dict={x: df.values[:, :2], y_: df.values[:, -1:]})
          if i % 2000 == 0:
              print("After %d steps, loss: %f" % (i,sess.run(mse_loss, feed_dict={x: df.values[:, :2], y_: df.values[:, -1:]})))

      xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
      grid = np.c_[xx.ravel(), yy.ravel()]
      probs = sess.run(y, feed_dict={x:grid})
      probs = probs.reshape(xx.shape)

  # Plot
  plt.subplot(3, 1, 3)
  plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)

  fig.show()
  ```
  ![image](images/mse_regulation_loss.png)
  - **Q: 在 ipython 中再次运行时报错 Placeholder[dtype=DT_FLOAT, shape=[?,1]**
    ```python
    在第二次运行时
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        sess.run(train_op, feed_dict={x: df.values[:, :2], y_: df.values[:, -1:]})
    报错
        InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'Placeholder_1' with dtype float and shape [?,1]
	       [[Node: Placeholder_1 = Placeholder[dtype=DT_FLOAT, shape=[?,1], _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
    ```
    A: 原因可能是因为第一次添加到 collection 中的 losses 在第二次运行时也获取到了，没有初始化，每次在运行时创建新的 Graph
    ```python
    # Create new graph
    graph1 = tf.Graph()
    with graph1.as_default():
        [ Define graph contains ]
    # Use new graph as default one when creating session
    with tf.Session(graph=graph1):
        [Session contains]
    ```
## 滑动平均模型 exponantial moving average
  - **滑动平均模型** 可以使模型在测试数据上更健壮 robust 在采用随机梯度下降算法训练神经网络时，使用滑动平均模型在很多应用中都可以在一定程度提高最终模型在测试数据上的表现
  - **tf.train.ExponentialMovingAverage 滑动平均模型**
    - **参数 decay 衰减率** 控制模型更新的速度，ExponentialMovingAverage 对每一个变量会维护一个 **影子变量 shadow variable**，初始值是相应变量的初始值，每次变量更新时影子变量的值会相应更新
      ```shell
      shadow_variable = decay * shadow_variable + (1 - decay) * variable
      ```
      decay 越大模型越趋于稳定，实际应用中，decay 一般会设成非常接近1的数
    - **参数 num_updates** 动态设置 decay 的大小，使得模型在训练前期可以更新得更快，每次使用的衰减率更新为
      ```shell
      # 0.1 - 1
      min (decay, (1 + num_updates) / (10 + num_updates))
      ```
  - **示例**
    ```python
    # v1 = 0, num_updates = 0, decay = 0.99
    v1 = tf.Variable(0, dtype=tf.float32)
    step = tf.Variable(0, trainable=False)
    ema = tf.train.ExponentialMovingAverage(0.99, step)
    maintain_averages_op = ema.apply([v1])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # v1, ema.average(v1) = [0., 0.]
        print(sess.run([v1, ema.average(v1)]))

        # v1 = 5
        sess.run(tf.assign(v1, 5))
        sess.run(maintain_averages_op)
        # decay = min(0.99, (1 + 0) / (10 + 0)) = 0.1
        # ema = 0.1 * 0 + (1 - 0.1) * 5 = 4.5
        print(sess.run([v1, ema.average(v1)]))

        # step = 10000, v1 = 10
        sess.run(tf.assign(step, 10000))
        sess.run(tf.assign(v1, 10))
        sess.run(maintain_averages_op)
        # decay = min(0.99, (1 + 10000) / (10 + 10000)) = 0.99
        # ema = 0.99 * 4.5 + (1 - 0.99) * 10 = 4.555
        print(sess.run([v1, ema.average(v1)]))

        # ema = 0.99 * 4.555 + (1 - 0.99) * 10 = 4.60945
        sess.run(maintain_averages_op)
        print(sess.run([v1, ema.average(v1)]))
    ```
    Output
    ```python
    [0.0, 0.0]
    [5.0, 4.5]
    [10.0, 4.555]
    [10.0, 4.60945]
    ```
***
```python
# 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了TensorFlow中提
# 供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类
# 问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST问题的图片中
# 只包含了0~9中的一个数字，所以可以使用这个函数来计算交叉熵损失。这个函数的第一个
# 参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。因为
# 标准答案是一个长度为10的一维数组，而该函数需要提供的是一个正确答案的数字，所以需
# 要使用tf.argmax函数来得到正确答案对应的类别编号。


# 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数， 
# 又要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了
# tf.control_dependencies和tf.group两种机制。下面两行程序和
# train_op = tf.group(train_step, variables_averages_op)是等价的。
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

除了使用验证数据集，还可以采用交叉验证（cross validation）的方式来验证模型效果。但因为神经网络训练时间本身就比较长，采用cross validation会花费大量时间。所以在海量数据的情况下，一般会更多地采用验证数据集的形式来评测模型的效果

只优化交叉熵的模型可以更好地拟合训练数据（交叉熵损失更小），但是却不能很好地挖掘数据中潜在的规律来判断未知的测试数据，所以在测试数据上的正确率低
```
## MNIST Softmax Regression
  - [mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
  - **MNIST 手写数字数据集**，每组数据包含两部分，手写图像数据 x 与对应的标签 y，每个图像包含 28x28 像素，所有数据划分成三部分
  	- 训练数据集 training data，55,000 组数据，mnist.train
  	- 测试数据集 test data，10,000 组数据，mnist.test
  	- 验证数据集 validation data，5,000 组数据，mnist.validation
  	```python
    import tensorflow as tf
  	# Load MNIST Data
  	from tensorflow.examples.tutorials.mnist import input_data
  	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
  	mnist.train.images.shape
  	# Out[3]: (55000, 784)	# 28 * 28 = 784
  	mnist.train.labels.shape
  	# Out[4]: (55000, 10)	# one-hot vectors，只在对应的数字处为1
  	mnist.train.labels[0]
  	# Out[5]: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.])
  	```
  - **实现回归模型** Implementing the Regression
    ```python
    # 特征值，None 指定数组的长度任意
    x = tf.placeholder(tf.float32, [None, 784])
    # 目标值
    y_ = tf.placeholder(tf.float32, [None, 10])
    # weights 和 biases，初始化成全0值
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 定义线性模型
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # 损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    ```
  - **模型训练 Training** 选择优化算法 optimization algorithm，改变变量值，使损失函数最小
    ```python
    # 使用梯度下降算法，使交叉熵最小，学习率为0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # 创建会话，并运行计算图
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    ```
  - **模型评估 Evaluating Model**
    ```python
    # tf.argmax 给出指定轴上最大元素的位置，此处 tf.argmax(y,1) 即模型的预测数值结果
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # 将 correct_prediction 的布尔型结果转化为float，并计算出正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # Out: 0.9077
    ```
    正确率90.77%，使用该简单模型的效果并不好
## MNIST exponential moving average
  ```python
  import tensorflow as tf
  from tensorflow.examples.tutorials.mnist import input_data

  LAYER1_NODE = 500
  MOVING_AVERAGE_DECAY = 0.99
  REGULARIZATION_RATE = 0.0001
  LEARNING_RATE_BASE = 0.8
  LEARNING_RATE_DECAY = 0.99
  TRAINING_STEPS = 10000
  BATCH_SIZE = 100

  ''' Two layers, ReLU activation function in the first layer '''
  inference_across_layers = lambda x, w1, b1, w2, b2: \
          tf.matmul(tf.nn.relu(tf.matmul(x, w1) + b1), w2) + b2

  def train(mnist):
      INPUT_NODE = mnist.train.images.shape[1]
      OUTPUT_NODE = mnist.train.labels.shape[1]

      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])
      weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1))
      biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
      weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1))
      biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

      y = inference_across_layers(x, weights1, biases1, weights2, biases2)

      ''' Exponential moving average to prevent over-fit '''
      global_step = tf.Variable(0, trainable=False)
      variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
      # tf.trainable_variables returns all the trainable variables
      variable_averages_op = variable_averages.apply(tf.trainable_variables())
      average_y = inference_across_layers(x,
                      variable_averages.average(weights1),
                      variable_averages.average(biases1),
                      variable_averages.average(weights2),
                      variable_averages.average(biases2)
                      )

      ''' Loss '''
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
      cross_entropy = tf.reduce_mean(cross_entropy)
      # L2 regularization loss
      regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
      regularization = regularizer(weights1) + regularizer(weights2)
      loss = cross_entropy + regularization

      ''' Train step with exponential decay learning rate '''
      learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                          mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      ''' Group up train_step and variables_averages_op '''
      with tf.control_dependencies([train_step, variable_averages_op]):
          train_op = tf.no_op(name='train')

      ''' Calculate accuracy '''
      correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      ''' Initialize session and start training '''
      with tf.Session() as sess:
          tf.global_variables_initializer().run()
          # Validate feed and test feed
          validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
          test_feed = {x: mnist.test.images, y_: mnist.test.labels}

          for i in range(TRAINING_STEPS):
              if i % 1000 == 0:
                  # Validate training per 1000 steps
                  validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
                  print("Steps = %d, validate accuracy using average model = %g" % (i, validate_accuracy))

              # Generate a test batch
              xs, ys = mnist.train.next_batch(BATCH_SIZE)
              sess.run(train_op, feed_dict={x: xs, y_: ys})

          ''' Finale test accuracy '''
          test_accuracy = sess.run(accuracy, feed_dict=test_feed)
          print("Steps = %d, test accuracy using average model = %g" % (i, test_accuracy))

  def main(argv=None):
      mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
      train(mnist)

  if __name__ == '__main__':
      # Tensorflow provided main function, will call the real main() function
      tf.app.run()
  ```
  **运行结果**
  ```python
  Steps = 0, validate accuracy using average model = 0.1292
  Steps = 1000, validate accuracy using average model = 0.9768
  Steps = 2000, validate accuracy using average model = 0.982
  Steps = 3000, validate accuracy using average model = 0.9834
  Steps = 4000, validate accuracy using average model = 0.983
  Steps = 5000, validate accuracy using average model = 0.9838
  Steps = 6000, validate accuracy using average model = 0.9834
  Steps = 7000, validate accuracy using average model = 0.9838
  Steps = 8000, validate accuracy using average model = 0.9852
  Steps = 9000, validate accuracy using average model = 0.984
  Steps = 9999, test accuracy using average model = 0.984
  ```
  从第 4000 轮开始，模型在验证数据集上的表现开始波动，这说明模型已经接近极小值了，所以迭代也就可以结束了
## FOO
