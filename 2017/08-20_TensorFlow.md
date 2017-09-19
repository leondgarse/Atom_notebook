# ___2017 - 08 - 20 TensorFlow___
***

- [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)
- TensorBoard
# 目录
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [___2017 - 08 - 20 TensorFlow___](#2017-08-20-tensorflow)
- [目录](#目录)
- [TensorFlow 基础](#tensorflow-基础)
	- [安装](#安装)
	- [启用 GPU 支持](#启用-gpu-支持)
	- [Hello World](#hello-world)
	- [基本概念](#基本概念)
	- [tf.train API](#tftrain-api)
	- [tf.estimator API](#tfestimator-api)
	- [自定义模型 custom model](#自定义模型-custom-model)

<!-- /TOC -->
***

# TensorFlow 基础
## 安装
  - [Installing TensorFlow from Sources](https://www.tensorflow.org/install/install_sources)
  - [Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)
  - Install newest ternsorflow version in anaconda
    ```shell
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
## 基本概念
  - **TensorFlow Core / High-level API** 最底层API，提供对程序运行的完全控制，其他上层接口基于 TensorFlow Core，上层接口的调用更易用且一致，如 **tf.estimator** 用于管理数据集 / 模型 / 训练以及评估
  - **导入 TensorFlow**
    ```python
    import tensorflow as tf
    ```
  - **Tensors** TensorFlow 中的数据单元，包含一个任意维度的数组，tensor 的 **秩 rank** 表示数组的维度
    ```python
    3 # a rank 0 tensor; this is a scalar with shape []
    [1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
    [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
    [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
    ```
  - TensorFlow 的程序一般由两部分组成
    - 创建计算图 Building the computational graph
    - 运行计算图 Running the computational graph
  - **计算图 Computational Graph** 由一系列 TensorFlow 操作作为节点组成的图，每个节点使用零个或多个 tensor 作为 **输入**，并产生一个 tensor 作为 **输出**，如 **constant** 没有输入，产生一个内部存储的值
    ```python
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # also tf.float32 implicitly

    # 作为print参数输出时，并不会输出内部值，只有在运行时才会计算输出
    print(node1, node2)
    Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
    ```
  - **会话 Session** 可使用 **run** 方法运行创建好的图
    ```python
    sess = tf.Session()
    sess.run([node1, node2])
    Out[15]: [3.0, 4.0]
    ```
  - **运算符** 作为节点在创建图时指定
    ```python
    node3 = tf.add(node1, node2)
    node3
    Out[17]: <tf.Tensor 'Add_1:0' shape=() dtype=float32>

    sess.run(node3)
    Out[18]: 7.0

    sess.run(node3 * 3)
    Out[19]: 21.0

    sess.run(node3 + node3)
    Out[20]: 14.0
    ```
  - **占位符 placeholders** 可以在运行时指定的参数
    ```python
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b

    # 运行时通过字典参数指定值
    sess.run(adder_node, {a:3, b:4.5})
    Out[22]: 7.5

    sess.run(adder_node, {a:[1, 2], b:[2, 4]})
    Out[23]: array([ 3.,  6.], dtype=float32)
    ```
  - **变量 Variables** 向图中添加可训练的参数，包含初值与类型
    ```python
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    # 线性模型
    linear_model = W * x + b
    ```
    调用 **tf.Variable** 不会初始化变量，在 TensorFlow 中需要显式调用 **tf.global_variables_initializer** 方法创建 **子图 sub-graph**，并调用 **sess.run** 运行子图完成变量初始化
    ```python
    init = tf.global_variables_initializer()
    sess.run(init)

    sess.run(linear_model, {x : [1, 2, 3, 4]})
    Out[37]: array([ 0.        ,  0.30000001,  0.60000002,  0.90000004], dtype=float32)
    ```
  - **损失函数 loss function** 评估当前模型的预测结果与目标值的距离
    ```python
    # 标准损失模型 standard loss model
    # 目标值
    y = tf.placeholder(tf.float32)
    # 各个预测结果与目标值距离的平方
    squared_deltas = tf.square(linear_model - y)
    # 取和
    loss = tf.reduce_sum(squared_deltas)
    sess.run(loss, {x : [1, 2, 3, 4], y : [0, -1, -2, -3]})
    Out[42]: 23.66
    ```
  - **调整变量值 assign** 变量值可以通过 **tf.assign** 等函数修改
    ```python
    # W = -1, b = 1 是该模型的理想值
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    Out[45]: [array([-1.], dtype=float32), array([ 1.], dtype=float32)]

    sess.run(loss, {x : [1, 2, 3, 4], y : [0, -1, -2, -3]})
    Out[46]: 0.0
    ```
## tf.train API
  TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone. Consequently, TensorFlow can automatically produce derivatives given only a description of the model using the function tf.gradients. For simplicity, optimizers typically do this for you. For example,
  ```python
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(loss)

  sess.run(init)  # reset values to incorrect defaults.
  for i in range(1000):
      sess.run(train, {x : [1, 2, 3, 4], y : [0, -1, -2, -3]})
  sess.run([W, b])
  # Out[52]: [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]

  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x : [1, 2, 3, 4], y : [0, -1, -2, -3]})
  curr_W, curr_b, curr_loss
  Out[55]:
  (array([-0.9999969], dtype=float32),
   array([ 0.99999082], dtype=float32),
   5.6999738e-11)
  ```
  Now we have done actual machine learning! Although doing this simple linear regression doesn't require much TensorFlow core code, more complicated models and methods to feed data into your model necessitate more code. Thus TensorFlow provides higher level abstractions for common patterns, structures, and functionality. We will learn how to use some of these abstractions in the next section.
## tf.estimator API
  tf.estimator is a high-level TensorFlow library that simplifies the mechanics of machine learning, including the following:

      running training loops
      running evaluation loops
      managing data sets

  tf.estimator defines many common models.
  Basic usage

  Notice how much simpler the linear regression program becomes with tf.estimator:

  ```python
  feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
  estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
  x_train = np.array([1., 2., 3., 4.])
  y_train = np.array([0., -1., -2., -3.])
  x_eval = np.array([2., 5., 8., 1.])
  y_eval = np.array([-1.01, -4.1, -7, 0.])
  input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
  train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=True)

  estimator.train(input_fn=input_fn, steps=1000)
  estimator.evaluate(input_fn=train_input_fn)
  Out[19]: {'average_loss': 1.3505429e-08, 'global_step': 1000, 'loss': 5.4021715e-08}

  estimator.evaluate(input_fn=eval_input_fn)
  Out[20]: {'average_loss': 0.0025362344, 'global_step': 1000, 'loss': 0.010144938}
  ```
## 自定义模型 custom model
  tf.estimator does not lock you into its predefined models. Suppose we wanted to create a custom model that is not built into TensorFlow. We can still retain the high level abstraction of data set, feeding, training, etc. of tf.estimator. For illustration, we will show how to implement our own equivalent model to LinearRegressor using our knowledge of the lower level TensorFlow API.

  To define a custom model that works with tf.estimator, we need to use tf.estimator.Estimator. tf.estimator.LinearRegressor is actually a sub-class of tf.estimator.Estimator. Instead of sub-classing Estimator, we simply provide Estimator a function model_fn that tells tf.estimator how it can evaluate predictions, training steps, and loss. The code is as follows:
  ```python
  def model_fn(features, labels, mode):
      W = tf.get_variable("W", [1], dtype=tf.float64)
      b = tf.get_variable("b", [1], dtype=tf.float64)
      y = W * features['x'] + b

      loss = tf.reduce_sum(tf.square(y-labels))
      global_step = tf.train.get_global_step()
      optimizer = tf.train.GradientDescentOptimizer(0.01)
      train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

      return tf.estimator.EstimatorSpec(
          mode = mode,
          predictions = y,
          loss = loss,
          train_op = train)
  estimator = tf.estimator.Estimator(model_fn=model_fn)

  x_train = np.array([1., 2., 3., 4.])
  y_train = np.array([0., -1., -2., -3.])
  x_eval = np.array([2., 5., 8., 1.])
  y_eval = np.array([-1.01, -4.1, -7, 0.])
  input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
  train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=True)

  estimator.train(input_fn=input_fn, steps=1000)
  estimator.evaluate(input_fn=train_input_fn)
  Out[22]: {'global_step': 1000, 'loss': 1.0836827e-11}

  estimator.evaluate(input_fn=eval_input_fn)
  Out[23]: {'global_step': 1000, 'loss': 0.010100709}
  ```
