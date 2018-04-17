# ___2017 - 08 - 20 TensorFlow___
***

- [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)

# 目录
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [___2017 - 08 - 20 TensorFlow___](#2017-08-20-tensorflow)
- [目录](#目录)
- [FOO](#foo)
- [TensorFlow 基础](#tensorflow-基础)
	- [安装](#安装)
	- [启用 GPU 支持](#启用-gpu-支持)
	- [Hello World](#hello-world)
	- [基本概念 Tensors / Graph / Session](#基本概念-tensors-graph-session)
	- [placeholders / Variables](#placeholders-variables)
	- [损失函数 loss function](#损失函数-loss-function)
	- [tf.train API](#tftrain-api)
	- [tf.estimator API](#tfestimator-api)
	- [自定义模型 custom model](#自定义模型-custom-model)
	- [tf.estimator 的输入功能 Input Function](#tfestimator-的输入功能-input-function)
	- [模型使用 input_fn 的数据 ???](#模型使用-inputfn-的数据-)
- [应用示例](#应用示例)
	- [MNIST Softmax Regression](#mnist-softmax-regression)
	- [MNIST 多层卷积神经网络 CNN](#mnist-多层卷积神经网络-cnn)
	- [tf.estimator DNNClassifier 用于 Iris 数据集](#tfestimator-dnnclassifier-用于-iris-数据集)
	- [预测 Boston 房价的神经网络模型](#预测-boston-房价的神经网络模型)
- [TensorFlow Mechanics 101](#tensorflow-mechanics-101)
- [TensorBoard](#tensorboard)

<!-- /TOC -->
***

# FOO
  ```python
  ===
  a placeholder, a value that we'll input when we ask TensorFlow to run a computation.
  The shape argument to placeholder is optional, but it allows TensorFlow to automatically catch bugs stemming from inconsistent tensor shapes.
  ===
  A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. It can be used and even modified by the computation. For machine learning applications, one generally has the model parameters be Variables.
  ===
  Here mnist is a lightweight class which stores the training, validation, and testing sets as NumPy arrays. It also provides a function for iterating through data minibatches, which we will use below.
  Start TensorFlow InteractiveSession

  TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.
  ===
  Here we instead use the convenient InteractiveSession class, which makes TensorFlow more flexible about how you structure your code. It allows you to interleave operations which build a computation graph with ones that run the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.

  import tensorflow as tf
  sess = tf.InteractiveSession()

  We will also use tf.Session rather than tf.InteractiveSession. This better separates the process of creating the graph (model specification) and the process of evaluating the graph (model fitting). It generally makes for cleaner code. The tf.Session is created within a with block so that it is automatically destroyed once the block is exited.
  ===
  To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.

  TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. This approach is similar to that used in  a few machine learning libraries like Theano or Torch.

  The role of the Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run. See the Computation Graph section of Getting Started With TensorFlow for more detail.
  Build a Softmax Regression Model
  ===
  What TensorFlow actually did in that single line was to add new operations to the computation graph. These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.

  The returned operation train_step, when run, will apply the gradient descent updates to the parameters. Training the model can therefore be accomplished by repeatedly running train_step.

  We then run the train_step operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples. Note that you can replace any tensor in your computation graph using feed_dict -- it's not restricted to just placeholders
  ===
  For this small convolutional network, performance is actually nearly identical with and without dropout. Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks.
  ```
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
## 基本概念 Tensors / Graph / Session
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
## placeholders / Variables
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
## 损失函数 loss function
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
## tf.train API
  - **优化器 optimizers** 用于缓慢改变变量的值，使得 **损失函数 loss function** 最小
  - **梯度下降算法 gradient descent** 沿损失函数梯度下降最大的方向调整参数，**tf.gradients 方法** 用于计算模型的导数，**tf.train.GradientDescentOptimizer 方法** 用于自动完成参数选取
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
## tf.estimator API
  - **tf.estimator** TensorFlow 提供的上层接口，简化机器学习的架构，如训练 / 评估 / 管理数据集
  ```python
  # tf.estimator 实现线性模型
  # 声明特征列表，只包含一列数值型特征
  feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

  # tf.estimator 方法提供训练 / 评估模型，包含很多预定义的模型
  # LinearRegressor 用于线性回归
  estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

  # TensorFlow 提供一些用于读取 / 设置数据集的方法，分别定义训练集与测试集
  x_train = np.array([1., 2., 3., 4.])
  y_train = np.array([0., -1., -2., -3.])
  x_eval = np.array([2., 5., 8., 1.])
  y_eval = np.array([-1.01, -4.1, -7, 0.])
  # num_epochs 指定 batches 数量，batch_size 指定每个 batch 大小
  input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
  train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=True)

  # 模型训练，指定训练数据集，并指定迭代1000次
  estimator.train(input_fn=input_fn, steps=1000)
  # 分别在训练集与测试集上评估模型
  estimator.evaluate(input_fn=train_input_fn)
  # Out[19]: {'average_loss': 1.3505429e-08, 'global_step': 1000, 'loss': 5.4021715e-08}

  estimator.evaluate(input_fn=eval_input_fn)
  # Out[20]: {'average_loss': 0.0025362344, 'global_step': 1000, 'loss': 0.010144938}
  ```
## 自定义模型 custom model
  - 除 tf.estimator 的预定义模型，可以定义自己的模型 custom model，并保持 tf.estimator 的上层接口
  - **tf.estimator.Estimator** 用于定义模型，tf.estimator.LinearRegressor 是 tf.estimator.Estimator 的一个子类，自定义的模型不需要定义成一个子类，只需要实现一个 **model_fn 方法** 指定预测 / 训练步骤 / 损失函数等方法
  ```python
  # 自定义的线性回归模型
  # 参数：数据集, 目标值, mode
  def model_fn(features, labels, mode):
      # 线型模型与预测方法
      W = tf.get_variable("W", [1], dtype=tf.float64)
      b = tf.get_variable("b", [1], dtype=tf.float64)
      y = W * features['x'] + b
      # 损失子图 Loss sub-graph
      loss = tf.reduce_sum(tf.square(y-labels))
      # 训练子图 Training sub-graph
      global_step = tf.train.get_global_step()
      optimizer = tf.train.GradientDescentOptimizer(0.01)
      train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
      # EstimatorSpec 方法指定对应的方法
      return tf.estimator.EstimatorSpec(
          mode = mode,
          predictions = y,
          loss = loss,
          train_op = train)

  # Estimator 指定 model_fn
  estimator = tf.estimator.Estimator(model_fn=model_fn)
  # 定义数据集与训练 / 评估流程
  x_train = np.array([1., 2., 3., 4.])
  y_train = np.array([0., -1., -2., -3.])
  x_eval = np.array([2., 5., 8., 1.])
  y_eval = np.array([-1.01, -4.1, -7, 0.])

  input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
  train_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=True)

  # 训练与评估模型
  estimator.train(input_fn=input_fn, steps=1000)
  estimator.evaluate(input_fn=train_input_fn)
  # Out[22]: {'global_step': 1000, 'loss': 1.0836827e-11}

  estimator.evaluate(input_fn=eval_input_fn)
  Out[23]: {'global_step': 1000, 'loss': 0.010100709}
  ```
## tf.estimator 的输入功能 Input Function
  - **input_fn** 用于向 Estimator 中训练 train / 评估 evaluate / 预测 predict 方法传递特征 / 目标数据，包括数据预处理，如清除异常值
  - Input functions 的返回值中必须包含最终的特征 / 目标值
    - **feature_cols** 键值对的字典值，特征列名对应包含特征数据的 Tensors / SparseTensors
    - **labels** 用于预测的目标值 Tensor
  - **特征数据转化为 Tensors** 如果输入数据类型是 pandas dataframes / numpy arrays，可以使用 pandas_input_fn / numpy_input_fn 狗找 input_fn
    ```python
    import numpy as np
    # numpy input_fn.
    my_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x_data)},
        y=np.array(y_data),
        ...)
    ```
    ```python
    import pandas as pd
    # pandas input_fn.
    my_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({"x": x_data}),
        y=pd.Series(y_data),
        ...)
    ```
  - 对于稀疏数据集 sparse data，大多数数据值为0，可以使用 **SparseTensor**
    - **dense_shape** Tensor 的形状，如 dense_shape=[3,6]
    - **indices** 非0值的位置，如 indices=[[1,3], [2,4]]
    - **values** 非0值的值，如 values=[18, 3.6]
    ```python
    sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                    values=[6, 0.5],
                                    dense_shape=[3, 5])
    # 定义的 tensor 数据
    sess = tf.Session()
    sess.run(tf.sparse_tensor_to_dense(sparse_tensor))
    Out[30]:
    array([[ 0. ,  6. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ,  0.5]], dtype=float32)
    ```
## 模型使用 input_fn 的数据 ???
  - 模型 train / evaluate / predict 方法的 input_fn 参数用于传递用户自定义的 input function
    ```python
    classifier.train(input_fn=my_input_fn, steps=2000)
    ```
  - 通过创建接受一个 dataset 参数的 input_fn，train / evaluate / predict 数据可以使用统一的接口
    ```python
    # 分别定义的输入功能函数
    classifier.train(input_fn=input_fn_train, steps=2000)
    classifier.evaluate(input_fn=input_fn_test, steps=2000)
    classifier.predict(input_fn=input_fn_predict, steps=2000)
    ```
    input_fn 参数的值必须是一个函数，而不能是函数的返回值，如果需要向定义的 input_fn 中传递参数，直接在参数中传递将产生类型错误 TypeError
    ```python
    def my_input_fn(data_set):
        ...

    # 将产生类型错误 TypeError    
    classifier.train(input_fn=my_input_fn(training_set), steps=2000)
    ```
  - **方法一** 定义一个包装函数 wrapper function
    ```python
    def my_input_fn(data_set):
        ...

    def my_input_fn_training_set():
        return my_input_fn(training_set)

    classifier.train(input_fn=my_input_fn_training_set, steps=2000)
    ```
  - **方法二** 使用 python 的 functools.partial 方法，构造一个所有参数固定的新函数
    ```python
    classifier.train(
        input_fn=functools.partial(my_input_fn, data_set=training_set),
        steps=2000)
    ```
  - **方法三** 使用 lambda 包装函数
    ```python
    classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)
    ```
  - 使用 pandas dataframes / numpy arrays 数据的 input_fn 示例
    ```python
    # num_epochs and shuffle control how the input_fn iterates over the data
    import pandas as pd

    def get_input_fn_from_pandas(data_set, num_epochs=None, shuffle=True):
      return tf.estimator.inputs.pandas_input_fn(
          x=pdDataFrame(...),
          y=pd.Series(...),
          num_epochs=num_epochs,
          shuffle=shuffle)

    import numpy as np

    def get_input_fn_from_numpy(data_set, num_epochs=None, shuffle=True):
      return tf.estimator.inputs.numpy_input_fn(
          x={...},
          y=np.array(...),
          num_epochs=num_epochs,
          shuffle=shuffle)
    ```
***

# 应用示例
## MNIST Softmax Regression
  - **MNIST 手写数字数据集**，每组数据包含两部分，手写图像数据 x 与对应的标签 y，每个图像包含 28x28 像素，所有数据划分成三部分
  	- 训练数据集 training data，55,000 组数据，mnist.train
  	- 测试数据集 test data，10,000 组数据，mnist.test
  	- 验证数据集 validation data，5,000 组数据，mnist.validation
  	```python
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
    ```
  - **交叉熵 Cross-entropy** 评估模型的预测结果与目标值的差距，目标使该值尽量小
    ```python
    # 其中y'表示预测值，y表示目标值
    H(y', y) = -Σ(i)(y'(i) * log (y(i)))
    ```
    实现
    ```python
    # reduction_indices=[1] 指定在列轴上求和，随后 tf.reduce_mean 计算所有样本的平均值
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    ```
    使用更稳定的内建版本 **softmax_cross_entropy_with_logits**
    ```python
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
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
  - **完整代码**
    - [mnist_softmax.py](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
    ```python
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))    
    ```
## MNIST 多层卷积神经网络 CNN
  - **CNN** 多层卷积神经网络 Multilayer Convolutional Neural Network
  - **权重初始化 Weight Initialization** 初始化时加入少量的噪声，以 **打破对称性 Symmetry Breaking** 以及避免倒数为 0
    ```python
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    ```
  - **偏差初始化 Bias Initialization** 使用 **ReLU 神经元 neurons** 时，应将 bias 初始化成一组很小的正值，以避免神经元节点输出恒为0 dead neurons
    ```python
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    ```
  - **卷积和池化 Convolution and Pooling** TensorFlow 在卷积和池化上有很强的灵活性，包括确定 **边界 boundaries** / **步长 stride size**
    ```python
    # 卷积 Convolution 使用步长 stride = 1, 边距 padded = 0，保证输出的大小与输入相同
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # 池化 Pooling 使用传统的 2x2 大小的模板做 max pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    ```
  - **第一层卷积 First Convolutional Layer** 由一个卷积接一个 max pooling 完成
    - 卷积在每个 5x5 的 patch 中算出 **32 个特征**
    - 卷积的权重形状是 [5, 5, 1, 32]，前两个维度是patch的大小，后两个维度是 [输入的通道数目, 输出的通道数目]
    - 对于每一个输出通道都有一个对应的偏置量 bias
    ```python
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    ```
  - 为了应用这一层，将 x 转换为一个 **4维tensor** 其中 2 / 3 维对应图片的宽 / 高，最后一维代表图片的颜色通道数，灰度图通道数为1，rgb彩色图为3
    ```python
    x_image = tf.reshape(x, [-1,28,28,1])
    ```
    将 **x_image** 与 **权重向量 weight** 进行卷积，加上 **偏置项 bias**，然后应用 **ReLU 激活函数**，最后进行 **max pooling**
    ```python
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # max_pool_2x2 将图片大小缩减到 14x14
    h_pool1 = max_pool_2x2(h_conv1)
    ```
  - **第二层卷积 Second Convolutional Layer** 几个类似的层堆叠起来，构建一个更深的网络，第二层中每个 5x5 的 patch 计算出 **64 个特征**
    ```python
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 图片大小缩减到 7x7
    h_pool2 = max_pool_2x2(h_conv2)
    ```
  - **密集连接层 Densely Connected Layer** 现在图片尺寸减小到 7x7，加入一个有1024个神经元的全连接层，用于处理整个图片
    ```python
    # 将第二层池化的结果向量转置 reshape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # 乘上权重矩阵，加上偏置，然后对其使用ReLU
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    ```
  - **Dropout 减小过拟合 overfitting** 输出层 readout layer 之前加入 dropout
    - 使用一个 placeholder 来表示 **在 dropout 层一个神经元的输出保持不变的概率**，这样可以 **在训练过程中启用dropout，在测试过程中关闭dropout**
    - TensorFlow的 **tf.nn.dropout方法** 除了可以屏蔽神经元的输出，还可以自动处理神经元输出值的 **定比 scale**，因此 dropout 不需要额外的 scaling
    ```python
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    ```
  - **输出层 Readout Layer** 类似于 softmax regression 的输出层
    ```python
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    ```
  - **训练和评估模型 Train and Evaluate the Model** 类似于单层 SoftMax 的测试 / 评估方法，区别在于
    - 使用更复杂的 **ADAM 优化器** 代替梯度最速下降 steepest gradient descent optimizer
    - 在 feed_dict 中加入额外的 **参数 keep_prob 控制 dropout 比例**
    - 每 100 次迭代输出一次日志
    ```python
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    ```
  - **完整代码**
    - [mnist_deep.py](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py)
    ```python
    # Weight Initialization
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    # bias Initialization
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 卷积 Convolution 使用步长 stride = 1, 边距 padded = 0，保证输出的大小与输入相同
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # 池化 Pooling 使用传统的 2x2 大小的模板做 max pooling
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    # Dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # max_pool_2x2 将图片大小缩减到 14x14
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 图片大小缩减到 7x7
    h_pool2 = max_pool_2x2(h_conv2)

    # Densely Connected Layer
    # 将第二层池化的结果向量转置 reshape
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # 乘上权重矩阵，加上偏置，然后对其使用ReLU
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Train and Evaluate the Model
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    ```
    运行结果
    ```python
    step 19900, training accuracy 1
    test accuracy 0.9922
    ```
    最终测试集上的准确率大概是 99.2%
## tf.estimator DNNClassifier 用于 Iris 数据集
  - 使用 Iris 数据集，该数据集随机分割成两个 csv 文件
    - 训练数据集，120 个样本
    - 测试数据集，30 个样本
  - **导入模块 / 数据集**
    ```python
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import os
    import urllib

    import tensorflow as tf
    import numpy as np

    IRIS_TRAINING = "iris_training.csv"
    IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

    IRIS_TEST = "iris_test.csv"
    IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        raw = raw.decode()
        with open(IRIS_TRAINING,'w') as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        raw = raw.decode()
        with open(IRIS_TEST,'w') as f:
            f.write(raw)
    ```
  - **learn.datasets.base.load_csv_with_header 方法加载csv文件**，需要三个参数
    - **文件名 filename** 指向 CSV 文件
    - **目标值类型 target_dtype** 数据集中目标值 target 的类型
    - **特征值类型 features_dtype** 数据集中特征值 feature 的类型
    ```python
    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)
    ```
  - **tf.contrib.learn** 中的数据集是 **命名元组 named tuples**，通过 **data** 与 **target** 域可以访问数据集的特征值与目标值
    ```python
    training_set.data.shape
    Out[3]: (120, 4)

    training_set.target.shape
    Out[4]: (120,)

    training_set.data[:3]
    Out[6]:
    array([[ 6.4000001 ,  2.79999995,  5.5999999 ,  2.20000005],
           [ 5.        ,  2.29999995,  3.29999995,  1.        ],
           [ 4.9000001 ,  2.5       ,  4.5       ,  1.70000005]], dtype=float32)

    training_set.target[:3]
    Out[7]: array([2, 1, 2])
    ```
  - **构造深度神经网络分类模型 Deep Neural Network Classifier** tf.estimator 提供多种预定义的模型用于训练 / 评估，称为 **Estimators**
    - **tf.feature_column.numeric_column** 定义特征列为数字类型，每一项数据有 4 个特征
    - 使用 **tf.estimator.DNNClassifier** Deep Neural Network Classifier model
      - **feature_columns** 特征列
      - **hidden_units** 隐含层，分别定义每一层的神经元数量
      - **n_classes** 目标值数量
      - **model_dir** 模型训练中的数据以及 TensorBoard 的结果目录
    ```python
    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")
    ```
  - **定义输入的 pipeline** tf.estimator API 使用 **输入功能 input function** 为模型提供数据，**tf.estimator.inputs.numpy_input_fn** 用于定义输入的 pipeline
    ```python
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)
    ```
  - **DNNClassifier 模型训练 fit** 使用模型的 train 方法，train_input_fn 作为 input_fn
    ```python
    # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)
    # The state of the model is preserved in the classifier
    # which means you can train iteratively if you like
    # For example, the above is equivalent to the following
    # classifier.train(input_fn=train_input_fn, steps=1000)
    # classifier.train(input_fn=train_input_fn, steps=1000)
    ```
  - **评估模型准确率 Evaluate Model Accuracy** 使用 **evaluate 方法** 在测试数据集上验证模型准确率
    ```python
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
    ```
    运行结果
    ```python
    Test Accuracy: 0.966667
    ```
    其中参数中的 **num_epochs=1** 指定 test_input_fn 遍历数据一次，然后抛出异常 **OutOfRangeError**，该异常通知分类器停止评估
  - **分类新数据 Classify New Samples** 模型的 **predict 方法** 用于分类新数据
    ```python
    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "New Samples, Class Predictions:    {}\n"
        .format(predicted_classes))
    ```
    运行结果
    ```python
    New Samples, Class Predictions:    [array([b'1'], dtype=object), array([b'2'], dtype=object)]
    ```
  - **完整代码**
    ```python
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import os
    import urllib

    import tensorflow as tf
    import numpy as np

    IRIS_TRAINING = "iris_training.csv"
    IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

    IRIS_TEST = "iris_test.csv"
    IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        raw = raw.decode()
        with open(IRIS_TRAINING,'w') as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        raw = raw.decode()
        with open(IRIS_TEST,'w') as f:
            f.write(raw)

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                              hidden_units=[10, 20, 10],
                              n_classes=3,
                              model_dir="/tmp/iris_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=2000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "New Samples, Class Predictions:    {}\n"
        .format(predicted_classes))
    ```
## 预测 Boston 房价的神经网络模型
  - [boston.py](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/input_fn/boston.py)
  - [boston_train.csv](download.tensorflow.org/data/boston_train.csv)
  - [boston_test.csv](download.tensorflow.org/data/boston_test.csv)
  - [boston_predict.csv](download.tensorflow.org/data/boston_predict.csv)
  - 特征
    ```markdown
    | Feature | Description                                                     |
    | ------- | --------------------------------------------------------------- |
    | CRIM    | Crime rate per capita                                           |
    | ZN      | Fraction of residential land zoned to permit 25,000+ sq ft lots |
    | INDUS   | Fraction of land that is non-retail business                    |
    | NOX     | Concentration of nitric oxides in parts per 10 million          |
    | RM      | Average Rooms per dwelling                                      |
    | AGE     | Fraction of owner-occupied residences built before 1940         |
    | DIS     | Distance to Boston-area employment centers                      |
    | TAX     | Property tax rate per $10,000                                   |
    | PTRATIO | Student-teacher ratio                                           |
    ```
  - 预测的目标值 median value MEDV
  - 加载数据集，并将 log 等级设置为 INFO
    ```python
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import itertools
    import pandas as pd
    import tensorflow as tf

    tf.logging.set_verbosity(tf.logging.INFO)

    COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
               "dis", "tax", "ptratio", "medv"]
    FEATURES = ["crim", "zn", "indus", "nox", "rm",
                "age", "dis", "tax", "ptratio"]
    LABEL = "medv"

    training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)
    ```
  - 定义 FeatureColumns，创建回归模型 Regressor
    ```python
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[10, 10],
                                        model_dir="/tmp/boston_model")
    ```
  - 定义输入功能 input_fn
    - 参数 data_set，可以用于training_set / test_set / prediction_set
    - 参数 num_epochs，控制数据集迭代的次数，用于训练时置为 None，表示不限迭代次数，评估与预测时，置为1
    - 参数 shuffle，是否进行数据混洗，用于训练时置为 True，评估与预测时置为 False
    ```python
    def get_input_fn(data_set, num_epochs=None, shuffle=True):
      tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y = pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)
    ```
  - 模型训练
    ```python
    regressor.train(input_fn=get_input_fn(training_set), steps=5000)
    ```
  - 模型评估
    ```python
    ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))
    # Loss: 1608.965698
    ```
  - 预测
    ```python
    y = regressor.predict(
        input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
    # .predict() returns an iterator of dicts; convert to a list and print
    # predictions
    predictions = list(p["predictions"][0] for p in itertools.islice(y, 6))
    print("Predictions: {}".format(str(predictions)))
    ```
    运行结果
    ```python
    Predictions: [35.306267, 18.697575, 24.233162, 35.991249, 16.141064, 20.229273]
    ```
***

# TensorFlow Mechanics 101
***

# TensorBoard
***
