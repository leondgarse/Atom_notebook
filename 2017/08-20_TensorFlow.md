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
***

# MNIST
	- MNIST 手写数字数据集，每组数据包含两部分，手写图像数据 x 与对应的标签 y，每个图像包含 28x28 像素，所有数据划分成三部分
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
	- Softmax Regression
		-
We know that every image in MNIST is of a handwritten digit between zero and nine. So there are only ten possible things that a given image can be. We want to be able to look at an image and give the probabilities for it being each digit. For example, our model might look at a picture of a nine and be 80% sure it's a nine, but give a 5% chance to it being an eight (because of the top loop) and a bit of probability to all the others because it isn't 100% sure.

This is a classic case where a softmax regression is a natural, simple model. If you want to assign probabilities to an object being one of several different things, softmax is the thing to do, because softmax gives us a list of values between 0 and 1 that add up to 1. Even later on, when we train more sophisticated models, the final step will be a layer of softmax.

A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.

To tally up the evidence that a given image is in a particular class, we do a weighted sum of the pixel intensities. The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor.

The following diagram shows the weights one model learned for each of these classes. Red represents negative weights, while blue represents positive weights.


# FOO
Here mnist is a lightweight class which stores the training, validation, and testing sets as NumPy arrays. It also provides a function for iterating through data minibatches, which we will use below.
Start TensorFlow InteractiveSession

TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.

Here we instead use the convenient InteractiveSession class, which makes TensorFlow more flexible about how you structure your code. It allows you to interleave operations which build a computation graph with ones that run the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.

import tensorflow as tf
sess = tf.InteractiveSession()

Computation Graph

To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.

TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. This approach is similar to that used in Theano or Torch.

The role of the Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run. See the Computation Graph section of Getting Started With TensorFlow for more detail.
Build a Softmax Regression Model

In this section we will build a softmax regression model with a single linear layer. In the next section, we will extend this to the case of softmax regression with a multilayer convolutional network.
Placeholders

We start building the computation graph by creating nodes for the input images and target output classes.

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

Here x and y_ aren't specific values. Rather, they are each a placeholder -- a value that we'll input when we ask TensorFlow to run a computation.

The input images x will consist of a 2d tensor of floating point numbers. Here we assign it a shape of [None, 784], where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image, and None indicates that the first dimension, corresponding to the batch size, can be of any size. The target output classes y_ will also consist of a 2d tensor, where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine) the corresponding MNIST image belongs to.

The shape argument to placeholder is optional, but it allows TensorFlow to automatically catch bugs stemming from inconsistent tensor shapes.
Variables

We now define the weights W and biases b for our model. We could imagine treating these like additional inputs, but TensorFlow has an even better way to handle them: Variable. A Variable is a value that lives in TensorFlow's computation graph. It can be used and even modified by the computation. In machine learning applications, one generally has the model parameters be Variables.

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

We pass the initial value for each parameter in the call to tf.Variable. In this case, we initialize both W and b as tensors full of zeros. W is a 784x10 matrix (because we have 784 input features and 10 outputs) and b is a 10-dimensional vector (because we have 10 classes).

Before Variables can be used within a session, they must be initialized using that session. This step takes the initial values (in this case tensors full of zeros) that have already been specified, and assigns them to each Variable. This can be done for all Variables at once:

sess.run(tf.global_variables_initializer())

Predicted Class and Loss Function

We can now implement our regression model. It only takes one line! We multiply the vectorized input images x by the weight matrix W, add the bias b.

y = tf.matmul(x,W) + b

We can specify a loss function just as easily. Loss indicates how bad the model's prediction was on a single example; we try to minimize that while training across all the examples. Here, our loss function is the cross-entropy between the target and the softmax activation function applied to the model's prediction. As in the beginners tutorial, we use the stable formulation:

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes, and tf.reduce_mean takes the average over these sums.
Train the Model

Now that we have defined our model and training loss function, it is straightforward to train using TensorFlow. Because TensorFlow knows the entire computation graph, it can use automatic differentiation to find the gradients of the loss with respect to each of the variables. TensorFlow has a variety of built-in optimization algorithms. For this example, we will use steepest gradient descent, with a step length of 0.5, to descend the cross entropy.

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

What TensorFlow actually did in that single line was to add new operations to the computation graph. These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.

The returned operation train_step, when run, will apply the gradient descent updates to the parameters. Training the model can therefore be accomplished by repeatedly running train_step.

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

We load 100 training examples in each training iteration. We then run the train_step operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples. Note that you can replace any tensor in your computation graph using feed_dict -- it's not restricted to just placeholders.
Evaluate the Model

How well did our model do?

First we'll figure out where we predicted the correct label. tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the true label. We can use tf.equal to check if our prediction matches the truth.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean. For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

Finally, we can evaluate our accuracy on the test data. This should be about 92% correct.

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
