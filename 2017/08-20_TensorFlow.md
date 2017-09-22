# ___2017 - 08 - 20 TensorFlow___
***

- [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started)
- [TensorFlow 官方文档中文版](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)
- TensorBoard
# 目录
# FOO
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
  ===
  To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.

  TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. This approach is similar to that used in  a few machine learning libraries like Theano or Torch.

  The role of the Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run. See the Computation Graph section of Getting Started With TensorFlow for more detail.
  Build a Softmax Regression Model
  ===
  What TensorFlow actually did in that single line was to add new operations to the computation graph. These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.

  The returned operation train_step, when run, will apply the gradient descent updates to the parameters. Training the model can therefore be accomplished by repeatedly running train_step.

  We then run the train_step operation, using feed_dict to replace the placeholder tensors x and y_ with the training examples. Note that you can replace any tensor in your computation graph using feed_dict -- it's not restricted to just placeholders
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
## MNIST 多层卷积神经网络 CNN Multilayer Convolutional Neural Network


## FOO
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
  - **第一层卷积 First Convolutional Layer**  由一个卷积接一个 max pooling 完成
    - 卷积在每个 5x5 的 patch 中算出 32 个特征
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


We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling

We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. The max_pool_2x2 method will reduce the image size to 14x14.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

第二层卷积

为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

密集连接层

现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

Dropout

为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale。

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

输出层

最后，我们添加一个softmax层，就像前面的单层softmax regression一样。

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

训练和评估模型

这个模型的效果如何呢？

为了进行训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，只是我们会用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。然后每100次迭代输出一次日志。

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

以上代码，在最终测试集上的准确率大概是99.2%。

目前为止，我们已经学会了用TensorFlow快捷地搭建、训练和评估一个复杂一点儿的深度学习模型。

Second Convolutional Layer

In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

Densely Connected Layer

Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. We reshape the tensor from the pooling layer into a batch of vectors, multiply by a weight matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

Dropout

To reduce overfitting, we will apply dropout before the readout layer. We create a placeholder for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing. TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them, so dropout just works without any additional scaling.1

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

Readout Layer

Finally, we add a layer, just like for the one layer softmax regression above.

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

Train and Evaluate the Model

How well does this model do? To train and evaluate it we will use code that is nearly identical to that for the simple one layer SoftMax network above.

The differences are that:

    We will replace the steepest gradient descent optimizer with the more sophisticated ADAM optimizer.

    We will include the additional parameter keep_prob in feed_dict to control the dropout rate.

    We will add logging to every 100th iteration in the training process.

We will also use tf.Session rather than tf.InteractiveSession. This better separates the process of creating the graph (model specification) and the process of evaluating the graph (model fitting). It generally makes for cleaner code. The tf.Session is created within a with block so that it is automatically destroyed once the block is exited.

Feel free to run this code. Be aware that it does 20,000 training iterations and may take a while (possibly up to half an hour), depending on your processor.

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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

The final test set accuracy after running this code should be approximately 99.2%.

We have learned how to quickly and easily build, train, and evaluate a fairly sophisticated deep learning model using TensorFlow.

1: For this small convolutional network, performance is actually nearly identical with and without dropout. Dropout is often very effective at reducing overfitting, but it is most useful when training very large neural networks
