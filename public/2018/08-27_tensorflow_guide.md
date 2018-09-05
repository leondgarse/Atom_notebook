# ___2018 - 08 - 27 Tensorflow Programmer's Guide___
***
- [Programmer's Guide](https://www.tensorflow.org/programmers_guide/)
- [ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50)
- [mnist_eager.py](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py)
- [tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples).
```python
InternalError: Could not find valid device for node name: "SparseSoftmaxCrossEntropyWithLogits"
```

# Keras
## Keras 简介
  - **Keras** high-level API，用于创建与训练深度学习模型
  - **tf.keras** 是 Keras 的 TensorFlow 实现，支持 TensorFlow 的功能，如 eager execution / tf.data pipelines / Estimators
    ```python
    import tensorflow as tf
    from tensorflow import keras
    ```
  - **Eager execution** tf.keras 创建模型的 API 都支持 eager execution，也可以用于自定义模型 / 层 的序列化保存
## 建立模型
  - **tf.keras.Sequential** 顺序模型 Sequential model，将各个 layer 叠加到一起
    ```python
    ''' To build a simple, fully-connected network (i.e. multi-layer perceptron) '''
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense(64, activation='relu'))
    # Add another:
    model.add(keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(10, activation='softmax'))
    ```
  - **神经网络层配置** 通用参数
    - **activation** 设置激活函数，使用字符串指定内置的激活函数，默认不使用激活函数
    - **kernel_initializer** / **bias_initializer** 权重 weights 的初始化方式，字符串指定内置的初始化函数，默认使用 `Glorot uniform`
    - **kernel_regularizer** / **bias_regularizer** 权重 weights 的正则化方式，默认不使用正则化
    ```python
    # Create a sigmoid layer:
    layers.Dense(64, activation='sigmoid')
    # Or:
    layers.Dense(64, activation=tf.sigmoid)

    # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
    layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
    # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
    layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

    # A linear layer with a kernel initialized to a random orthogonal matrix:
    layers.Dense(64, kernel_initializer='orthogonal')
    # A linear layer with a bias vector initialized to 2.0s:
    layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))
    ```
## 训练
  - **model.compile** 创建好模型结构以后，设置模型训练过程
    ```python
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
    ```
  - **model.compile 参数**
    - **optimizer** 优化器，指定 `tf.train module` 中的方法，如 `AdamOptimizer` / `RMSPropOptimizer` / `GradientDescentOptimizer`
    - **loss** 损失函数，使用字符串或 `tf.keras.losses` 中的方法，通常使用 `mean square error (mse)` / `categorical_crossentropy` / `binary_crossentropy`
    - **metrics** 训练过程度量监控，使用字符串或 `tf.keras.metrics` 中的方法
    ```python
    # Configure a model for mean-squared error regression.
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                 loss='mse',       # mean squared error
                 metrics=['mae'])  # mean absolute error

    # Configure a model for categorical classification.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
                 loss=keras.losses.categorical_crossentropy,
                 metrics=[keras.metrics.categorical_accuracy])
    ```
  - **model.fit 输入 Numpy 数据** 小的数据集可以直接使用 NumPy arrays 作为训练 / 评估的输入
    ```python
    import numpy as np

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    model.fit(data, labels, epochs=10, batch_size=32)
    ```
  - **model.fit 参数**
    - **epochs** 迭代 epoch 次数，每次 epoch 代表遍历整个数据集一次
    - **batch_size** 每次训练使用的 batch 大小
    - **validation_data** 验证数据用于监控模型训练过程中的表现
    ```python
    import numpy as np

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    val_data = np.random.random((100, 32))
    val_labels = np.random.random((100, 10))

    model.fit(data, labels, epochs=10, batch_size=32,
             validation_data=(val_data, val_labels))
    ```
  - **model.fit 输入 tf.data.datasets 数据** 用于输入大型数据集
    ```python
    # Instantiates a toy dataset instance:
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)
    dataset = dataset.repeat()

    # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
    model.fit(dataset, epochs=10, steps_per_epoch=30)
    ```
    - **steps_per_epoch 参数** 指定模型每次 epoch 训练的次数，**batch_size** 不需要再指定
    ```python
    # 指定 validation_data 验证数据集
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataset = val_dataset.batch(32).repeat()

    model.fit(dataset, epochs=10, steps_per_epoch=30,
             validation_data=val_dataset,
             validation_steps=3)
    ```
## 评估与预测
  - **tf.keras.Model.evaluate** / **tf.keras.Model.predict** 可以使用 NumPy 数据 / `tf.data.Dataset` 用于 评估 / 预测
  - **model.evaluate** 评估模型效果
    ```python
    model.evaluate(x, y, batch_size=32)

    model.evaluate(dataset, steps=30)
    ```
  - **model.predict** 预测，输出模型最后一层的结果
    ```python
    model.predict(x, batch_size=32)

    model.predict(dataset, steps=30)
    ```
## 建立复杂模型
  - **Keras functional API** 建立任意组合的模型
    - **Multi-input models** 多个输入层的模型
    - **Multi-output models** 多个输出层的模型
    - **Models with shared layers** 同一层重复调用多次的模型
    - **Models with non-sequential data flows (e.g. residual connections)** 非顺序数据流模型，如残差网络连接 residual connections
  - **建立过程**
    - keras 的 layer 可以调用，并返回一个 tensor
    - 输入 / 输出 tensor 用于定义 `tf.keras.Model`
    - 训练过程类似顺序模型 Sequential model
    ```python
    ''' The following example uses the functional API to build a simple, fully-connected network '''

    inputs = keras.Input(shape=(32,))  # Returns a placeholder tensor

    # A layer instance is callable on a tensor, and returns a tensor.
    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.Dense(64, activation='relu')(x)
    predictions = keras.layers.Dense(10, activation='softmax')(x)

    # Instantiate the model given inputs and outputs.
    model = keras.Model(inputs=inputs, outputs=predictions)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Trains for 5 epochs
    model.fit(data, labels, batch_size=32, epochs=5)
    ```
## Model 与 Layer 自定义继承类
  - **tf.keras.Model** 自定义继承类，建立完全自定义的模型
    - **__init__ 方法** 中建立层级结构，并定义成类属性
    - **call 方法** 中定义前向传播过程
    ```python
    class MyModel(keras.Model):

     def __init__(self, num_classes=10):
       super(MyModel, self).__init__(name='my_model')
       self.num_classes = num_classes
       # Define your layers here.
       self.dense_1 = keras.layers.Dense(32, activation='relu')
       self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

     def call(self, inputs):
       # Define your forward pass here,
       # using layers you previously defined (in `__init__`).
       x = self.dense_1(inputs)
       return self.dense_2(x)

     def compute_output_shape(self, input_shape):
       # You need to override this function if you want to use the subclassed model
       # as part of a functional-style model.
       # Otherwise, this method is optional.
       shape = tf.TensorShape(input_shape).as_list()
       shape[-1] = self.num_classes
       return tf.TensorShape(shape)


    # Instantiates the subclassed model.
    model = MyModel(num_classes=10)

    # The compile step specifies the training configuration.
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Trains for 5 epochs.
    model.fit(data, labels, batch_size=32, epochs=5)
    ```
  - **tf.keras.layers.Layer** 自定义继承类，建立自定义的层，实现以下方法
    - **build** 创建该层的权重 weights，使用 `add_weight` 方法添加权重
    - **call** 定义前向传播过程
    - **compute_output_shape** 指定如何根据输入的维度计算输出的维度
    - **get_config** / **from_config**，可选的方法，实现这两个方法可以用于 serialized
    ```python
    class MyLayer(keras.layers.Layer):

     def __init__(self, output_dim, **kwargs):
       self.output_dim = output_dim
       super(MyLayer, self).__init__(**kwargs)

     def build(self, input_shape):
       shape = tf.TensorShape((input_shape[1], self.output_dim))
       # Create a trainable weight variable for this layer.
       self.kernel = self.add_weight(name='kernel',
                                     shape=shape,
                                     initializer='uniform',
                                     trainable=True)
       # Be sure to call this at the end
       super(MyLayer, self).build(input_shape)

     def call(self, inputs):
       return tf.matmul(inputs, self.kernel)

     def compute_output_shape(self, input_shape):
       shape = tf.TensorShape(input_shape).as_list()
       shape[-1] = self.output_dim
       return tf.TensorShape(shape)

     def get_config(self):
       base_config = super(MyLayer, self).get_config()
       base_config['output_dim'] = self.output_dim

     @classmethod
     def from_config(cls, config):
       return cls(**config)


    # Create a model using the custom layer
    model = keras.Sequential([MyLayer(10),
                             keras.layers.Activation('softmax')])

    # The compile step specifies the training configuration
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # Trains for 5 epochs.
    model.fit(data, targets, batch_size=32, epochs=5)
    ```
## 回调 Callbacks
  - **callback** 传递给模型的参数，用于扩展训练过程中的行为
  - **tf.keras.callbacks** 中预定义的方法
    - **tf.keras.callbacks.ModelCheckpoint** 在指定的时间间隔保存模型数据 checkpoints
    - **tf.keras.callbacks.LearningRateScheduler** 动态调整学习率
    - **tf.keras.callbacks.EarlyStopping** 评估没有改变时终止训练过程
    - **tf.keras.callbacks.TensorBoard** 使用 TensorBoard 监控模型行为
    ```python
    callbacks = [
     # Interrupt training if `val_loss` stops improving for over 2 epochs
     keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
     # Write TensorBoard logs to `./logs` directory
     keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
             validation_data=(val_data, val_targets))
    ```
## 模型存储与加载
  - **save_weights / load_weights** 只保存权重 Weights only
    ```python
    # Save weights to a TensorFlow Checkpoint file
    model.save_weights('./my_model')

    # Restore the model's state,
    # this requires a model with the same architecture.
    model.load_weights('my_model')
    ```
    默认使用 TensorFlow checkpoint 格式保存，也可以使用 keras HDF5 格式保存
    ```python
    # Save weights to a HDF5 file
    model.save_weights('my_model.h5', save_format='h5')

    # Restore the model's state
    model.load_weights('my_model.h5')
    ```
  - **to_json / from_json** / **to_yaml / from_yaml** 只保存模型配置方式
    ```python
    # Serialize a model to JSON format
    json_string = model.to_json()

    # Recreate the model (freshly initialized)
    fresh_model = keras.models.from_json(json_string)

    # Serializes a model to YAML format
    yaml_string = model.to_yaml()

    # Recreate the model
    fresh_model = keras.models.from_yaml(yaml_string)
    ```
    - 继承类的模型不会被序列化保存
  - **save / load_model** 保存整个模型，包含权重 / 模型配置 / 优化器配置等
    ```python
    # Create a trivial model
    model = keras.Sequential([
     keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
     keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='rmsprop',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    model.fit(data, targets, batch_size=32, epochs=5)


    # Save entire model to a HDF5 file
    model.save('my_model.h5')

    # Recreate the exact same model, including weights and optimizer.
    model = keras.models.load_model('my_model.h5')
    ```
## Estimators
  - **tf.keras.estimator.model_to_estimator** 将 keras 模型转化为 `tf.estimator.Estimator`，之后可以使用 `tf.estimator` API
    ```python
    model = keras.Sequential([layers.Dense(10,activation='softmax'),
                             layers.Dense(10,activation='softmax')])

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    estimator = keras.estimator.model_to_estimator(model)
    ```
## 多 GPU 运行
  - **tf.contrib.distribute.DistributionStrategy** 指定使用多个 GPU 策略
  - **tf.contrib.distribute.MirroredStrategy** 是目前唯一支持的分配策略，MirroredStrategy does in-graph replication with synchronous training using all-reduce on a single machine
  - **定义 keras 模型**
    ```python
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(10,)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.train.GradientDescentOptimizer(0.2)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    model.summary()
    ```
  - **定义输入 pipeline** 使用 `tf.data.Dataset` 分发输入数据
    ```python
    def input_fn():
        x = np.random.random((1024, 10))
        y = np.random.randint(2, size=(1024, 1))
        x = tf.cast(x, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.repeat(10)
        dataset = dataset.batch(32)
        return dataset
    ```
  - **定义 tf.estimator.RunConfig** 指定使用 `tf.contrib.distribute.MirroredStrategy`，可以指定使用的 GPU 数量等，默认使用全部
    ```python
    strategy = tf.contrib.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
    ```
  - **转为 keras 模型为 tf.estimator.Estimator** 指定使用创建好的 `RunConfig`
    ```python
    keras_estimator = keras.estimator.model_to_estimator(
        keras_model=model,
        config=config,
        model_dir='/tmp/model_dir')
    ```
  - **训练模型**
    ```python
    keras_estimator.train(input_fn=input_fn, steps=10)
    ```
***

# Eager Execution
## 基本使用
  - **Eager** tensorflow 的动态图机制，直接返回执行结果，不同于传统的 tensorflow 构建静态图方法，构建类似 pytorch 的动态图
  - **tf.enable_eager_execution** 初始化 **Eager** 执行环境
    ```python
    tf.enable_eager_execution()
    tf.executing_eagerly()
    # Out[12]: True

    x = [[2.]]
    m = tf.matmul(x, x)
    m.numpy()
    # Out[13]: array([[4.]], dtype=float32)
    ```
  - **numpy** Eager 执行环境可以很好地配合 Numpy 使用，`tf.Tensor.numpy` 返回 `ndarray` 值
    ```py
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.add(a, 1)
    (a * b).numpy()
    # Out[219]: array([[ 2,  6], [12, 20]], dtype=int32)

    np.matmul(a, b)
    # Out[221]: array([[10, 13], [22, 29]], dtype=int32)
    ```
  - **tf.convert_to_tensor** 转化为 tensor
    ```python
    aa = tf.convert_to_tensor([1, 2, 3])
    aa.numpy()
    # Out[234]: array([1, 2, 3], dtype=int32)
    ```
  - **tf.contrib.eager** 包含一些同时支持 eager / graph 的操作，如 `Variable`
    ```py
    tfe = tf.contrib.eager
    # Or
    import tensorflow.contrib.eager as tfe

    # Eager 环境下使用 tf.Variable 将报错
    w = tf.Variable(10) # RuntimeError: tf.Variable not supported when eager execution is enabled

    # 使用 tfe.Variable
    w = tfe.Variable(10)
    ```
## 计算梯度 GradientTape
  - **tf.GradientTape** 可以提高计算性能，将前向传播的计算记录到 `tape` 上，并在训练结束后可以回放梯度计算过程，只可回放一次
    ```py
    w = tf.Variable([[1.0]])
    with tf.GradientTape() as tape:
      loss = w * w

    tape.gradient(loss, w).numpy()
    # Out[11]: array([[2.]], dtype=float32)

    # 再次调用将出错 RuntimeError
    grad = tape.gradient(loss, w)
    # RuntimeError: GradientTape.gradient can only be called once on non-persistent tapes.
    ```
    ```py
    dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                                  data.train.labels))
    ...
    for (batch, (images, labels)) in enumerate(dataset):
      ...
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss(logits, labels)
      ...
      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(zip(grads, model.variables),
                                global_step=tf.train.get_or_create_global_step())
    ```
  - **tf.GradientTape** 用于模型训练示例
    ```py
    # A toy dataset of points around 3 * x + 2
    NUM_EXAMPLES = 1000
    training_inputs = tf.random_normal([NUM_EXAMPLES])
    noise = tf.random_normal([NUM_EXAMPLES])
    training_outputs = training_inputs * 3 + 2 + noise

    def prediction(input, weight, bias):
      return input * weight + bias

    # A loss function using mean-squared error
    def loss(weights, biases):
      error = prediction(training_inputs, weights, biases) - training_outputs
      return tf.reduce_mean(tf.square(error))

    # Return the derivative of loss with respect to weight and bias
    def grad(weights, biases):
      with tf.GradientTape() as tape:
        loss_value = loss(weights, biases)
      return tape.gradient(loss_value, [weights, biases])

    train_steps = 200
    learning_rate = 0.01
    # Start with arbitrary values for W and B on the same batch of data
    W = tfe.Variable(5.)
    B = tfe.Variable(10.)

    print("Initial loss: {:.3f}".format(loss(W, B)))
    # Initial loss: 70.372

    for i in range(train_steps):
      dW, dB = grad(W, B)
      W.assign_sub(dW * learning_rate)
      B.assign_sub(dB * learning_rate)
      if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

    print("Finale loss: {:.3f}".format(loss(W, B)))
    print("W = {}, B = {}".format(W.numpy(), B.numpy()))
    ```
    **运行结果**
    ```py
    Initial loss: 70.372
    Loss at step 000: 67.532
    Loss at step 020: 29.866
    Loss at step 040: 13.536
    Loss at step 060: 6.436
    Loss at step 080: 3.339
    Loss at step 100: 1.986
    Loss at step 120: 1.392
    Loss at step 140: 1.131
    Loss at step 160: 1.016
    Loss at step 180: 0.965
    Finale loss: 0.944
    W = 3.035646915435791, B = 2.159491539001465
    ```
  - 定义模型继承 `tf.keras.Model`，并使用 `tf.GradientTape` 封装 `tf.Variable`
    ```py
    class Model(tf.keras.Model):
      def __init__(self):
        super(Model, self).__init__()
        self.W = tfe.Variable(5., name="weight")
        self.B = tfe.Variable(5., name="bias")

      def call(self, inputs):
        return inputs * self.W + self.B

    # The loss function to be optimized
    def loss(model, inputs, targets):
      error = model(inputs) - targets
      return tf.reduce_mean(tf.square(error))

    def grad(model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
      return tape.gradient(loss_value, [model.W, model.B])

    # Define:
    # 1. A model.
    # 2. Derivatives of a loss function with respect to model parameters.
    # 3. A strategy for updating the variables based on the derivatives.
    model = Model()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

    for i in range(300):
      grads = grad(model, training_inputs, training_outputs)
      optimizer.apply_gradients(zip(grads, [model.W, model.B]),
              global_step=tf.train.get_or_create_global_step())
      if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

    print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
    print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
    ```
## 模型训练 MNIST
  - **建立模型** 使用 `tf.keras.Sequential`
    ```py
    tf.enable_eager_execution()

    # Conv2D 值支持 NHWC 格式数据，即 'channels_last'，对应的是 'channels_first'
    data_format = 'channels_last'
    input_shape = [1, 28, 28]
    model = tf.keras.Sequential( [
        tf.keras.layers.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
        tf.keras.layers.Conv2D(32, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format),
        tf.keras.layers.Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10)
    ])

    batch = tf.zeros([1, 1, 784])
    model(batch).numpy()
    # Out[145]: array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)
    ```
  - **加载数据** 使用 `keras.datasets.mnist`
    ```python
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.int64)
    dataset_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset_train = dataset_train.shuffle(60000).repeat(4).batch(32)

    x, y = iter(dataset_train).next()
    x.numpy().shape
    # Out[150]: (32, 28, 28)

    y.numpy().shape
    # Out[151]: (32,)

    model(x).numpy().shape
    # Out[156]: (32, 10)
    ```
  - **损失函数与梯度** 使用 `GradientTape`
    ```py
    def loss(model, x, y):
      prediction = model(x)
      return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

    def grad(model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
      return tape.gradient(loss_value, model.variables)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    x, y = iter(dataset_train).next()
    print("Initial loss: {:.3f}".format(loss(model, x, y)))
    # Initial loss: 17.040
    ```
  - **模型训练**
    ```py
    for (i, (x, y)) in enumerate(dataset_train):
      # Calculate derivatives of the input function with respect to its parameters.
      grads = grad(model, x, y)
      # Apply the gradient to the model
      optimizer.apply_gradients(zip(grads, model.variables),
                                global_step=tf.train.get_or_create_global_step())
      if i % 200 == 0:
        print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

    print("Final loss: {:.3f}".format(loss(model, x, y)))
    ```
  - **运行结果**
    ```py
    Loss at step 0000: 7.151
    Loss at step 0200: 0.248
    Loss at step 0400: 0.235
    Loss at step 0600: 0.081
    ...
    Loss at step 7200: 0.003
    Loss at step 7400: 0.025
    Final loss: 0.005
    ```
  - **指定 GPU 训练**
    ```py
    with tf.device("/gpu:0"):
      for (i, (x, y)) in enumerate(dataset_train):
        # minimize() is equivalent to the grad() and apply_gradients() calls.
        optimizer.minimize(lambda: loss(model, x, y), global_step=tf.train.get_or_create_global_step())
    ```
## 保存加载模型 Checkpoint
  - **tf.train.Checkpoint** 用于保存变量到 checkpoints
    ```py
    x = tfe.Variable(10.)
    checkpoint = tf.train.Checkpoint(x=x)  # save as "x"

    x.assign(2.)   # Assign a new value to the variables and save.
    save_path = checkpoint.save('./ckpt/')

    x.assign(11.)  # Change the variable after saving.

    # Restore values from the checkpoint
    checkpoint.restore(save_path)
    print(x.numpy())
    # 2.0
    ```
  - **tf.train.Checkpoint** 保存 / 加载模型
    ```py
    model = Model()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    checkpoint_dir = './ckpt/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())

    root.save(file_prefix=checkpoint_prefix)
    # or
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))
    ```
## 度量方式与汇总 metrics and summary
  - **tfe.metrics** 直接调用传递新值可以更新 metrics，查看结果使用 `tfe.metrics.result`
    ```py
    m = tfe.metrics.Mean("loss")
    m(0)
    m(5)
    m.result().numpy()
    # Out[58]: 2.5

    m([8, 9])
    m.result().numpy()
    # Out[60]: 5.5
    ```
  - **tf.contrib.summary** 可以记录数据用于 TensorBoard
    ```py
    global_step = tf.train.get_or_create_global_step()
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    for _ in range(iterations):
      global_step.assign_add(1)
      # Must include a record_summaries method
      with tf.contrib.summary.record_summaries_every_n_global_steps(100):
        # your model code goes here
        tf.contrib.summary.scalar('loss', loss)
         ...
    ```
## 自动梯度计算 automatic differentiation
  - **Dynamic models** `tf.GradientTape` 可以用在动态模式中，示例算法 [backtracking line search](https://wikipedia.org/wiki/Backtracking_line_search)
    ```py
    def line_search_step(fn, init_x, rate=1.0):
      with tf.GradientTape() as tape:
        # Variables are automatically recorded, but manually watch a tensor
        tape.watch(init_x)
        value = fn(init_x)
      grad = tape.gradient(value, init_x)
      grad_norm = tf.reduce_sum(grad * grad)
      init_value = value
      while value > init_value - rate * grad_norm:
        x = init_x - rate * grad
        value = fn(x)
        rate /= 2.0
      return x, value
    ```
  - **tfe.gradients_function** 返回一个函数，根据输入函数的功能，计算相应的导数，**参数值一定要用浮点数**
    ```py
    def square(x):
      return tf.multiply(x, x)

    # df / dx = 2 * x
    grad = tfe.gradients_function(square)

    square(3.)  # => 9.0
    # 参数值一定要用浮点数
    grad(3.)    # => [6.0]

    # The second-order derivative of square
    # df / dx = 2
    gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
    gradgrad(3.)  # => [2.0]

    # The third-order derivative is None
    # df / dx = 0
    gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
    gradgradgrad(3.)  # => [None]
    ```
    ```py
    # With flow control:
    def abs(x):
      return x if x > 0. else -x

    grad = tfe.gradients_function(abs)

    grad(3.)   # => [1.0]
    grad(-3.)  # => [-1.0]
    ```
  - **tfe.value_and_gradients_function** 类似 `tfe.gradients_function`, 同时返回函数值与导数值，**参数值一定要用浮点数**
    ```py
    gg = tfe.value_and_gradients_function(square)
    aa, bb = gg(3.)
    aa.numpy()
    # Out[131]: 9.0

    bb[0].numpy()
    # Out[133]: 6.0
    ```
    ```python
    # f(x, y) = (x ^ 3) * y - x * (y ^ 2)
    # df / dx = 3 * (x ^ 2) * y - y ^ 2
    # df / dy = x ^ 3 - 2 * x * y
    def f(x, y):
      return x * x * x * y - x * y * y

    # Obtain a function that returns the function value and the 1st order gradients.
    val_grads_fn = tfe.value_and_gradients_function(f)
    # Invoke the value-and-gradients function.
    f_val, (x_grad, y_grad) = val_grads_fn(2.0, 3.0)
    assert f_val.numpy() == (2 ** 3) * 3 - 2 * (3 ** 2)
    assert x_grad.numpy() == 3 * (2 ** 2) * 3 - 3 ** 2
    assert y_grad.numpy() == (2 ** 3) - 2 * 2 * 3
    ```
## 自定义梯度计算 Custom gradients
  - **自定义梯度计算** 根据 输入 / 输出 / 中间结果 重新定义梯度计算方式，如使用 `clip_by_norm` 限定梯度范围
    ```py
    @tf.custom_gradient
    def clip_gradient_by_norm(x, norm):
      y = tf.identity(x)
      def grad_fn(dresult):
        return [tf.clip_by_norm(dresult, norm), None]
      return y, grad_fn
    ```
  - **自定义梯度计算** 通常用于为算法提供更稳定的梯度结果
    ```py
    def log1pexp(x):
      return tf.log(1 + tf.exp(x))
    grad_log1pexp = tfe.gradients_function(log1pexp)

    # The gradient computation works fine at x = 0.
    grad_log1pexp(0.)  # => [0.5]

    # However, x = 100 fails because of numerical instability.
    grad_log1pexp(100.)  # => [nan]
    ```
## 性能 Performance
  - 计算过程会自动使用 GPU，也可以用过 `tf.device('/gpu:0')` 指定使用的设备
    ```py
    import time

    def measure(x, steps):
      # TensorFlow initializes a GPU the first time it's used, exclude from timing.
      tf.matmul(x, x)
      start = time.time()
      for i in range(steps):
        x = tf.matmul(x, x)
      # tf.matmul can return before completing the matrix multiplication
      # (e.g., can return after enqueing the operation on a CUDA stream).
      # The x.numpy() call below will ensure that all enqueued operations
      # have completed (and will also copy the result to host memory,
      # so we're including a little more than just the matmul operation
      # time).
      _ = x.numpy()
      end = time.time()
      return end - start

    shape = (1000, 1000)
    steps = 200
    print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

    # Run on CPU:
    with tf.device("/cpu:0"):
      print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

    # Run on GPU, if available:
    if tfe.num_gpus() > 0:
      with tf.device("/gpu:0"):
        print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
    else:
      print("GPU: not found")
    ```
    **运行结果**
    ```py
    Time to multiply a (1000, 1000) matrix by itself 200 times:
    CPU: 2.79622220993042 secs
    GPU: 0.39931392669677734 secs
    ```
  - **gpu / cpu** 指定 `tf.Tensor` 使用不同的设备
    ```py
    x = tf.random_normal([10, 10])

    x_gpu0 = x.gpu()
    x_cpu = x.cpu()

    _ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
    _ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

    if tfe.num_gpus() > 1:
      x_gpu1 = x.gpu(1)
      _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1
    ```
## Graph 运行环境中使用 eager execution
  - **tfe.py_func** 用于在 graph 运行环境中使用 eager execution 调用
    ```py
    tfe = tf.contrib.eager

    def my_py_func(x):
      x = tf.matmul(x, x)  # You can use tf ops
      print(x)  # but it's eager!
      return x

    with tf.Session() as sess:
      x = tf.placeholder(dtype=tf.float32)
      # Call eager function in graph!
      pf = tfe.py_func(my_py_func, [x], tf.float32)
      sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
    ```
***

# Datasets Importing Data
## data API
  - **tf.data** 创建输入 pipelines，引入了两个重要的概念 `tf.data.Dataset` / `tf.data.Iterator`
  - **tf.data.Dataset** 代表元素序列，每个元素可能包含多个 `tf.Tensor`
    - **数据源 source** 从一个或多个 `tf.Tensor` 创建出一个 dataset，如使用 `Dataset.from_tensor_slices`
    - **转化 transformation** 从一个或多个 `tf.data.Dataset` 组合出一个 dataset，如使用 `Dataset.batch`
  - **tf.data.Iterator** 从 dataset 中提取数据，如 `Dataset.make_one_shot_iterator`
    - **Iterator.get_next** 用于获取下一个数据
    - **Iterator.initializer** 重新初始化迭代器，并且可以指向其他数据源
  - **基本过程**
    - 首先定义一个数据源，可以使用 `tf.data.Dataset.from_tensors()` / `tf.data.Dataset.from_tensor_slices()` 从内存中数据初始化，或使用 `tf.data.TFRecordDataset` 从硬盘上存储的 `TFRecord` 格式文件中初始化
    - 然后可以转化为一个新的 `Dataset`，如 `Dataset.map()` 将转化应用到每个元素上，`Dataset.batch()` 将元素组合成一个 `batch`
    - 之后可以使用迭代器 `tf.data.Iterator` 遍历数据，如调用 `Dataset.make_one_shot_iterator()` 创建 one-shot 迭代器
## Dataset 结构
  - **Dataset.output_types** / **Dataset.output_shapes** 表示内部元素的数据类型 / 数据维度
    ```python
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    print(dataset1.output_types)  # ==> "tf.float32"
    print(dataset1.output_shapes)  # ==> "(10,)"

    dataset2 = tf.data.Dataset.from_tensor_slices(
       (tf.random_uniform([4]),
        tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
    print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
    print(dataset2.output_shapes)  # ==> "((), (100,))"

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
    ```
  - **字典命名** 可以给 Dataset 中不同的部分指定名称
    ```python
    dataset = tf.data.Dataset.from_tensor_slices(
       {"a": tf.random_uniform([4]),
        "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
    print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
    print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
    ```
  - **Dataset.map()** / **Dataset.flat_map()** / **Dataset.filter()** 将一个转化函数应用到每一个元素上
    ```python
    dataset1 = dataset1.map(lambda x: ...)

    dataset2 = dataset2.flat_map(lambda x, y: ...)

    # Note: Argument destructuring is not available in Python 3.
    dataset3 = dataset3.filter(lambda x, (y, z): ...)
    ```
## 创建迭代器 Iterator
  - **Dataset.make_one_shot_iterator** one-shot 迭代器，只遍历一遍 dataset，目前是唯一可以很方便用到 `Estimator` 上的
    ```python
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
      for i in range(100):
        value = sess.run(next_element)
        assert i == value
    ```
  - **Dataset.make_initializable_iterator** initializable 迭代器，要求每次调用前使用 `iterator.initializer` 进行初始化，可以每次通过配合使用 `tf.placeholder` 指定不同的数据源
    ```python
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Initialize an iterator over a dataset with 10 elements.
    with tf.Session() as sess:
      sess.run(iterator.initializer, feed_dict={max_value: 10})
      for i in range(10):
        value = sess.run(next_element)
        assert i == value

    # Initialize the same iterator over a dataset with 100 elements.
    with tf.Session() as sess:
      sess.run(iterator.initializer, feed_dict={max_value: 100})
      for i in range(100):
        value = sess.run(next_element)
        assert i == value
    ```
  - **Iterator.from_structure** / **Iterator.make_initializer** reinitializable 迭代器，可以初始化成指向多个数据源，如使用一个迭代器指向 训练 / 验证数据集
    ```python
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    with tf.Session() as sess:
      for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
          sess.run(next_element)

        # Initialize an iterator over the validation dataset.
        sess.run(validation_init_op)
        for _ in range(50):
          sess.run(next_element)
    ```
  - **Iterator.from_string_handle** feedable 迭代器，功能类似 `reinitializable` 迭代器，但在不同数据源切换时，不需要从数据开始重新初始化迭代器
    ```python
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
    validation_dataset = tf.data.Dataset.range(50)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
    next_element = iterator.get_next()

    # You can use feedable iterators with a variety of different kinds of iterator
    # (such as one-shot and initializable iterators).
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    with tf.Session() as sess:
      training_handle = sess.run(training_iterator.string_handle())
      validation_handle = sess.run(validation_iterator.string_handle())

      # Loop forever, alternating between training and validation.
      while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
          sess.run(next_element, feed_dict={handle: training_handle})

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
          sess.run(next_element, feed_dict={handle: validation_handle})
    ```
## 从迭代器中读取数据
  - **Iterator.get_next** 从迭代器中获取一个数据元素，到达结尾时抛出 `tf.errors.OutOfRangeError` 异常，如果要继续使用，需要重新初始化
    ```python
    dataset = tf.data.Dataset.range(5)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Typically `result` will be the output of a model, or an optimizer's
    # training operation.
    result = tf.add(next_element, next_element)
    sess = tf.InteractiveSession()

    sess.run(iterator.initializer)
    print(sess.run(result))  # ==> "0"
    print(sess.run(result))  # ==> "2"
    print(sess.run(result))  # ==> "4"
    print(sess.run(result))  # ==> "6"
    print(sess.run(result))  # ==> "8"

    try:
      sess.run(result)
    except tf.errors.OutOfRangeError:
      print("End of dataset")  # ==> "End of dataset"
    ```
  - **嵌套结构的数据元素 nested structure dataset**
    ```python
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

    iterator = dataset3.make_initializable_iterator()

    sess.run(iterator.initializer)
    next1, (next2, next3) = iterator.get_next()

    print(next1.eval())
    print(next2.eval())
    print(next3.eval())
    ```
## 保存迭代器状态
  - **tf.contrib.data.make_saveable_from_iterator**
    - 从迭代器创建一个 `SaveableObject`，用于保存 / 加载迭代器的当前状态，或者是保存整个输入 pipeline
    - 创建出的 `SaveableObject` 可以加到 `tf.train.Saver` 的变量列表中，或 `tf.GraphKeys.SAVEABLE_OBJECTS` 集合中，用于类似 `tf.Variable` 的保存 / 加载方式
    ```python
    # Create saveable object from iterator.
    saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

    # Save the iterator state by adding it to the saveable objects collection.
    tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
    saver = tf.train.Saver()

    with tf.Session() as sess:

      if should_checkpoint:
        saver.save(path_to_checkpoint)

    # Restore the iterator state.
    with tf.Session() as sess:
      saver.restore(sess, path_to_checkpoint)
    ```
## 读取输入数据 NumPy arrays
  - **Dataset.from_tensor_slices** 从内存数据中创建 dataset
    ```python
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load("~/.keras/datasets/mnist.npz") as data:
      features = data["x_train"]
      labels = data["y_train"]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    ```
  - **tf.placeholder**
    - 直接使用 features / labels 数据创建 dataset，将会使用 `tf.constant()` 创建数据，并消耗大量内存
    - 可以使用 `tf.placeholder()` 定义 datset，并在初始化迭代器的时候指定 Numpy 数据
    ```python
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    with np.load("~/.keras/datasets/mnist.npz") as data:
      features = data["x_train"]
      labels = data["y_train"]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    # [Other transformations on `dataset`...]
    dataset = ...
    iterator = dataset.make_initializable_iterator()

    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                              labels_placeholder: labels})
    ```
## 读取输入数据 TFRecord data
  - **tf.data.TFRecordDataset** 可以将一个或多个 TFRecord 文件组合成输入 pipeline
    ```python
    # Creates a dataset that reads all of the examples from two files.
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    ```
    其中 `filenames` 参数可以是 一个字符串 / 字符串列表 / `tf.Tensor` 形式的字符串
    ```python
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(...)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()

    # You can feed the initializer with the appropriate filenames for the current
    # phase of execution, e.g. training vs. validation.

    # Initialize `iterator` with training data.
    training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

    # Initialize `iterator` with validation data.
    validation_filenames = ["/var/data/validation1.tfrecord", ...]
    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
    ```
## 读取输入数据 text data
  - **tf.data.TextLineDataset** 从一个或多个文本文件中解析文本行，类似 `TFRecordDataset`，同样支持使用 `tf.placeholder(tf.string)` 参数化文件名
    ```python
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
    dataset = tf.data.TextLineDataset(filenames)
    ```
  - **Dataset.flat_map** 在每个文件上应用一个转化函数，然后重新组合成一个 dataset，`Dataset.skip()` 用于跳过文件开头几行，`Dataset.filter()` 过滤行
    ```python
    filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
            .skip(1)
            .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
    ```
## 读取输入数据 CSV data
  - **tf.contrib.data.CsvDataset** 从一个或多个 csv 文件中解析数据，支持使用 `tf.placeholder(tf.string)` 参数化文件名
    ```py
    # Creates a dataset that reads all of the records from two CSV files, each with
    # eight float columns
    filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
    record_defaults = [tf.float32] * 8   # Eight required float columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
    ```
    对于有空列存在的数据，也可以提供默认值
    ```py
    # Creates a dataset that reads all of the records from two CSV files, each with
    # four float columns which may have missing values
    record_defaults = [[0.0]] * 8
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
    ```
    **header** / **select_cols** 参数指定过滤条件
    ```py
    # Creates a dataset that reads all of the records from two CSV files with
    # headers, extracting float data from columns 2 and 4.
    record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2,4])
    ```
## map 转化数据
  - **Dataset.map(f)** 在一个 dataset 上应用转化函数 `f`，并返回新的 dataset
  - **转化 tf.train.Example Protocol Buffer 数据**，TFRecord 使用的数据格式，每条记录包含一个或多个 features
    ```python
    # Transforms a scalar string `example_proto` into a pair of a scalar string and
    # a scalar integer, representing an image and its label, respectively.
    def _parse_function(example_proto):
      features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                  "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
      parsed_features = tf.parse_single_example(example_proto, features)
      return parsed_features["image"], parsed_features["label"]

    # Creates a dataset that reads all of the examples from two files, and extracts
    # the image and label features.
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    ```
  - **转化图片数据**，读取并转化为同样大小
    ```python
    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def _parse_function(filename, label):
      image_string = tf.read_file(filename)
      image_decoded = tf.image.decode_jpeg(image_string)
      image_resized = tf.image.resize_images(image_decoded, [28, 28])
      return image_resized, label

    # A vector of filenames.
    filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant([0, 37, ...])

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    ```
  - **tf.py_func** 调用任意的 python 函数转化数据
    ```python
    import cv2

    # Use a custom OpenCV function to read the image, instead of the standard
    # TensorFlow `tf.read_file()` operation.
    def _read_py_function(filename, label):
      image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
      return image_decoded, label

    # Use standard TensorFlow operations to resize the image to a fixed shape.
    def _resize_function(image_decoded, label):
      image_decoded.set_shape([None, None, None])
      image_resized = tf.image.resize_images(image_decoded, [28, 28])
      return image_resized, label

    filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
    labels = [0, 37, 29, 1, ...]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)
    ```
## 打包 Batching
  - **Dataset.batch** 类似 `tf.stack()` 的操作，将 `n` 个连续的元素组合成一个元素，要求所有元素有相同的维度
    ```python
    inc_dataset = tf.data.Dataset.range(100)
    dec_dataset = tf.data.Dataset.range(0, -100, -1)
    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    batched_dataset = dataset.batch(4)

    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()
    print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
    ```
  - **Dataset.padded_batch** 支持组合维度不同的元素，如单词向量等，可以指定一个或多个用于组合的维度
    ```python
    dataset = tf.data.Dataset.range(100)

    # 创建矩阵 tf.fill([2, 3], 3).eval() --> array([[3, 3, 3], [3, 3, 3]], dtype=int32)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    dataset = dataset.padded_batch(4, padded_shapes=[None])

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                                   #      [5, 5, 5, 5, 5, 0, 0],
                                   #      [6, 6, 6, 6, 6, 6, 0],
                                   #      [7, 7, 7, 7, 7, 7, 7]]
    ```
## Dataset 多次迭代
  - **Dataset.repeat** 可以指定数据集迭代次数，如果不指定参数，将无限重复输入数据
    ```python
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(...)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(32)
    ```
  - **tf.errors.OutOfRangeError** 数据结束时抛出的异常，可以用于在每次输入数据遍历一遍时，做一些处理，再开始下一迭代，如可以检查验证数据集
    ```python
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(...)
    dataset = dataset.batch(32)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Compute for 100 epochs.
    for _ in range(100):
      sess.run(iterator.initializer)
      while True:
        try:
          sess.run(next_element)
        except tf.errors.OutOfRangeError:
          break

      # [Perform end-of-epoch calculations here.]
    ```
## 随机打乱数据 Randomly shuffling input data
  - **Dataset.shuffle** 使用类似 `tf.RandomShuffleQueue` 的算法随机打乱输入数据
    ```python
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(...)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    ```
## 在其他高级 APIs 中的使用
  - **tf.train.MonitoredTrainingSession** 可以简化 TensorFlow 在分布式环境中运行的很多方面，使用 `tf.errors.OutOfRangeError` 异常作为训练结束的标志
    ```python
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(...)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    next_example, next_label = iterator.get_next()
    loss = model_function(next_example, next_label)

    training_op = tf.train.AdagradOptimizer(...).minimize(loss)

    with tf.train.MonitoredTrainingSession(...) as sess:
      while not sess.should_stop():
        sess.run(training_op)
    ```
  - 在 `tf.estimator.Estimator` 的 `input_fn` 使用 `Dataset`，只需要在定义的输入函数中返回 dataset，框架自动创建迭代器并初始化
    ```python
    def dataset_input_fn():
      filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
      dataset = tf.data.TFRecordDataset(filenames)

      # Use `tf.parse_single_example()` to extract data from a `tf.Example`
      # protocol buffer, and perform any additional per-record preprocessing.
      def parser(record):
        keys_to_features = {
            "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
            "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image_data"])
        image = tf.reshape(image, [299, 299, 1])
        label = tf.cast(parsed["label"], tf.int32)

        return {"image_data": image, "date_time": parsed["date_time"]}, label

      # Use `Dataset.map()` to build a pair of a feature dictionary and a label
      # tensor for each example.
      dataset = dataset.map(parser)
      dataset = dataset.shuffle(buffer_size=10000)
      dataset = dataset.batch(32)
      dataset = dataset.repeat(num_epochs)

      # Each element of `dataset` is tuple containing a dictionary of features
      # (in which each value is a batch of values for that feature), and a batch of
      # labels.
      return dataset
    ```
***

# Estimators

This document introduces `tf.estimator`--a high-level TensorFlow
API that greatly simplifies machine learning programming. Estimators encapsulate
the following actions:

*   training
*   evaluation
*   prediction
*   export for serving

You may either use the pre-made Estimators we provide or write your
own custom Estimators.  All Estimators--whether pre-made or custom--are
classes based on the `tf.estimator.Estimator` class.

For a quick example try [Estimator tutorials]](../tutorials/estimators/linear).
To see each sub-topic in depth, see the [Estimator guides](premade_estimators).

Note: TensorFlow also includes a deprecated `Estimator` class at
`tf.contrib.learn.Estimator`, which you should not use.


## Advantages of Estimators

Estimators provide the following benefits:

*   You can run Estimator-based models on a local host or on a
    distributed multi-server environment without changing your model.
    Furthermore, you can run Estimator-based models on CPUs, GPUs,
    or TPUs without recoding your model.
*   Estimators simplify sharing implementations between model developers.
*   You can develop a state of the art model with high-level intuitive code.
    In short, it is generally much easier to create models with Estimators
    than with the low-level TensorFlow APIs.
*   Estimators are themselves built on `tf.keras.layers`, which
    simplifies customization.
*   Estimators build the graph for you.
*   Estimators provide a safe distributed training loop that controls how and
    when to:
    *   build the graph
    *   initialize variables
    *   load data
    *   handle exceptions
    *   create checkpoint files and recover from failures
    *   save summaries for TensorBoard

When writing an application with Estimators, you must separate the data input
pipeline from the model.  This separation simplifies experiments with
different data sets.


## Pre-made Estimators

Pre-made Estimators enable you to work at a much higher conceptual level
than the base TensorFlow APIs. You no longer have to worry about creating
the computational graph or sessions since Estimators handle all
the "plumbing" for you.  That is, pre-made Estimators create and manage
`tf.Graph` and `tf.Session` objects for you.  Furthermore,
pre-made Estimators let you experiment with different model architectures by
making only minimal code changes.  `tf.estimator.DNNClassifier`,
for example, is a pre-made Estimator class that trains classification models
based on dense, feed-forward neural networks.


### Structure of a pre-made Estimators program

A TensorFlow program relying on a pre-made Estimator typically consists
of the following four steps:

1.  **Write one or more dataset importing functions.** For example, you might
    create one function to import the training set and another function to
    import the test set. Each dataset importing function must return two
    objects:

    *   a dictionary in which the keys are feature names and the
        values are Tensors (or SparseTensors) containing the corresponding
        feature data
    *   a Tensor containing one or more labels

    For example, the following code illustrates the basic skeleton for
    an input function:

        def input_fn(dataset):
           ...  # manipulate dataset, extracting the feature dict and the label
           return feature_dict, label

    (See [Importing Data](../guide/datasets.md) for full details.)

2.  **Define the feature columns.** Each `tf.feature_column`
    identifies a feature name, its type, and any input pre-processing.
    For example, the following snippet creates three feature
    columns that hold integer or floating-point data.  The first two
    feature columns simply identify the feature's name and type. The
    third feature column also specifies a lambda the program will invoke
    to scale the raw data:

        # Define three numeric feature columns.
        population = tf.feature_column.numeric_column('population')
        crime_rate = tf.feature_column.numeric_column('crime_rate')
        median_education = tf.feature_column.numeric_column('median_education',
                            normalizer_fn=lambda x: x - global_education_mean)

3.  **Instantiate the relevant pre-made Estimator.**  For example, here's
    a sample instantiation of a pre-made Estimator named `LinearClassifier`:

        # Instantiate an estimator, passing the feature columns.
        estimator = tf.estimator.LinearClassifier(
            feature_columns=[population, crime_rate, median_education],
            )

4.  **Call a training, evaluation, or inference method.**
    For example, all Estimators provide a `train` method, which trains a model.

        # my_training_set is the function created in Step 1
        estimator.train(input_fn=my_training_set, steps=2000)


### Benefits of pre-made Estimators

Pre-made Estimators encode best practices, providing the following benefits:

*   Best practices for determining where different parts of the computational
    graph should run, implementing strategies on a single machine or on a
    cluster.
*   Best practices for event (summary) writing and universally useful
    summaries.

If you don't use pre-made Estimators, you must implement the preceding
features yourself.


## Custom Estimators

The heart of every Estimator--whether pre-made or custom--is its
**model function**, which is a method that builds graphs for training,
evaluation, and prediction. When you are using a pre-made Estimator,
someone else has already implemented the model function. When relying
on a custom Estimator, you must write the model function yourself. A
[companion document](../guide/custom_estimators.md)
explains how to write the model function.


## Recommended workflow

We recommend the following workflow:

1.  Assuming a suitable pre-made Estimator exists, use it to build your
    first model and use its results to establish a baseline.
2.  Build and test your overall pipeline, including the integrity and
    reliability of your data with this pre-made Estimator.
3.  If suitable alternative pre-made Estimators are available, run
    experiments to determine which pre-made Estimator produces the
    best results.
4.  Possibly, further improve your model by building your own custom Estimator.


## Creating Estimators from Keras models

You can convert existing Keras models to Estimators. Doing so enables your Keras
model to access Estimator's strengths, such as distributed training. Call
`tf.keras.estimator.model_to_estimator` as in the
following sample:

```python
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```
Note that the names of feature columns and labels of a keras estimator come from
the corresponding compiled keras model. For example, the input key names for
`train_input_fn` above can be obtained from `keras_inception_v3.input_names`,
and similarly, the predicted output names can be obtained from
`keras_inception_v3.output_names`.

For more details, please refer to the documentation for
`tf.keras.estimator.model_to_estimator`.
