# ___2018 - 09 - 06 Tensorflow Tutorials___
***

- [Tensorflow Tutorials](https://www.tensorflow.org/tutorials/)
# 目录
***

# Learn and use ML
## 基本分类模型 Fasion MNIST 数据集
  - **import**
    ```py
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import matplotlib.pyplot as plt

    print(tf.__version__) # 1.10.1
    ```
  - **Fashion MNIST dataset** 类似 MNIST 的数据集，包含 10 中类别的流行物品图片，每个样本包含一个 `28 * 28` 的图片
    ```py
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images.shape
    # Out[20]: (60000, 28, 28)

    train_labels.shape
    # Out[21]: (60000,)

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    ```
    ![](images/tensorflow_mnist_fashion.png)
  - **keras 模型训练 / 验证**
    ```py
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    # Epoch 1/5
    # 60000/60000 [==============================] - 17s 280us/step - loss: 0.4958 - acc: 0.8248
    # Epoch 2/5
    # 60000/60000 [==============================] - 16s 269us/step - loss: 0.3766 - acc: 0.8640
    # Epoch 3/5
    # 60000/60000 [==============================] - 16s 275us/step - loss: 0.3366 - acc: 0.8748
    # Epoch 4/5
    # 60000/60000 [==============================] - 16s 275us/step - loss: 0.3129 - acc: 0.8851
    # Epoch 5/5
    # 60000/60000 [==============================] - 18s 303us/step - loss: 0.2952 - acc: 0.8914

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test loss = {}, Test accuracy = {}".format(test_loss, test_acc))
    # Test loss = 0.3613173280715942, Test accuracy = 0.868
    ```
  - **模型预测**
    ```py
    predictions = model.predict(test_images)
    predictions.shape
    # Out[39]: (10000, 10)
    assert np.argmax(predictions[0]) == test_labels[0]
    ```
    ```py
    def plot_image(i, predictions_array, true_label, img):
      predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])

      plt.imshow(img, cmap=plt.cm.binary)

      predicted_label = np.argmax(predictions_array)
      if predicted_label == true_label:
        color = 'blue'
      else:
        color = 'red'

      plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

    def plot_value_array(i, predictions_array, true_label):
      predictions_array, true_label = predictions_array[i], true_label[i]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      thisplot = plt.bar(range(10), predictions_array, color="#777777")
      plt.ylim([0, 1])
      predicted_label = np.argmax(predictions_array)

      thisplot[predicted_label].set_color('red')
      thisplot[true_label].set_color('blue')

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions, test_labels, test_images)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions, test_labels)
    ```
    ![](images/tensorflow_mnist_fashion_predict.png)
## 文本分类 IMDB 评论数据集
  - **pad_sequences** 将序列中的元素长度整理成相同长度
    ```py
    pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
    ```
    - 元素长度 **小于** 指定长度 `maxlen`，使用 `value` 填充
    - 元素长度 **大于** 最大长度 `maxlen`，按照 `padding` 与 `truncating` 指定的方式截取
    - **maxlen 参数** 指定整理后的最大长度，`None` 表示取序列中的最大长度
    - **dtype 参数** 输出元素类型
    - **padding 参数** 填充方式，字符串 `pre` / `post`，指定在结尾还是开头填充
    - **truncating 参数** 截取方式，字符串 `pre` / `post`，指定在结尾还是开头截取
    - **value 参数** Float，填充值
  - **加载 IMDB 电影评论数据集** 包含 50,000 条电影评论，以及是正面评论 positive / 负面评论 negative 的标签，划分成 25,000 条训练数据，25,000 条测试数据
    ```py
    from tensorflow import keras

    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    print(train_data.shape) # (25000,)
    print(len(train_data[0])) # 218
    train_data_len = np.array([len(ii) for ii in train_data])
    print(train_data_len.max()) # 2494
    print(train_data_len.min()) # 11
    print(train_data_len.argmin())  # 6719
    print(train_data[6719]) # [1, 13, 586, 851, 14, 31, 60, 23, 2863, 2364, 314]
    print(train_labels[:10])  # [1 0 0 1 0 0 1 0 1 0]

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = lambda text: ' '.join([reverse_word_index.get(ii, '?') for ii in text])

    print(decode_review(train_data[6719]))
    # <START> i wouldn't rent this one even on dollar rental night

    print(np.max([np.max(ii) for ii in train_data])) # 9999

    # use the pad_sequences function to standardize the lengths
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post', maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)
    print(train_data.shape, test_data.shape)  # (25000, 256) (25000, 256)
    ```
  - **keras 模型训练 / 验证 / 预测**
    - **Embedding layer** 将每个整数编码的单词转化成 embedding vector，输出维度 (batch, sequence, embedding)
    - **GlobalAveragePooling1D layer** 输出一个固定长度的向量，使模型可以处理可变长度的输入
    - **Dense layer 1** 全连接层，包含 16 个隐藏节点
    - **Dense layer 2** 输出层，输出 0-1 的概率值
    ```py
    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    # Layer (type)                 Output Shape              Param #   
    # =================================================================
    # embedding (Embedding)        (None, None, 16)          160000    
    # _________________________________________________________________
    # global_average_pooling1d (Gl (None, 16)                0         
    # _________________________________________________________________
    # dense (Dense)                (None, 16)                272       
    # _________________________________________________________________
    # dense_1 (Dense)              (None, 1)                 17        
    # =================================================================
    # Total params: 160,289
    # Trainable params: 160,289
    # Non-trainable params: 0
    # _________________________________________________________________

    # binary_crossentropy is better for dealing with probabilities
    model.compile(optimizer=tf.train.AdamOptimizer(),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    # Create a validation set
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    # Train the model
    history = model.fit(partial_x_train, partial_y_train,
        epochs=40, batch_size=512,
        validation_data=(x_val, y_val), verbose=1)

    # Evaluate the model
    result = model.evaluate(test_data, test_labels)
    print(result) # [0.3106797323703766, 0.87256]
    ```
  - **准确率与损失的可视化图形** `model.fit` 的返回值包含训练过程中的准确率 / 损失等
    ```py
    print(history.history.keys()) # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    fig.add_subplot(2, 1, 2)
    plt.plot(history.history['acc'], label='Training acc')
    plt.plot(history.history['val_acc'], label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    fig.tight_layout()
    ```
    ![](images/tensorflow_text_classifier.png)
## 回归预测 Boston 房价数据集
  - **EarlyStopping** 在指定的监控数据停止提高时，停止训练，可以防止过拟合
    ```py
    class EarlyStopping(Callback)
      __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
    ```
    - **monitor 参数** 监控的数据
    - **min_delta 参数** 最小改变的数值
    - **patience 参数** 没有提高的迭代次数
    - **verbose 参数** 冗余模式 verbosity mode
    - **mode 参数** `min` 指标不再下降 / `max` 指标不再上升 / `auto` 自动检测
    - **baseline 参数** 基准值 baseline value
  - **回归问题 regression**
    - 回归问题的预测结果是连续的
    - **MSE** Mean Squared Error 通常用于回归问题的损失函数
    - **MAE** Mean Absolute Error 通常用于回归问题的评估方法 metrics
    - 训练数据量过少时，尽量使用小的神经网络，防止过拟合
  - **加载 Boston 房价数据集** 包含 Boston 1970s 中期 的房价，以及房价可能相关的特征量
    ```py
    boston_housing = keras.datasets.boston_housing
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]

    print(train_data.shape, test_data.shape)  # (404, 13) (102, 13)
    print(train_data[0])  # [1.23247, 0., 8.14, 0., 0.538, 6.142, 91.7, 3.9769, 4., 307., 21., 396.9, 18.72]
    # The labels are the house prices in thousands of dollars
    print(train_labels[:10])  # [15.2 42.3 50.  21.1 17.7 18.5 11.3 15.6 15.6 14.4]

    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                    'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = DataFrame(train_data, columns=column_names)
    print(df.head())
    #       CRIM    ZN  INDUS  CHAS    NOX     RM    AGE     DIS   RAD    TAX  PTRATIO       B  LSTAT
    # 0  1.23247   0.0   8.14   0.0  0.538  6.142   91.7  3.9769   4.0  307.0     21.0  396.90  18.72
    # 1  0.02177  82.5   2.03   0.0  0.415  7.610   15.7  6.2700   2.0  348.0     14.7  395.38   3.11
    # 2  4.89822   0.0  18.10   0.0  0.631  4.970  100.0  1.3325  24.0  666.0     20.2  375.52   3.26
    # 3  0.03961   0.0   5.19   0.0  0.515  6.037   34.5  5.9853   5.0  224.0     20.2  396.90   8.01
    # 4  3.69311   0.0  18.10   0.0  0.713  6.376   88.4  2.5671  24.0  666.0     20.2  391.43  14.65

    # Normalize features
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    print(train_data[0])
    # [-0.27224633 -0.48361547 -0.43576161 -0.25683275 -0.1652266  -0.1764426
    #   0.81306188  0.1166983  -0.62624905 -0.59517003  1.14850044  0.44807713  0.8252202]
    ```
  - **keras 模型训练 / 验证 / 预测**
    ```py
    def build_model():
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1], )),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

        return model

    model = build_model()
    model.summary()
    # Layer (type)                 Output Shape              Param #   
    # =================================================================
    # dense_2 (Dense)              (None, 64)                896       
    # _________________________________________________________________
    # dense_3 (Dense)              (None, 64)                4160      
    # _________________________________________________________________
    # dense_4 (Dense)              (None, 1)                 65        
    # =================================================================
    # Total params: 5,121
    # Trainable params: 5,121
    # Non-trainable params: 0
    # _________________________________________________________________

    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    # Automatically stop training when the validation score doesn't improve
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    # Store training stats
    history = model.fit(train_data, train_labels, epochs=500, validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
    print(history.epoch[-1])  # 148

    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))  # Testing set Mean Abs Error: $2641.25
    ```
  - **准确率与损失的可视化图形**
    ```py
    ''' MAE 损失 '''
    fig = plt.figure()
    plt.plot(history.history['mean_absolute_error'], label='Train loss')
    plt.plot(history.history['val_mean_absolute_error'], label='Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    fig.tight_layout()
    ```
    ![](images/tensorlow_regression_boston_mae.png)
    ```py
    ''' 预测值偏差 '''
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [1000$]')
    plt.ylabel('Predictions [1000$]')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])

    error = test_predictions - test_labels
    fig.add_subplot(2, 1, 2)
    plt.hist(error, bins=50)
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error [1000$]")
    plt.ylabel("Count")

    fig.tight_layout()
    ```
    ![](images/tensorlow_regression_boston_predict.png)
## 过拟合与欠拟合
  - **欠拟合 Underfitting** 模型没有充分学习训练数据集上的数据相关性，在测试数据集上仍有提升空间，通常由于模型太简单 / 过度正则化 / 训练时间不够
  - **过拟合 Overfitting** 模型在训练数据集上获得很高的正确率，但学习的数据相关性不适用于测试数据集，使测试数据集上的正确率降低
  - **防止过拟合**
    - 最好的方法是使用更多的训练数据，模型可以获得更普遍的数据相关性
    - 使用参数正则化 regularization，模型可以学习更主要的数据特征
    - 使用 dropout，随机将某些特征替换为 0
    - 减小模型复杂度，缩减模型层数 / 单层参数的数量，复杂模型通常可以学习训练数据集中更多的数据相关性，但不一定适用于测试数据集
  - **加载 IMDB 电影评论数据集** 将数据转换为 one-hot 向量
    ```py
    NUM_WORDS = 10000

    (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
    def multi_hot_sequence(sequence, dimension):
        results = np.zeros((len(sequence), dimension))
        for ii, ww in enumerate(sequence):
            results[ii, ww] = 1.0
        return results

    train_data = multi_hot_sequence(train_data, dimension=NUM_WORDS)
    test_data = multi_hot_sequence(test_data, dimension=NUM_WORDS)
    ```
    ![](images/tensorflow_overfit_imdb_data.png)
  - **定义 keras 模型** 定义多个模型，复杂模型更容易过拟合
    ```py
    def define_simple_model(hidden_units):
        model = keras.Sequential([
            # `input_shape` is only required here so that `.summary` works.
            keras.layers.Dense(hidden_units, activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
            keras.layers.Dense(hidden_units, activation=tf.nn.relu),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        return model

    def train_model(model):
        model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

        history = model.fit(
                train_data, train_labels,
                epochs=20, batch_size=512,
                validation_data=(test_data, test_labels),
                verbose=2)

        return history

    # Train on a baseline model
    baseline_model = define_simple_model(16)
    baseline_history = train_model(baseline_model)

    # Train on a smaller model
    smaller_history = train_model(define_simple_model(4))

    # Train on a bigger model
    bigger_history = train_model(define_simple_model(512))

    ''' Plot the training and validation loss '''
    def plot_histories(histories, key='binary_crossentropy'):
        fig = plt.figure()
        for name, history in histories:
            val = plt.plot(history.history['val_' + key], '--', label=name.title() + ' Val')
            plt.plot(history.history[key], color=val[0].get_color(), label=name.title() + ' Train')

        plt.xlabel('Epochs')
        plt.ylabel(key.replace('_', ' ').title())
        plt.xlim([0, np.max(history.epoch)])
        plt.legend()
        fig.tight_layout()

    plot_histories([
        ('baseline', baseline_history),
        ('smaller', smaller_history),
        ('bigger', bigger_history)])
    ```
    ![](images/tensorflow_overfit_models.png)

    最大的模型可以更好地拟合训练数据集，但是会有更大的过拟合
  - **添加权重正则化 weight regularization** 在损失函数中添加权重的相关项，使模型选择更小的权重
    - **L1 正则化** 添加的损失与权重的绝对值相关，使用 keras 模型层的 `kernel_regularizer=keras.regularizers.l1()` 参数添加
    - **L2 正则化** 添加的损失与权重的平方值相关，使用 keras 模型层的 `kernel_regularizer=keras.regularizers.l2()` 参数添加
      ```py
      # 添加 l2 损失 0.001 * weight_coefficient_value ** 2
      l2(0.001)
      ```
    ```py
    l2_model = keras.Sequential([
        # `input_shape` is only required here so that `.summary` works.
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
            activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
            activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    l2_history = train_model(l2_model)

    plot_histories([
        ('baseline', baseline_history),
        ('l2', l2_history)])
    ```
    ![](images/tensorflow_overfit_l2.png)
  - **添加 dropout 层** 训练过程中随机丢弃上一层输出中的某些特征，如将 `[0.2, 0.5, 1.3, 0.8, 1.1]`转化为 `[0, 0.5, 1.3, 0, 1.1`，通常 `dropout rate` 设置为 **0.2 - 0.5**
    ```py
    dpt_model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS, )),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    dpt_history = train_model(dpt_model)

    dpt_with_l2_model = keras.Sequential([
        # `input_shape` is only required here so that `.summary` works.
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
            activation=tf.nn.relu, input_shape=(NUM_WORDS, )),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
            activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    dpt_with_l2_history = train_model(dpt_with_l2_model)
    plot_histories([
        ('baseline', baseline_history),
        ('dropout', dpt_history),
        ('dropout with l2', dpt_with_l2_history)])
    ```
    ![](images/tensorflow_overfit_dropout.png)
## Keras 模型保存与加载
  - 依赖 `h5py` `pyyaml`
    ```shell
    pip install h5py pyyaml
    ```
  - **keras 定义模型**，使用 MNIST 数据集
    ```py
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    def create_model():
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(28 * 28, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='sigmoid')
        ])

        model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])

        return model
    ```
  - **Model.save_weights** / **Model.load_weights** 保存 / 加载模型权重
    ```py
    ''' save_weights 保存 '''
    checkpoint_path = './model_checkpoint/cp.ckpt'

    model = create_model()
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images,test_labels))
    model.save_weights(checkpoint_path)

    ''' load_weights 加载 '''
    model = create_model()
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    # Restored model, accuracy: 11.60%

    model.load_weights(checkpoit_path)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    # Restored model, accuracy: 87.00%
    ```
  - **Model.save** / **keras.models.load_model** 保存 / 加载整个模型
    ```py
    # Save entire model to a HDF5 file
    model.save('my_model.h5')

    new_model = keras.models.load_model('my_model.h5')
    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    # Restored model, accuracy: 87.00%
    ```
  - **tf.keras.callbacks.ModelCheckpoint** 回调函数，训练过程中与训练结束后自动保存模型
    ```py
    class ModelCheckpoint(Callback)
    __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    ```
    - **period 参数** 指定每几次迭代保存一次
    ```py
    checkpoint_path = './model_checkpoint/cp.ckpt'
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

    model = create_model()
    model.fit(train_images, train_labels, epochs=10,
            validation_data=(test_images,test_labels),
            callbacks=[cp_callback])

    !ls {checkpoint_dir}
    # checkpoint  cp.ckpt.data-00000-of-00001  cp.ckpt.index
    ```
    ```py
    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "model_checkpoint_2/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)

    model = create_model()
    model.fit(train_images, train_labels,
              epochs = 50, callbacks = [cp_callback],
              validation_data = (test_images,test_labels),
              verbose=0)

    ls model_checkpoint_2/ -1t
    # checkpoint
    # cp-0050.ckpt.data-00000-of-00001
    # cp-0050.ckpt.index
    # cp-0045.ckpt.index
    # cp-0045.ckpt.data-00000-of-00001
    # cp-0040.ckpt.data-00000-of-00001
    # cp-0040.ckpt.index
    # cp-0035.ckpt.data-00000-of-00001
    # cp-0035.ckpt.index
    # cp-0030.ckpt.data-00000-of-00001
    # cp-0030.ckpt.index

    import pathlib

    # 查找目录下 .index 结尾的文件
    checkpoints = pathlib.Path(os.path.dirname(checkpoint_path)).glob('*.index')
    # 按照修改时间排序
    checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
    # 去掉后缀 .index
    checkpoints = [cp.with_suffix('') for cp in checkpoints]
    print([str(ii) for ii in checkpoints])
    # ['model_checkpoint_2/cp-0030.ckpt', 'model_checkpoint_2/cp-0035.ckpt',
    # 'model_checkpoint_2/cp-0040.ckpt', 'model_checkpoint_2/cp-0045.ckpt',
    # 'model_checkpoint_2/cp-0050.ckpt']

    # 取最新的
    latest = str(checkpoints[-1])
    model = create_model()
    model.load_weights(latest)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    Restored model, accuracy: 86.40%
    ```
***

# ML at production scale
# Build a linear model with Estimators
  - **Census 收入数据集** 包含了 1994 - 1995 个人的年龄 / 教育水平 / 婚姻状况 / 职业等信息，预测年收入是否达到 50,000 美元
  - [Predicting Income with the Census Income Dataset](https://github.com/tensorflow/models/tree/master/official/wide_deep)
    ```py
    tf.enable_eager_execution()

    ! git clone --depth 1 https://github.com/tensorflow/models
    ! mv models tensorflow_models

    models_path = os.path.join(os.getcwd(), 'tensorflow_models')
    sys.path.append(models_path)
    from official.wide_deep import census_dataset
    from official.wide_deep import census_main

    census_dataset.download('datasets/census_data')

    cd tensorflow_models/
    ! python -m official.wide_deep.census_main --model_type=wide --train_epochs=2 --dd '../datasets/census_data/'
    cd -
    ```
  - **读取 Census 数据集** 数据中包含 离散的类别列 Categorical column/ 连续数字列 continuous column / 分桶列 Bucketized column
    ```py
    train_file = "datasets/census_data/adult.data"
    test_file = "datasets/census_data/adult.test"

    train_df = pd.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
    test_df = pd.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)

    train_df.head()
    # Out[18]:
    #    age         workclass  education         occupation  ...   race  gender native_country income_bracket
    # 0   39         State-gov  Bachelors       Adm-clerical  ...  White    Male  United-States          <=50K
    # 1   50  Self-emp-not-inc  Bachelors    Exec-managerial  ...  White    Male  United-States          <=50K
    # 2   38           Private    HS-grad  Handlers-cleaners  ...  White    Male  United-States          <=50K
    # 3   53           Private       11th  Handlers-cleaners  ...  Black    Male  United-States          <=50K
    # 4   28           Private  Bachelors     Prof-specialty  ...  Black  Female           Cuba          <=50K

    # [5 rows x 15 columns]
    ```
  - **数据转化为 Tensor** `tf.estimator` 使用输入功能 `input_fn`
    ```py
    def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
        label = df[label_key]
        ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

        if shuffle:
          ds = ds.shuffle(10000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

    for feature_batch, label_batch in ds.take(1):
      print('Some feature keys:', list(feature_batch.keys())[:5])
      print()
      print('A batch of Ages  :', feature_batch['age'])
      print()
      print('A batch of Labels:', label_batch )

    import functools

    train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
    test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)
    ```
  - Selecting and Engineering Features for the Model

    Estimators use a system called feature columns to describe how the model should interpret each of the raw input features. An Estimator expects a vector of numeric inputs, and feature columns describe how the model should convert each feature.
    ```py
    classifier = tf.estimator.LinearClassifier(feature_columns=[age])
    classifier.train(train_inpf)
    result = classifier.evaluate(test_inpf)

    clear_output()  # used for display in notebook
    print(result)

    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    my_numeric_columns = [age,education_num, capital_gain, capital_loss, hours_per_week]

    fc.input_layer(feature_batch, my_numeric_columns).numpy()

    classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
    classifier.train(train_inpf)

    result = classifier.evaluate(test_inpf)

    clear_output()

    for key,value in sorted(result.items()):
      print('%s: %s' % (key, value))

    relationship = fc.categorical_column_with_vocabulary_list(
      'relationship',
      ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
    fc.input_layer(feature_batch, [age, fc.indicator_column(relationship)])

    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
    for item in feature_batch['occupation'].numpy():
        print(item.decode())

    occupation_result = fc.input_layer(feature_batch, [fc.indicator_column(occupation)])

    occupation_result.numpy().shape
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


    my_categorical_columns = [relationship, occupation, education, marital_status, workclass]

    classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns+my_categorical_columns)
    classifier.train(train_inpf)
    result = classifier.evaluate(test_inpf)

    clear_output()

    for key,value in sorted(result.items()):
      print('%s: %s' % (key, value))
    ```
***
```py
import inspect
print(inspect.getsource(inspect.getsource))
```
# optimizer
![](images/opt1.gif)
***

# 应用示例
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
