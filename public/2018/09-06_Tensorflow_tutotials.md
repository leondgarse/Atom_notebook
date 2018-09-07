# ___2018 - 09 - 06 Tensorflow Tutorials___
***

- [Tensorflow Tutorials](https://www.tensorflow.org/tutorials/)
# 目录
***

# Learn and use ML
## Train your first neural network: basic classification
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
## Text classification with movie reviews
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
## Predict house prices: regression
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
***

# optimizer
![](images/opt1.gif)
