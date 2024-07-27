# ___2018 - 10 - 22 Tensorflow Examples___
***

# How to Retrain an Image Classifier for New Categories
  - [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)
  - https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/
  ```py
  import tensorflow_hub as hub

  hub_module = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
  module = hub.Module(hub_module)
  ```
  **测试**
  ```py
  height, width = hub.get_expected_image_size(module)

  image_file = './datasets/flower_photos/daisy/100080576_f52e8ee070_n.jpg'
  images = tf.gfile.FastGFile(image_file, 'rb').read()
  images = tf.image.decode_jpeg(images)

  sess = tf.InteractiveSession()
  images.eval().shape

  imm = tf.image.resize_images(images, (height, width))
  imm = tf.expand_dims(imm, 0)  # A batch of images with shape [batch_size, height, width, 3].
  plt.imshow(imm[0].eval().astype('int'))

  tf.global_variables_initializer().run()
  features = module(imm).eval()  # Features with shape [batch_size, num_features].
  print(features.shape)
  # (1, 2048)
  ```
  ```py
  def jpeg_decoder_layer(module_spec):
      height, width = hub.get_expected_image_size(module_spec)
      input_depth = hub.get_num_image_channels(module_spec)
      jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
      imm = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

      imm = tf.image.convert_image_dtype(imm, dtype=tf.float32)
      imm = tf.expand_dims(imm, 0)
      imm = tf.image.resize_images(images, (height, width))

      return jpeg_data, imm
  ```
  **测试**
  ```py
  jj, ii = jpeg_decoder_layer(module)
  tt = sess.run(ii, {jj: tf.gfile.FastGFile(image_file, 'rb').read()})
  print(tt.shape)
  # (299, 299, 3)
  ```
  ```py
  CLASS_COUNT = 5
  def add_classifier_op(class_count, bottleneck_module, is_training, learning_rate=0.01):
      height, width = hub.get_expected_image_size(bottleneck_module)
      resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
      bottleneck_tensor = bottleneck_module(resized_input_tensor)
      batch_size, bottleneck_out = bottleneck_tensor.get_shape().as_list()  # None, 2048

      # Add a fully connected layer and a softmax layer
      with tf.name_scope('input'):
          bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_out], name='BottleneckInputPlaceholder')
          target_label = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')

      with tf.name_scope('final_retrain_ops'):
          with tf.name_scope('weights'):
              init_value = tf.truncated_normal([bottleneck_out, class_count], stddev=0.001)
              weights = tf.Variable(init_value, name='final_weights')
          with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
          with tf.name_scope('dense'):
              logits = tf.matmul(bottleneck_input, weights) + biases

      final_tensor = tf.nn.softmax(logits, name='final_result')

      # The tf.contrib.quantize functions rewrite the graph in place for
      # quantization. The imported model graph has already been rewritten, so upon
      # calling these rewrites, only the newly added final layer will be
      # transformed.
      if is_training:
          tf.contrib.quantize.create_training_graph()
      else:
          tf.contrib.quantize.create_eval_graph()

      # If this is an eval graph, we don't need to add loss ops or an optimizer.
      if not is_training:
          return None, None, bottleneck_input, target_label, final_tensor

      with tf.name_scope('cross_entropy'):
          cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=target_label, logits=logits)

      with tf.name_scope('train'):
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          train_step = optimizer.minimize(cross_entropy_mean)

      return (train_step, cross_entropy_mean, bottleneck_input, target_label, final_tensor)
  ```
  ```py
  flower_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
  train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(flower_url), origin=flower_url)

  def load_image_train_test(data_path, test_rate=10):
      rr = {}
      for sub_dir_name in tf.gfile.ListDirectory(data_path):
          sub_dir = os.path.join(data_path, sub_dir_name)
          print(sub_dir)
          if not tf.gfile.IsDirectory(sub_dir):
              continue

          item_num = len(tf.gfile.ListDirectory(sub_dir))

          train_dd = []
          test_dd = []
          for item_name in tf.gfile.ListDirectory(sub_dir):
              hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(item_name)).hexdigest()
              percentage_hash = int(hash_name_hashed, 16) % (item_num + 1) * (100 / item_num)
              if percentage_hash < 10:
                  test_dd.append(os.path.join(sub_dir, item_name))
              else:
                  train_dd.append(os.path.join(sub_dir, item_name))
          rr[sub_dir_name] = {'train': train_dd, 'test': test_dd}

      return rr
  ```
***

# Dog Species Classifier
## Tensorflow 1.14
  ```py
  import tensorflow as tf
  from tensorflow import keras
  # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True))
  config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  sess = tf.Session(config=config)
  keras.backend.set_session(sess)

  from tensorflow.python.keras import layers
  from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
  from PIL import ImageFile
  ImageFile.LOAD_TRUNCATED_IMAGES = True

  train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
      width_shift_range=0.2, height_shift_range=0.2, brightness_range=(0.1, 2),
      shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

  train_img_gen = train_data_gen.flow_from_directory('./dogImages/train/', target_size=(512, 512), batch_size=4, seed=1)
  val_data_gen = ImageDataGenerator(rescale=1./255)
  val_img_gen = val_data_gen.flow_from_directory('./dogImages/valid/', target_size=(512, 512), seed=1)

  img_shape = (512, 512, 3)
  xx = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
  # xx = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  xx.trainable = True
  model = tf.keras.Sequential([
      layers.Input(shape=img_shape),
      xx,
      layers.Conv2D(512, 1, strides=1, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)),
      # layers.MaxPooling2D(2),
      layers.Dropout(0.5),
      # layers.AveragePooling2D(pool_size=512, strides=512, padding='same'),
      layers.GlobalAveragePooling2D(),
      layers.Flatten(),
      layers.Dense(133, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.00001)),        
  ])
  model.summary()

  callbacks = [
      keras.callbacks.TensorBoard(log_dir='./logs'),
      keras.callbacks.ModelCheckpoint("./keras_checkpoints", monitor='val_loss', save_best_only=True),
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
  ]
  model.compile(optimizer=keras.optimizers.Adadelta(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(train_img_gen, validation_data=val_img_gen, epochs=50, callbacks=callbacks, verbose=1, workers=10)

  import glob2
  from skimage.io import imread
  from skimage.transform import resize

  model = tf.keras.models.load_model('keras_checkpoints')
  index_2_name = {vv: kk for kk, vv in train_img_gen.class_indices.items()}
  aa = resize(imread('./dogImages/1806687557.jpg'), (512, 512))
  pp = model.predict(np.expand_dims(aa, 0))
  print(index_2_name[pp.argmax()])
  # 029.Border_collie

  imm = glob2.glob('./dogImages/test/*/*')
  xx = np.array([resize(imread(ii), (512, 512)) for ii in imm])
  yy = np.array([int(os.path.basename(os.path.dirname(ii)).split('.')[0]) -1 for ii in imm])
  pp = model.predict(xx)
  tt = np.argmax(pp, 1)
  print((tt == yy).sum() / yy.shape[0])
  # 0.8588516746411483

  top_3_err = [(np.sort(ii)[-3:], ii.argmax(), imm[id]) for id, (ii, jj) in enumerate(zip(pp, yy)) if jj not in ii.argsort()[-3:]]
  print(1 - len(top_3_err) / yy.shape[0])
  # 0.965311004784689
  ```
## Tensorflow 2.0
  ```py
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])

  from tensorflow import keras
  from tensorflow.python.keras import layers
  from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
  from PIL import ImageFile

  ImageFile.LOAD_TRUNCATED_IMAGES = True

  img_shape = (224, 224, 3)
  train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
      width_shift_range=0.2, height_shift_range=0.2, brightness_range=(0.1, 2),
      shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
  train_img_gen = train_data_gen.flow_from_directory('./dogImages/train/', target_size=img_shape[:2], batch_size=4, seed=1)
  val_data_gen = ImageDataGenerator(rescale=1./255)
  val_img_gen = val_data_gen.flow_from_directory('./dogImages/valid/', target_size=img_shape[:2], batch_size=4, seed=1)                           

  xx = keras.applications.ResNet50V2(include_top=False, weights='imagenet')

  xx.trainable = True
  model = tf.keras.Sequential([
      layers.Input(shape=img_shape),
      xx,
      layers.Conv2D(512, 1, strides=1, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)),
      # layers.MaxPooling2D(2),
      layers.Dropout(0.5),
      # layers.AveragePooling2D(pool_size=512, strides=512, padding='same'),
      layers.GlobalAveragePooling2D(),
      layers.Flatten(),
      layers.Dense(133, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.00001)),                                                                                                                 
  ])
  model.summary()
  callbacks = [
      keras.callbacks.TensorBoard(log_dir='./logs'),
      keras.callbacks.ModelCheckpoint("./keras_checkpoints", monitor='val_loss', save_best_only=True),
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
  ]
  model.compile(optimizer=keras.optimizers.Adadelta(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(train_img_gen, validation_data=val_img_gen, epochs=50, callbacks=callbacks, verbose=1, workers=10)
  ```
  ```sh
  toco --saved_model_dir ./keras_checkpoints --output_file foo.tflite
  ```
  ```py
  from tensorflow_model_optimization.sparsity import keras as sparsity
  batch_size = 4
  end_step = np.ceil(train_img_gen.classes.shape[0] / batch_size).astype(np.int32) * 55
  pruning_params = {
      "pruning_schedule": sparsity.PolynomialDecay(
          initial_sparsity=0.5,
          final_sparsity=0.9,
          begin_step=2000,
          end_step=end_step,
          frequency=100)
  }

  pruned_model = tf.keras.Sequential([
      layers.Input(shape=img_shape),
      # sparsity.prune_low_magnitude(keras.applications.ResNet50V2(include_top=False, weights='imagenet'), **pruning_params),
      keras.applications.ResNet50V2(include_top=False, weights='imagenet'),
      sparsity.prune_low_magnitude(layers.Conv2D(512, 1, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)), **pruning_params),
      layers.Dropout(0.5),
      layers.GlobalAveragePooling2D(),
      layers.Flatten(),
      sparsity.prune_low_magnitude(layers.Dense(133, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.00001)), **pruning_params)
  ])
  ```
  ```py
  import glob2
  from skimage.io import imread
  from skimage.transform import resize
  imm = glob2.glob('./dogImages/test/*/*')
  xx = np.array([resize(imread(ii), (224, 224)) for ii in imm])
  ixx = tf.convert_to_tensor(xx, dtype='float32')
  # ixx = tf.convert_to_tensor(xx, dtype=tf.uint8)
  idd = tf.data.Dataset.from_tensor_slices((ixx)).batch(1)

  def representative_data_gen():
      for ii in idd.take(100):
          yield [ii]
  converter = tf.lite.TFLiteConverter.from_saved_model('./keras_checkpoints/')
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_data_gen
  tflite_quant_all_model = converter.convert()
  ```
***

# Kaggle Facial Keypoints
## Kaggle
  - [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook)
  - [Kaggle competitions gettingStarted](https://www.kaggle.com/competitions?sortBy=grouped&group=general&page=1&pageSize=20&category=gettingStarted)
  - [Kaggle Facial Keypoints](https://www.kaggle.com/c/facial-keypoints-detection)
  - [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
## Colab
  - [Colab kaggle_facial_keypoints_detection.ipynb](https://colab.research.google.com/drive/1lLHxtXTnbzW5M-mlk59av6rKtqqh_fwW)
## Fillna in data and conv model
  ```py
  cd ~/workspace/datasets/facial-keypoints-detection/

  train_data = pd.read_csv('./training.csv')

  aa = train_data['Image'][0]
  bb = np.array(aa.split(' '), dtype='int').reshape(96, 96, 1)
  plt.imshow(bb[:, :, 0], cmap='gray')

  train_data.isnull().any().value_counts()
  train_data.isnull().any(1).value_counts()
  train_data.fillna(method='ffill', inplace=True)

  imags = []
  for imm in train_data.Image:
      img = [int(ii) if len(ii.strip()) != 0 else 0 for ii in imm.strip().split(' ')]
      imags.append(img)
  train_x = np.array(imags).reshape(-1, 96, 96, 1) / 255
  train_y = train_data.drop('Image', axis=1).to_numpy() / 96
  np.fromstring(x, dtype=int, sep=' ').reshape((96,96))

  test_data = pd.read_csv('./test.csv')
  images = [np.fromstring(ii, dtype=int, sep=' ') for ii in test_data.Image]
  test_x = np.array(images).reshape(-1, 96, 96, 1) / 255

  np.savez('train_test', train_x=train_x, train_y=train_y, test_x=test_x)
  ```
  ```py
  fig, axis = plt.subplots(2, 5)
  axis = axis.flatten()
  for ax, imm, ipp in zip(axis, train_x, train_y):
      ax.imshow(imm[:, :, 0], cmap='gray')
      ax.scatter(ipp[0::2] * 96, ipp[1::2] * 96)
      ax.set_axis_off()

  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  aa = np.load('train_test.npz')
  train_x, train_y, test_x = aa['train_x'], aa['train_y'], aa['test_x']
  print(train_x.shape, train_y.shape)
  # (7049, 96, 96, 1) (7049, 30)

  from tensorflow import keras
  from tensorflow.keras.layers import LeakyReLU
  from tensorflow.keras.models import Sequential, Model
  from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

  def Conv_model(input_shape=(96,96,1), num_classes=30):
      model = Sequential()

      model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
      # model.add(BatchNormalization())
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())


      model.add(Flatten())
      model.add(Dense(512,activation='relu'))
      model.add(Dropout(0.1))
      model.add(Dense(num_classes))

      return model

  model = Conv_model(input_shape=(96,96,1), num_classes=30)
  model.summary()

  model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  hist = model.fit(train_x, train_y, epochs=150, batch_size=256, validation_split=0.2)
  # Epoch 50/50
  # 5639/5639 [==============================] - 3s 534us/sample - loss: 0.0055 - mae: 0.0551 - val_loss: 0.0015 - val_mae: 0.0275
  # Epoch 200/200
  # 5639/5639 [==============================] - 3s 550us/sample - loss: 0.0017 - mae: 0.0290 - val_loss: 6.4047e-04 - val_mae: 0.0148
  # Epoch 250/250
  # 5639/5639 [==============================] - 3s 550us/sample - loss: 6.3462e-04 - mae: 0.0157 - val_loss: 4.1988e-04 - val_mae: 0.0107
  tf.saved_model.save(model, './1')

  loaded = tf.saved_model.load('./1')
  interf = loaded.signatures['serving_default']
  pred = interf(tf.convert_to_tensor(test_x, dtype=tf.float32))
  pp = pred['dense_1'].numpy()

  fig, axis = plt.subplots(5, 5)
  axis = axis.flatten()
  for ax, imm, ipp in zip(axis, test_x[1000:], pp[1000:]):
      ax.imshow(imm[:, :, 0], cmap='gray')
      ax.scatter(ipp[0::2] * 96, ipp[1::2] * 96)
      ax.set_axis_off()
  ```
## Train two mini xception model with separated data
  ```py
  csv_data = pd.read_csv('./training.csv')
  all_image = csv_data.Image
  all_data = csv_data.drop('Image', axis=1)

  aa = all_data.isnull().apply(pd.value_counts)
  vv = aa.iloc[0] > 7000
  vv = all_data.count() > 7000
  # print(aa.iloc[0])

  integrity_columns = vv[vv == True].index
  integrity_data = all_data[integrity_columns]
  unintegrity_columns = vv[vv == False].index
  unintegrity_data = all_data[unintegrity_columns]
  print(integrity_data.shape, unintegrity_data.shape)
  # (7049, 8) (7049, 22)

  integrity_data_select = integrity_data.notnull().all(1)
  integrity_data = integrity_data[integrity_data_select].to_numpy() / 96
  unintegrity_data_select = unintegrity_data.notnull().all(1)
  unintegrity_data = unintegrity_data[unintegrity_data_select].to_numpy() / 96
  print(integrity_data.shape, unintegrity_data.shape)
  # (7000, 8) (2155, 22)

  image_data = [np.fromstring(ii, dtype=int, sep=' ') for ii in all_image]
  image_data = np.array(image_data).reshape(-1, 96, 96, 1) / 255
  integrity_image_data = image_data[integrity_data_select]
  unintegrity_image_data = image_data[unintegrity_data_select]
  print(integrity_image_data.shape, unintegrity_image_data.shape)
  # (7000, 96, 96, 1) (2155, 96, 96, 1)

  aa = integrity_image_data[:, :, ::-1, :]
  integrity_image_data_2 = np.concatenate([integrity_image_data, aa])
  bb = np.array([np.abs([1, 0] * 4 - ii) for ii in integrity_data])
  integrity_data_2 = np.concatenate([integrity_data, bb])
  print(integrity_image_data_2.shape, integrity_data_2.shape)
  # (14000, 96, 96, 1) (14000, 8)
  aa = unintegrity_image_data[:, :, ::-1, :]
  unintegrity_image_data_2 = np.concatenate([unintegrity_image_data, aa])
  bb = np.array([np.abs([1, 0] * 11 - ii) for ii in unintegrity_data])
  unintegrity_data_2 = np.concatenate([unintegrity_data, bb])
  print(unintegrity_image_data_2.shape, unintegrity_data_2.shape)
  # (4310, 96, 96, 1) (4310, 22)
  ```
  ```py
  ## model
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Activation, Dropout, Conv2D, Dense, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, concatenate
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
      regularization = l2(l2_regularization)

      # base
      img_input = Input(input_shape)
      x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      # module 1
      residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      # module 2
      residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      # module 3
      residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      # module 4
      residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      x = Conv2D(num_classes, (3, 3), kernel_regularizer=regularization, padding='same')(x)
      x = GlobalAveragePooling2D()(x)
      x = Dropout(0.3)(x)
      output = Dense(num_classes)(x)

      model = Model(img_input, output)
      return model

  #training the model
  model = mini_XCEPTION((96, 96, 1), 30)
  model.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
  model.summary()

  # callbacks
  early_stop = EarlyStopping('val_loss', patience=50)
  reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(50/4), verbose=1)
  model_checkpoint = ModelCheckpoint("./keras_checkpoints", 'val_loss', verbose=1, save_best_only=True)
  callbacks = [model_checkpoint, early_stop, reduce_lr]
  hist = model.fit(train_x, train_y, batch_size=256, epochs=200, verbose=1, callbacks=callbacks, validation_split=0.2)
  # 5639/5639 [==============================] - 9s 2ms/sample - loss: 0.0014 - mae: 0.0263 - accuracy: 0.5779 - val_loss: 9.2967e-04 - val_mae: 0.0209 - val_accuracy: 0.6149

  modela = mini_XCEPTION((96, 96, 1), 8)
  modela.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
  modela.summary()
  hist = modela.fit(integrity_image_data, integrity_data, batch_size=256, epochs=200, verbose=1, callbacks=callbacks, validation_split=0.2)

  modelb = mini_XCEPTION((96, 96, 1), 22)
  modelb.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
  modelb.summary()
  hist = modelb.fit(unintegrity_image_data, unintegrity_data, batch_size=256, epochs=200, verbose=1, callbacks=callbacks, validation_split=0.2)
  ```
## Fill missing data by model predict
  ```py
  train_select = label_data.notnull().all(1)
  test_select = label_data.isnull().any(1)
  feature_data_train = feature_data[train_select]
  feature_data_test = feature_data[test_select]
  label_data_train = label_data[train_select]
  label_data_test = label_data[test_select]
  print(feature_data_train.shape, feature_data_test.shape, label_data_train.shape, label_data_test.shape)
  # (2140, 8) (4860, 8) (2140, 22) (4860, 22)

  data_model = tf.keras.models.Sequential([
      layers.Input(shape=[8,]),
      layers.Dense(32),
      layers.Dense(64),
      layers.Dropout(0.1),
      layers.Dense(22)
  ])
  data_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  data_model.fit(feature_data_train.to_numpy() / 96, label_data_train.to_numpy() / 96, epochs=150, verbose=1, validation_split=0.2)
  pred = data_model.predict(feature_data_train.to_numpy() / 96) * 96
  yy = label_data_train.to_numpy()
  print(((pred - yy) ** 2).sum(0))
  # [ 3621.08980817  1323.51842721  4242.13195221  1211.73041032
  #   3259.46735638  1228.29699784  4461.2025572   1299.45073064
  #   7967.08317693  9896.2162609   9108.37986825 11682.81402873
  #   8811.50593809 10063.70865708  8610.68334281 10994.46973902
  #  10112.65304633 15877.52864348 10719.43984155 16190.33853022
  #   1266.60829962 25825.21071116]
  print(((pred - yy) ** 2).max(0))
  # [ 29.89434935  10.22904536  39.20663097  15.79585616  25.61755941
  #   10.01605354  41.52014917  17.64380907  43.83821083 151.01494419
  #  414.56626675 133.13552419  44.85152737 143.92310498 408.80228546
  #   92.91965074  71.18659086 253.96232386 135.8481257  180.55287795
  #   16.11288025 335.5595574 ]

  train_image = train_image[select_data.notnull().all(1)]
  image_string_train = train_image[train_select]
  image_string_test = train_image[test_select]

  image_data_train = [np.fromstring(ii, dtype=int, sep=' ') for ii in image_string_train]
  image_data_train = np.array(image_data_train).reshape(-1, 96, 96, 1) / 255
  image_data_test = [np.fromstring(ii, dtype=int, sep=' ') for ii in image_string_test]
  image_data_test = np.array(image_data_test).reshape(-1, 96, 96, 1) / 255
  print(image_data_train.shape, image_data_test.shape)
  # (2140, 96, 96, 1) (4860, 96, 96, 1)

  mouth_sub_data = train_data[train_data.columns[-10:]]
  mouth_train = mouth_sub_data[mouth_sub_data.notnull().all(1)]
  mouth_test = mouth_sub_data[mouth_sub_data.isnull().any(1)]

  mouth_train_colx_x = mouth_train[['nose_tip_x', 'mouth_center_bottom_lip_x']]
  mouth_train_colx_y = mouth_train[['mouth_left_corner_x', 'mouth_right_corner_x', 'mouth_center_top_lip_x']]

  model_mouth_x = tf.keras.models.Sequential([
      layers.Input(shape=[2,]),
      layers.Dense(10),
      layers.Dense(10),
      layers.Dense(3)])
  model_mouth_x.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  model_mouth_x.fit(mouth_train_x_x.to_numpy() / 96, mouth_train_x_y.to_numpy() / 96, epochs=150, verbose=1, validation_split=0.2)
  pred = model_mouth_x.predict(mouth_train_x_x.to_numpy() / 96) * 96
  print(((pred - yy) ** 2).sum(0))
  # [17116.85021261 14906.93838227   988.74043935]
  print(((pred - yy) ** 2).max(0))
  # [220.70537516 158.71248198  17.21451939]

  mouth_test_colx_x = mouth_test[['nose_tip_x', 'mouth_center_bottom_lip_x']]
  mouth_test_colx_y = mouth_test[['mouth_left_corner_x', 'mouth_right_corner_x', 'mouth_center_top_lip_x']]
  ```
  ```py
  df.describe().loc['count'].plot.bar()
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import MinMaxScaler

  output_pipe = make_pipeline(
      MinMaxScaler(feature_range=(-1, 1))
  )

  y_train = output_pipe.fit_transform(y)
  xy_predictions = output_pipe.inverse_transform(predictions).reshape(15, 2)
  ```
***

# Links
## Advanced Convolutional Neural Networks
  - [Advanced Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/deep_cnn)
  - [tensorflow/models/tutorials/image/cifar10/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/)
  - [tensorflow/models/tutorials/image/cifar10_estimator/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
## Recurrent Neural Networks
  - [Recurrent Neural Networks](https://www.tensorflow.org/tutorials/sequences/recurrent)
  - [tensorflow/models/tutorials/rnn/ptb/](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)
## Recurrent Neural Networks for Drawing Classification
  - [Recurrent Neural Networks for Drawing Classification](https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw)
  - [tensorflow/models/tutorials/rnn/quickdraw/](https://github.com/tensorflow/models/tree/master/tutorials/rnn/quickdraw)
## Simple Audio Recognition
  - [Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
  - [tensorflow/tensorflow/examples/speech_commands/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands)
## Neural Machine Translation seq2seq Tutorial
  - [tensorflow/nmt](https://github.com/tensorflow/nmt)
## Optimizers
  ![](images/opt1.gif)
***
