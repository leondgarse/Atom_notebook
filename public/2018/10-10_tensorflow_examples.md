# ___2018 - 10 - 10 Tensorflow Examples___
***

## How to Retrain an Image Classifier for New Categories
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
## Advanced Convolutional Neural Networks
  - [Advanced Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/deep_cnn)
  - [tensorflow/models/tutorials/image/cifar10/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/)
  - [tensorflow/models/tutorials/image/cifar10_estimator/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
***

# Sequences
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
***

# 数据表示 data representation
## Vector Representations of Words
## Improving Linear Models Using Explicit Kernel Methods
## Large-scale Linear Models with TensorFlow
***

# Non ML
## Mandelbrot set
## Partial differential equations
***

## GOO
  - [TensorFlow Hub](https://www.tensorflow.org/hub/)
  - [基于字符的LSTM+CRF中文实体抽取](https://github.com/jakeywu/chinese_ner)
  - [Matplotlib tutorial](http://www.labri.fr/perso/nrougier/teaching/matplotlib/)
  - [TensorFlow 实战电影个性化推荐](https://blog.csdn.net/chengcheng1394/article/details/78820529)
  - [TensorRec](https://github.com/jfkirk/tensorrec)
  - [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)

  ![](images/opt1.gif)
