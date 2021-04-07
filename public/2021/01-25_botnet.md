- [aravindsrinivas/botnet.py](https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2)
- [leondgarse/botnet.py](https://gist.github.com/leondgarse/351dba9457c5a36516aea3ce1950ac74)
- **botnet MHSA**
  ```py
  from icecream import ic
  inputs = keras.layers.Input([14, 16, 1024])
  featuremap = inputs

  print(botnet.MHSA(featuremap, 512, pos_enc_type='relative', heads=4).shape)
  # (None, 14, 16, 512)

  q = botnet.group_pointwise(featuremap, proj_factor=1, name='q_proj', heads=4, target_dimension=512)
  k = botnet.group_pointwise(featuremap, proj_factor=1, name='k_proj', heads=4, target_dimension=512)
  v = botnet.group_pointwise(featuremap, proj_factor=1, name='v_proj', heads=4, target_dimension=512)

  ic(q.shape.as_list(), k.shape.as_list(), v.shape.as_list())
  # q.shape.as_list(): [None, 4, 14, 16, 128]
  print(botnet.relpos_self_attention(q=q, k=k, v=v, relative=True, fold_heads=True).shape)
  # (None, 14, 16, 512)

  relative, fold_heads = True, True
  bs, heads, h, w, dim = q.shape
  int_dim = int(dim)
  q = q * (dim ** -0.5) # scaled dot-product
  logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
  if relative:
      logits += botnet.relative_logits(q)
  # weights = tf.reshape(logits, [-1, heads, h, w, h * w])
  # weights = tf.nn.softmax(weights)
  # weights = tf.reshape(weights, [-1, heads, h, w, h, w])
  weights = tf.nn.softmax(logits)
  attn_out = tf.einsum('bhHWPQ,bhPQd->bHWhd', weights, v)
  if fold_heads:
      attn_out = tf.reshape(attn_out, [-1, h, w, heads * dim])
  ic(attn_out.shape.as_list())
  # ic| attn_out.shape.as_list(): [None, 14, 16, 512]
  ```
- **relative_logits**
  ```py
  def rel_to_abs(x):
      """
      Converts relative indexing to absolute.
      Input: [bs, heads, h, w, 2*w - 1]
      Output: [bs, heads, h, w, w]
      """
      bs, heads, h, w, dim = x.shape
      col_pad = tf.zeros_like(x[:, :, :, :, :1], dtype=x.dtype)
      x = tf.concat([x, col_pad], axis=-1)
      flat_x = tf.reshape(x, [-1, heads, h, w * 2 * w])
      flat_pad = tf.zeros_like(flat_x[:, :, :, :w-1], dtype=x.dtype)
      flat_x_padded = tf.concat([flat_x, flat_pad], axis=-1)
      final_x = tf.reshape(flat_x_padded, [-1, heads, h, w+1, 2*w-1])
      final_x = final_x[:, :, :, :w, w-1:]
      return final_x


  def relative_logits_1d(*, q, rel_k, transpose_mask):
      """
      Compute relative logits along one dimenion.
      `q`: [bs, heads, height, width, dim]
      `rel_k`: [2*width - 1, dim]
      """
      bs, heads, h, w, dim = q.shape
      # rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
      rel_logits = tf.matmul(q, tf.transpose(rel_k, [1, 0]))
      rel_logits = rel_to_abs(rel_logits)
      rel_logits = tf.expand_dims(rel_logits, axis=3)
      rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
      rel_logits = tf.transpose(rel_logits, transpose_mask)
      return rel_logits


  def relative_logits(q):
      bs, heads, h, w, dim = q.shape
      stddev = dim ** -0.5
      rel_emb_w = tf.compat.v1.get_variable('r_width', shape=(2*w - 1, dim), dtype=q.dtype, initializer=tf.random_normal_initializer(stddev=stddev))
      rel_logits_w = relative_logits_1d(q=q, rel_k=rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])

      # Relative logits in height dimension.
      rel_emb_h = tf.compat.v1.get_variable('r_height', shape=(2*h - 1, dim), dtype=q.dtype, initializer=tf.random_normal_initializer(stddev=stddev))
      rel_logits_h = relative_logits_1d(q=tf.transpose(q, [0, 1, 3, 2, 4]), rel_k=rel_emb_h, transpose_mask=[0, 1, 4, 2, 5, 3])
      return rel_logits_h + rel_logits_w
  ```
  ```py
  aa = tf.convert_to_tensor(np.arange(45).reshape(1, 1, 3, 3, 5))
  rel_to_abs(aa)
  print(aa[0, 0].numpy())
  # [[[ 0  1  2  3  4]
  #   [ 5  6  7  8  9]
  #   [10 11 12 13 14]]
  #  [[15 16 17 18 19]
  #   [20 21 22 23 24]
  #   [25 26 27 28 29]]
  #  [[30 31 32 33 34]
  #   [35 36 37 38 39]
  #   [40 41 42 43 44]]]
  print(rel_to_abs(aa)[0, 0].numpy())
  # [[[ 2  3  4]
  #   [ 6  7  8]
  #   [10 11 12]]
  #  [[17 18 19]
  #   [21 22 23]
  #   [25 26 27]]
  #  [[32 33 34]
  #   [36 37 38]
  #   [40 41 42]]]
  ```
- **keras.layers.MultiHeadAttention**
  ```py
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import special_math_ops
  from icecream import ic
  inputs = keras.layers.Input([14, 16, 1024])

  nn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
  ic(nn(inputs, inputs).shape.as_list())
  # ic| nn(inputs, inputs).shape.as_list(): [None, 14, 16, 1024]

  query = nn._query_dense(inputs)
  key = nn._key_dense(inputs)
  value = nn._value_dense(inputs)
  ic(query.shape.as_list(), key.shape.as_list(), value.shape.as_list())
  # ic| query.shape.as_list(): [None, 14, 16, 4, 128]

  # attention_output, attention_scores = nn._compute_attention(query, key, value)
  query = math_ops.multiply(query, 1.0 / math.sqrt(float(nn._key_dim)))
  # 'afgde,abcde->adbcfg', 'bhHWd,bhPQd->bhHWPQ' == 'afgde,adbce->afgdbc'
  attention_scores = special_math_ops.einsum(nn._dot_product_equation, key, query)
  ic(attention_scores.shape.as_list())
  # ic| attention_scores.shape.as_list(): [None, 4, 14, 16, 14, 16]

  if relative:
      query = tf.transpose(query, [0, 3, 1, 2, 4])
      attention_scores += relative_logits(query)
  attention_scores = nn._masked_softmax(attention_scores, None)
  attention_scores_dropout = nn._dropout_layer(attention_scores, training=False)
  attention_output = special_math_ops.einsum(nn._combine_equation, attention_scores_dropout, value)
  ic(attention_output.shape.as_list())
  # ic| attention_output.shape.as_list(): [None, 14, 16, 4, 128]

  attention_output = nn._output_dense(attention_output)
  ic(attention_output.shape.as_list())
  # ic| attention_output.shape.as_list(): [None, 14, 16, 1024]
  ```
  ```py
  def rel_to_abs(x):
      bs, heads, h, w, dim = x.shape
      col_pad = tf.zeros_like(x[:, :, :, :, :1], dtype=x.dtype)
      x = tf.concat([x, col_pad], axis=-1)
      flat_x = tf.reshape(x, [-1, heads, h, w * 2 * w])
      flat_pad = tf.zeros_like(flat_x[:, :, :, :w-1], dtype=x.dtype)
      flat_x_padded = tf.concat([flat_x, flat_pad], axis=-1)
      final_x = tf.reshape(flat_x_padded, [-1, heads, h, w+1, 2*w-1])
      final_x = final_x[:, :, :, :w, w-1:]
      return final_x

  def relative_logits_1d(*, q, rel_k, transpose_mask):
      bs, heads, h, w, dim = q.shape
      rel_logits = tf.matmul(q, tf.transpose(rel_k, [1, 0]))
      rel_logits = rel_to_abs(rel_logits)
      rel_logits = tf.expand_dims(rel_logits, axis=3)
      rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
      rel_logits = tf.transpose(rel_logits, transpose_mask)
      return rel_logits

  def relative_logits(q):
      rel_logits_w = relative_logits_1d(q=q, rel_k=rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
      rel_logits_h = relative_logits_1d(q=tf.transpose(q, [0, 1, 3, 2, 4]), rel_k=rel_emb_h, transpose_mask=[0, 1, 4, 2, 5, 3])
      return rel_logits_h + rel_logits_w
  ```
***
# 图像超分
  - [Github idealo/image-super-resolution](https://github.com/idealo/image-super-resolution)
    ```py
    !cd image-super-resolution && pip install -q . && cd -

    import os
    import matplotlib.pyplot as plt
    from skimage.io import imread, imsave
    from ISR.models import RDN, RRDN

    # model = RRDN(weights='gans')        # 17462488
    # model = RDN(weights='noise-cancel') # 66071288
    # model = RDN(weights='psnr-small')   # 10694096
    model = RDN(weights='psnr-large')     # 66071288

    image_name = "307622911.jpg"
    imm = imread(image_name)
    sr_img = model.predict(imm, by_patch_of_size=100)
    print("Source:", imm.shape, "Target:", sr_img.shape)

    plt.imshow(sr_img)
    plt.axis("off")
    plt.tight_layout()

    output = os.path.splitext(os.path.basename(image_name))[0] + "_x2_psnr-large.png"
    print("Saving to:", output)
    imsave(output, sr_img)
    ```
  - [Github yinboc/liif](https://github.com/yinboc/liif)
    ```py
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.io import imread, imsave
    from tqdm import tqdm

    # import sys
    # sys.path.append("/content/liif")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import models

    class ImageSuperResolution:
        def __init__(self, model_path):
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
            self.device = torch.device(device_name)
            self.model = models.make(torch.load(model_path, map_location=self.device)['model'], load_sd=True).to(self.device)

        def make_coord(self, shape, ranges=None, flatten=True):
            """ Make coordinates at grid centers.0000
            """
            coord_seqs = []
            for i, n in enumerate(shape):
                if ranges is None:
                    v0, v1 = -1, 1
                else:
                    v0, v1 = ranges[i]
                r = (v1 - v0) / (2 * n)
                seq = v0 + r + (2 * r) * torch.arange(n).float()
                coord_seqs.append(seq)
            ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
            if flatten:
                ret = ret.view(-1, ret.shape[-1])
            return ret

        def batched_predict(self, inp, coord, cell, bsize):
            with torch.no_grad():
                self.model.gen_feat(inp)
                total = coord.shape[1]
                steps = int(np.ceil(total / bsize))
                preds = [self.model.query_rgb(coord[:, ii: ii + bsize, :], cell[:, ii: ii + bsize, :]) for ii in tqdm(range(0, total, bsize))]
                preds = torch.cat(preds, dim=1)
            return preds

        def image_super_resolution(self, image, resolution, bsize=30000):
            if isinstance(image, str):
                image = imread(image) / 255
            inputs = torch.from_numpy(image.transpose(2, 0, 1).astype("float32"))
            inputs = ((inputs - 0.5) / 0.5).to(self.device)

            if isinstance(resolution, str) and ',' in resolution:
                hh, ww = [int(ii.strip()) for ii in resolution.split(',')]
            else:
                img_h, img_w = inputs.shape[1:]
                hh, ww = img_h * int(resolution), img_w * int(resolution)
            coord = self.make_coord((hh, ww)).to(self.device)
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / hh
            cell[:, 1] *= 2 / ww
            pred = self.batched_predict(inputs.unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=bsize)[0]
            pred = (pred * 0.5 + 0.5).clamp(0, 1).view(hh, ww, 3)

            return pred.cpu().numpy()

        def to_4k_resolution_2160_3840(self, src_hh, src_ww, target_hh=2160, target_ww=3840):
            hh_enlarge, ww_enlarge = target_hh / src_hh, target_ww / src_ww
            if hh_enlarge > ww_enlarge:
                target_ww = src_ww * hh_enlarge
            else:
                target_hh = src_hh * ww_enlarge
            return int(target_hh), int(target_ww)

        def to_4k_image(self, image_name, output=None, bsize=30000):
            image = imread(image_name) / 255
            src_hh, src_ww = image.shape[:2]
            hh, ww = self.to_4k_resolution_2160_3840(src_hh, src_ww)
            print(">>>> Source shape: [%d, %d], Target shape: [%d, %d]" % (src_hh, src_ww, hh, ww))
            if output is None:
                output = os.path.splitext(os.path.basename(image_name))[0] + "_4k.png"
            idd = self.image_super_resolution(image, resolution="{},{}".format(hh, ww), bsize=bsize)

            print(">>>> Saving to %s ..." % (output))
            imsave(output, (idd * 255).astype("uint8"))
            return idd
    ```
    ```py
    aa = ImageSuperResolution("pre_trained/edsr-baseline-liif.pth")
    idd = aa.image_super_resolution('demo/0829x4-crop.png', 4, 300)

    aa = ImageSuperResolution("pre_trained/rdn-liif.pth")
    idd = aa.to_4k_image("602.jpg")
    ```
  - [Github krasserm/super-resolution](https://github.com/krasserm/super-resolution)
    ```py
    from model import resolve_single
    from utils import load_image, plot_sample
    from model.srgan import generator
    model = generator()
    model.load_weights('weights/srgan/gan_generator.h5')

    image_name = "../liif/307622911/307622911.jpg"
    lr = load_image(image_name)
    sr = resolve_single(model, lr)
    output = os.path.splitext(os.path.basename(image_name))[0] + "_x4_srgan.png"
    imsave(output, sr.numpy())

    plot_sample(lr, sr)

    ''' edsr '''
    from model.edsr import edsr
    model = edsr(scale=4, num_res_blocks=16)
    model.load_weights('weights/edsr-16-x4/weights.h5')
    sr = resolve_single(model, lr)
    output = os.path.splitext(os.path.basename(image_name))[0] + "_x4_edsr.png"
    imsave(output, sr.numpy())

    ''' wdsr '''
    from model.wdsr import wdsr_b
    model = wdsr_b(scale=4, num_res_blocks=32)
    model.load_weights('weights/wdsr-b-32-x4/weights.h5')
    sr = resolve_single(model, lr)
    output = os.path.splitext(os.path.basename(image_name))[0] + "_x4_wdsr.png"
    imsave(output, sr.numpy())
    ```
  - **Comparing**
    ```py
    def plot_results_sub(image_folder, sub_left, sub_right, sub_top, sub_bottom, plot_cols):
        iaa = sort(glob(os.path.join(image_folder, '*')))
        imms = [imread(ii) for ii in iaa]
        base_height = imms[0].shape[0]
        ees = [ii.shape[0] // base_height for ii in imms]
        subs = [ii[sub_top * ee:sub_bottom * ee, sub_left * ee:sub_right * ee, :] for ii, ee in zip(imms, ees)]

        cols = plot_cols
        rows = int(np.ceil(len(iaa) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()
        name_skip_len = len(os.path.splitext(os.path.basename(iaa[0]))[0])
        for ax, name, sub in zip(axes, iaa, subs):
            ax.imshow(sub)
            title = os.path.splitext(os.path.basename(name))[0][name_skip_len:]
            title = "Orignal" if len(title) == 0 else title
            ax.set_title(title)
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    fig = plot_results_sub("Blueprint_Wallpaper", 400, 600, 600, 900, 3)
    fig = plot_results_sub("307622911", 150, 300, 100, 250, 4)
    ```
  ```py
  from to_4k_image import ImageSuperResolution

  isr = ImageSuperResolution("pre_trained/edsr-baseline-liif.pth")

  image_name = '307622911/307622911.jpg'
  imm = imread(image_name)
  fig = plt.figure()
  plt.imshow(imm)
  plt.tight_layout()

  imms = [imm / 255]
  resulution = 4
  while True:
      print("Waiting for input...")
      aa = plt.ginput(2)
      aa = np.array(aa)

      left, right = int(aa[:, 0].min()), int(aa[:, 0].max())
      top, bottom = int(aa[:, 1].min()), int(aa[:, 1].max())

      if right - left < 1 or bottom - top < 1:
          if len(imms) > 1:
              imms.pop(-1)
          else:
              break
              # pass
      else:
          sub_image = imms[-1][top:bottom, left:right, :]
          idd = isr.image_super_resolution(sub_image, resulution)
          imms.append(idd)
      plt.imshow(imms[-1])
      plt.draw()
      print("Sub image displayed")
  ```
***

# Float16 mixed precision
## Basic test
  ```py
  from icecream import ic

  keras.mixed_precision.set_global_policy('mixed_float16')
  policy = keras.mixed_precision.global_policy()
  ic(policy.compute_dtype, policy.variable_dtype)

  inputs = keras.layers.Input([10])
  dd = keras.layers.Dense(10)
  dd.build([10])
  mm = keras.models.Model(inputs, dd(inputs))

  ic(dd(np.ones([1, 10])).dtype)  # ic| dd(np.ones([1, 10])).dtype: tf.float16
  ic(dd.weights[0].dtype) # ic| dd.weights[0].dtype: tf.float32
  ic(inputs.dtype)  # ic| inputs.dtype: tf.float32
  ic(mm.outputs[0].dtype) # ic| mm.outputs[0].dtype: tf.float16
  ```
  ```py
  import json
  json_config = mm.to_json()
  aa = json.loads(json_config)
  with open('model_fp16.json', 'w') as ff:
      json.dump(aa, ff, indent=2)
  ```
  ```py
  keras.mixed_precision.set_global_policy('mixed_float16')
  # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

  def load_cifar10(batch_size=1024, image_shape=(32, 32), classes=10):
      # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
      # x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
      import tensorflow_datasets as tfds
      AUTOTUNE = tf.data.experimental.AUTOTUNE

      if image_shape[:2] == (32, 32):
          preprocessing = lambda data: (tf.cast(data["image"], tf.float32) / 255.0, tf.one_hot(data["label"], classes))
      else:
          preprocessing = lambda data: (tf.image.resize(data["image"], image_shape[:2]) / 255.0, tf.one_hot(data["label"], classes))
      dataset = tfds.load("cifar10", split="train").map(preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
      return dataset

def test_dense_model(num_classes=10):
    return keras.Sequential([
        # keras.Input(shape=(784,), name='digits'),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dense(num_classes),
        keras.layers.Activation('softmax', dtype='float32'),
    ])

def test_conv_model(num_classes=10, input_shape=(32, 32, 3)):
    return keras.models.Sequential([
        keras.layers.Conv2D(8, 3, padding="same", activation="relu", input_shape=input_shape),
        keras.layers.DepthwiseConv2D(3, depth_multiplier=8, padding="same", activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(num_classes),
        keras.layers.Activation("softmax", dtype="float32"),
    ])

  # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  # x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
  # initial_weights = model.get_weights()
  input_shape, classes = (128, 128, 3), 80000
  dataset = load_cifar10(batch_size=512, image_shape=input_shape, classes=classes)

  # model = test_dense_model(classes)
  # model = test_conv_model(classes, input_shape=input_shape)
  # model = keras.applications.MobileNet(include_top=True, classes=classes, input_shape=input_shape, weights=None)
  model = keras.applications.ResNet50(include_top=True, classes=classes, input_shape=input_shape, weights=None)
  model = keras.models.Model(model.inputs[0], keras.layers.Activation("linear", dtype="float32")(model.outputs[0]))

  # optimizer = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam())
  optimizer = keras.optimizers.Adam()
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)
  # history = model.fit(x_train, y_train, batch_size=1024, epochs=5, validation_split=0.2)
  history = model.fit(dataset, epochs=2)
  ```
| Model      | Dataset      | batchsize | float16 | XLA   | first epoch (ms/step) | min (ms/step) |
| ---------- | ------------ | --------- | ------- | ----- | --------------------- | ------------- |
| DenseModel | MNIST        | 8192      | False   | False | 223                   | 113           |
| DenseModel | MNIST        | 8192      | True    | False | 101                   | 56            |
| DenseModel | MNIST        | 8192      | False   | True  | 227                   | 111           |
| DenseModel | MNIST        | 8192      | True    | True  | 144                   | 56            |
| ConvModel  | cifar10      | 1024      | False   | False | 118                   | 110           |
| ConvModel  | cifar10      | 1024      | True    | False | 41                    | 38            |
| MobileNet  | cifar10, 32  | 1024      | Fasle   | False | 89                    | 59            |
| MobileNet  | cifar10, 32  | 1024      | True    | False | 44                    | 41            |
| MobileNet  | cifar10, 128 | 128       | False   | False | 142                   | 139           |
| MobileNet  | cifar10, 128 | 128       | True    | False | 65                    | 62            |
| Resnet50   | cifar10, 32  | 128       | False   | False | 69                    | 64            |
| Resnet50   | cifar10, 32  | 128       | True    | False | 74                    | 71            |
| Resnet50   | cifar10, 128 | 128       | False   | False | 187                   | 184           |
| Resnet50   | cifar10, 128 | 128       | True    | False | 128                   | 122           |

## Resnet50 cifar10
- [Keras mixed precision API 50x slower than mixed precision graph rewrite](https://github.com/tensorflow/tensorflow/issues/41715)
- [Much worse performance when using mixed precision training (using tensorflow.keras policy)](https://github.com/tensorflow/tensorflow/issues/39556)
```py
import tensorflow as tf
from tensorflow import keras

def load_cifar10(batch_size=1024, image_shape=(32, 32)):
    import tensorflow_datasets as tfds
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    if image_shape[:2] == (32, 32):
        preprocessing = lambda data: (tf.cast(data["image"], tf.float32) / 255.0, data["label"])
    else:
        preprocessing = lambda data: (tf.image.resize(data["image"], image_shape[:2]) / 255.0, data["label"])
    dataset = tfds.load("cifar10", split="train").map(preprocessing, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return dataset

def run_test(input_shape=(32, 32, 3), batch_size=512, use_fp16=True):
    if use_fp16:
        keras.mixed_precision.set_global_policy('mixed_float16')
        # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    dataset = load_cifar10(batch_size=batch_size, image_shape=input_shape)

    # model = keras.applications.MobileNet(include_top=True, classes=10, input_shape=input_shape, weights=None)
    model = keras.applications.ResNet50(include_top=True, classes=10, input_shape=input_shape, weights=None)
    model = keras.models.Model(model.inputs[0], keras.layers.Activation("linear", dtype="float32")(model.outputs[0]))

    # optimizer = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam())
    optimizer = keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    history = model.fit(dataset, epochs=2)

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("-i", "--input_shape", type=int, default=32, help="Input shape")

    args = parse_arguments(sys.argv[1:])
    run_test((args.input_shape, args.input_shape, 3), args.batch_size)
```
```sh
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 112 -b 512
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 112 -b 512 -f
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 128 -b 512
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 128 -b 512 -f
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 112 -b 256
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 112 -b 256 -f

CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 224 -b 512
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 224 -b 512 -f

CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 32 -b 512
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 32 -b 512 -f
```

| input_shape | batch_size | use_fp16 | first epoch (ms/step) | second epoch (ms/step) |
| ----------- | ---------- | -------- | --------------------- | ---------------------- |
| 112         | 512        | False    | 83                    | 67                     |
| 112         | 512        | True     | 1969                  | 1857                   |
| 128         | 512        | False    | 96                    | 85                     |
| 128         | 512        | True     | 96                    | 97                     |
| 112         | 256        | False    | 43                    | 38                     |
| 112         | 256        | True     | 46                    | 41                     |
| 224         | 512        | False    | 233                   | 212                    |
| 224         | 512        | True     | 228                   | 210                    |
| 32          | 512        | False    | 8                     | 5                      |
| 32          | 512        | True     | 38                    | 37                     |

```sh
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 112 -b 384 -f
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 112 -b 385 -f
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 120 -b 512 -f
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 121 -b 512 -f
CUDA_VISIBLE_DEVICES='0' python test_fp16.py -i 122 -b 512 -f
```

| input_shape | batch_size | use_fp16 | first epoch (ms/step) | second epoch (ms/step) |
| ----------- | ---------- | -------- | --------------------- | ---------------------- |
| 112         | 384        | True     | 63                    | 45                     |
| 112         | 385        | True     | 1407                  | 1353                   |
| 120         | 512        | True     | 2306                  | 2183                   |
| 121         | 512        | True     | 90                    | 84                     |
| 122         | 512        | True     | 88                    | 66                     |

**Conv2D, 112, 512, float16**

| kernel_size | padding | first epoch (ms/step) | second epoch (ms/step) |
| ----------- | ------- | --------------------- | ---------------------- |
| 7           | SAME    | 77                    | 60                     |
| 7           | VALID   | 1969                  | 1857                   |
| 5           | SAME    | 1132                  | 1061                   |
| 5           | VALID   | 953                   | 948                    |
| 3           | SAME    | 351                   | 342                    |
| 3           | VALID   | 339                   | 330                    |

**Resnet50, 112, 512**

| padding | float16 | first epoch (ms/step) | second epoch (ms/step) |
| ------- | ------- | --------------------- | ---------------------- |
| VALID   | False   | 608                   | 570                    |
| VALID   | True    | 2622                  | 2397                   |
| SAME    | False   | 691                   | 656                    |
| SAME    | True    | 438                   | 396                    |

## Convert from float32 model
  ```py
  def convert_to_mixed_float16(model):
      policy = keras.mixed_precision.Policy('mixed_float16')
      policy_config = keras.utils.serialize_keras_object(policy)
      from tensorflow.keras.layers import InputLayer, Activation
      from tensorflow.keras.activations import linear

      def do_convert_to_mixed_float16(layer):
          if not isinstance(layer, InputLayer) and not (isinstance(layer, Activation) and layer.activation == linear):
              aa = layer.get_config()
              aa.update({'dtype': policy_config})
              bb = layer.__class__.from_config(aa)
              bb.build(layer.input_shape)
              bb.set_weights(layer.get_weights())
              return bb
          return layer
      return keras.models.clone_model(model, clone_function=do_convert_to_mixed_float16)
  ```
## Resnet
```py
keras.mixed_precision.set_global_policy('mixed_float16')

import data
ds, steps = data.prepare_dataset('/datasets/faces_emore_112x112_folders/', batch_size=512)

model = keras.applications.ResNet50(include_top=True, classes=85742, input_shape=(112, 112, 3), weights=None)
model = keras.models.Model(model.inputs[0], keras.layers.Activation("linear", dtype="float32")(model.outputs[0]))

# optimizer = keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam())
optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# history = model.fit(x_train, y_train, batch_size=1024, epochs=5, validation_split=0.2)
history = model.fit(ds, epochs=2)
```
```py
# float32 100/11373 [..............................] - ETA: 2:07:29
# float16
```
```py
from tensorflow import keras

def test_conv(use_fp16=True, padding="VALID"):
    if use_fp16:
        dtype = keras.mixed_precision.Policy('mixed_float16')
    else:
        dtype = "float32"
    inputs = tf.ones([512, 112, 112, 3])
    nn = keras.layers.Conv2D(64, 7, strides=2, use_bias=False, padding=padding, name="conv1_conv", dtype=dtype)
    print(nn(inputs).dtype)

    %timeit nn(inputs)

test_conv(use_fp16=True, padding="VALID") # 5.62 ms ± 82.2 µs
test_conv(use_fp16=True, padding="SAME")  # 6.61 ms ± 109 µs
test_conv(use_fp16=False, padding="VALID")  # 4.77 ms ± 36.6 µs
test_conv(use_fp16=False, padding="SAME") # 5.14 ms ± 53.9 µs
```
```py
def bench(dtype, data_format):
    if data_format == 'NHWC':
        x = tf.random.normal((512, 112, 112, 3), dtype="float32")
        f = tf.random.normal((7, 7, 3, 64), dtype="float32")
    else:
        x = tf.random.normal((512, 3, 112, 112), dtype="float32")
        f = tf.random.normal((7, 7, 3, 64), dtype="float32")

    p = tf.constant(0.)

    # Warmup
    tf.nn.conv2d(x, f, strides=2, padding='VALID', data_format=data_format)

    start = time.time()
    for _ in range(10):
        tf.nn.conv2d(x, f, strides=2, padding='VALID', data_format=data_format)
    # Synchronize GPU by sending result of computation to CPU
    p = p + 1.
    p.numpy()

    end = time.time()
    print('time for %s %s: %s' % (dtype, data_format, end - start))

bench('float32', 'NHWC')
bench('float32', 'NCHW')
bench('float16', 'NHWC')
bench('float16', 'NCHW')
```
```py
keras.mixed_precision.set_global_policy('mixed_float16')

inputs = tf.ones([512, 112, 112, 3])
nn = keras.layers.Conv2D(64, 7, strides=2, use_bias=False, padding='VALID', name="conv1_conv")
nn.build([112, 112, 3])
print(inputs.dtype, nn.kernel.dtype, nn(inputs).dtype)
%timeit nn(inputs)


nn = keras.layers.Conv2D(64, 7, strides=2, use_bias=False, padding='SAME', name="conv1_conv")
nn.build([112, 112, 3])
nn(inputs).shape
%timeit nn(inputs)
# 7.34 ms ± 54.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
```py
def test_resnet(input_shape=None, classes=1000, padding='VALID'):
    img_input = keras.layers.Input(shape=input_shape)
    nn = img_input

    nn = keras.layers.Conv2D(64, 7, strides=2, use_bias=False, padding=padding, name="conv1_conv")(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(classes, name="predictions")(nn)
    nn = keras.layers.Activation('softmax', dtype='float32')(nn)
    return keras.models.Model(img_input, nn)

mm = test_resnet(classes=10, input_shape=(112, 112, 3), padding='VALID')
inputs = tf.ones([512, 112, 112, 3])
print(inputs.dtype, mm(inputs).dtype)
%timeit mm(inputs)

mm = test_resnet(classes=10, input_shape=(112, 112, 3), padding='SAME')
inputs = tf.ones([512, 112, 112, 3])
print(inputs.dtype, mm(inputs).dtype)
%timeit mm(inputs)
```
***
# XLA Accelerated Linear Algebra
  - [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla)
  ```py
  tf.config.optimizer.set_jit(True)

  @tf.function(jit_compile=True)
  ```
  ```sh
  $ TF_XLA_FLAGS=--tf_xla_auto_jit=2
  ```
***
# Visualizing Data using the Embedding Projector in TensorBoard
  ```py
  import os
  import tensorflow_datasets as tfds
  import tensorflow as tf
  from tensorboard.plugins import projector
  import models

  log_dir='/tmp/embedding-example/'
  model_path = "checkpoints/TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16.h5"
  if not os.path.exists(log_dir):
      os.makedirs(log_dir)

  mm = tf.keras.models.load_model(model_path, custom_objects={"NormDense": models.NormDense}, compile=False)
  checkpoint = tf.train.Checkpoint(embedding=tf.Variable(tf.transpose(mm.layers[-1].weights[0])))
  checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

  with open(os.path.join(log_dir, 'metadata.tsv'), "w") as ff:
      for ii in range(mm.layers[-1].output.shape[-1]):
          ff.write("{}\n".format(ii))
  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
  embedding.metadata_path = 'metadata.tsv'
  projector.visualize_embeddings(log_dir, config)

  !tensorboard --logdir /tmp/embedding-example/
  ```
***
