# ___2019 - 09 - 27 Car Plate Recognition___
***
- [车牌识别（一）-车牌定位](https://www.cnblogs.com/polly333/p/7367479.html)
- [车牌识别（一）——车牌定位(附详细代码及注释）](https://blog.csdn.net/lixiaoyu101/article/details/86626626)
- [车牌识别1：License Plate Detection and Recognition in Unconstrained Scenarios阅读笔记](https://www.cnblogs.com/greentomlee/p/10863363.html)
- [Github HyperLPR 高性能开源中文车牌识别框架](https://github.com/zeusees/HyperLPR)
- [Kaggle Car plates detection and recognition](https://www.kaggle.com/c/car-plates-detection-recognition/data)

# PLATE MASK
## skimage 角点检测
  ```py
  def plate_mask_corner(source_mask):
      mask = source_mask > 0.5
      cc = corner_peaks(corner_harris(mask, method='eps'), min_distance=1, num_peaks=4, exclude_border=False)
      return cc
  ```
## 巻积后按区域划分后的四个最小值点
  ```py
  def plate_mask_to_scatter_2(source_mask, conv_kernel_size=[9, 9]):
      mask = source_mask > 0.5
      cc = convolve2d(mask, np.ones(conv_kernel_size), mode='same')
      bb = np.where(mask, cc, np.zeros_like(cc))
      # print(pd.value_counts(bb.flatten()))

      image_shape = mask.shape[:2]
      half_x = image_shape[1] // 2 + 1
      half_y = image_shape[0] // 2 + 1
      rr = []
      for ixx in [0, half_x]:
          for iyy in [0, half_y]:
              # print("ixx = %d, iyy = %d" % (ixx, iyy))
              sub_bb = bb[iyy : iyy + half_y, ixx : ixx + half_x]
              sub_bb = np.where(sub_bb == 0, np.ones_like(sub_bb) * 255, sub_bb)
              tt_y, tt_x = np.where(sub_bb == sub_bb.min())
              point = [tt_y.mean() + iyy, tt_x.mean() + ixx]
              print("point = %s, sub_bb.min = %d" % (point, sub_bb.min()))
              rr.append(point)
      rr = np.array(rr)
      return rr
  ```
## 取巻积后所有最小值点
  ```py
  def plate_mask_to_scatter(source_mask, conv_kernel_size=[9, 9], corner_min=None, corner_max=None, min_dist=100, num_peaks=4):
      if corner_min == None:
          corner_min = conv_kernel_size[0]
      if corner_max == None:
          corner_max = (conv_kernel_size[0] * conv_kernel_size[1]) // 2
      mask = source_mask > 0.5
      cc = convolve2d(mask, np.ones(conv_kernel_size), mode='same')
      bb = np.where(mask, cc, np.zeros_like(cc))
      # print(pd.value_counts(bb.flatten()).sort_index())
      # print(np.logical_and(bb <corner_ max, bb > corner_min).sum())
      tt = np.where(np.logical_and(bb < corner_max, bb > corner_min))

      ''' Group by dists '''
      tt_coor = np.array(list(zip(tt[0].tolist(), tt[1].tolist())))
      dd = [np.expand_dims(tt_coor[0], 0)]
      base = [tt_coor[0]]
      for ii in tt_coor[1:]:
          dists = ((ii - base) ** 2).sum(-1)
          if np.any(dists < 100):
              id = dists.argmin()
              dd[id] = np.vstack([dd[id], ii])
          else:
              base.append(ii)
              dd.append(np.expand_dims(ii, 0))

      ''' Choose min conv value of each group '''
      rr = [ii[np.argmin([bb[ixx, iyy] for ixx, iyy in ii])] for ii in dd]
      rr = np.vstack(rr)

      ''' Choose by conv value if rr has more poits than <num_peaks> '''
      if num_peaks and rr.shape[0] > num_peaks:
          rr = np.array(sorted(rr.tolist(), key=lambda ii: bb[ii[0], ii[1]])[:num_peaks])
      return rr
  ```
## 测试
  ```py
  def test_with_points(polygon_points, test_func, image_shape=(128, 384), plot_image=True):
      mask = polygon2mask(image_shape, polygon_points)
      rr = test_func(mask)
      if plot_image:
          fig = plt.figure()
          plt.imshow(mask)
          plt.scatter(rr[:, 1], rr[:, 0])
      return rr

  def coord_sort(coord):
      coord_sort_1 = sorted(coord.tolist(), key=lambda ii: ii[0] + ii[1])
      coord_sort_2 = sorted(coord_sort_1[:2], key=lambda ii: ii[0])
      coord_sort_2.extend(sorted(coord_sort_1[2:], key=lambda ii: ii[0]))
      return np.array(coord_sort_2)

  def test_with_masks(path, test_func, test_num=None, image_shape=(128, 384)):
      dists = []
      pps = []
      ccs = []
      tests = os.listdir(path)
      if test_num:
          tests = tests[:test_num]
      for ii in tests:
          pp = np.array(os.path.splitext(ii)[0].split('_')[1:]).astype('float').reshape(-1, 2)[:, ::-1]
          mask = polygon2mask(image_shape, pp)
          rr = test_func(mask)
          pps.append(pp)
          ccs.append(rr)

          pp_sort = coord_sort(pp)
          rr_sort = coord_sort(rr)
          dd = ((pp_sort - rr_sort) ** 2).sum()
          dists.append(dd)
          print('pp = %s, rr = %s, dist = %.2f' % (pp.tolist(), rr.tolist(), dd))
      dda = np.array(dists)
      ppa = np.array(pps)
      cca = np.array(ccs)

      print("Max 10 dist: %s" % (np.sort(dda)[-10:]))
      return ppa, cca, dda

  path = './mask_128_384/'
  ppa, cca, dda = test_with_masks(path, plate_mask_to_scatter)
  # Max 10 dist: [13.36 13.36 14.02 14.02 14.64 14.7  15.4  16.08 16.08 16.14]

  dda.max()
  np.sort(dda)[-10:]
  np.sort(dda)[-50:]
  print(ppa[np.argsort(dda)[-10:]])
  # [[[13.2, 90.2], [59.8, 30.6], [114.8, 293.8], [68.2, 353.4]],
  #  [[13.2, 90.2], [59.8, 30.6], [114.8, 293.8], [68.2, 353.4]],
  #  [[15.1, 88.1], [56.8, 37.5], [112.9, 295.9], [71.2, 346.5]],
  #  [[15.1, 88.1], [56.8, 37.5], [112.9, 295.9], [71.2, 346.5]],
  #  [[10.9, 85.4], [60.8, 25.9], [117.1, 298.6], [67.2, 358.1]],
  #  [[11.9, 80.3], [58.9, 26.8], [116.1, 303.7], [69.1, 357.2]],
  #  [[14.1, 74.1], [53.8, 30.2], [113.9, 309.9], [74.2, 353.8]],
  #  [[12.9, 32.1], [67.6, 83.6], [115.1, 351.9], [60.4, 300.4]],
  #  [[12.9, 32.1], [67.6, 83.6], [115.1, 351.9], [60.4, 300.4]],
  #  [[14.1, 85.1], [60.8, 32.1], [113.9, 298.9], [67.2, 351.9]]]
  ```
***

# Unet 车牌定位
## Unet
  - [Colab Plate_location_3.ipynb](https://colab.research.google.com/drive/1sAvhfn-gZdIxo0_7_ODJx1BvRoXYGkoS)
## Mobile Unet
  - [Tensorflow Tutorials Image segmentation](https://www.tensorflow.org/tutorials/images/segmentation)
  ```py
  base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

  # Use the activations of these layers
  layer_names = [
      'block_1_expand_relu',   # 64x64
      'block_3_expand_relu',   # 32x32
      'block_6_expand_relu',   # 16x16
      'block_13_expand_relu',  # 8x8
      'block_16_project',      # 4x4
  ]
  layers = [base_model.get_layer(name).output for name in layer_names]

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
  down_stack.trainable = False

  def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
      initializer = tf.random_normal_initializer(0., 0.02)
      result = tf.keras.Sequential()
      result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

      if norm_type.lower() == 'batchnorm':
          result.add(tf.keras.layers.BatchNormalization())
      elif norm_type.lower() == 'instancenorm':
          result.add(InstanceNormalization())

      if apply_dropout:
          result.add(tf.keras.layers.Dropout(0.5))
      result.add(tf.keras.layers.ReLU())
      return result

  up_stack = [
      upsample(512, 3),  # 4x4 -> 8x8
      upsample(256, 3),  # 8x8 -> 16x16
      upsample(128, 3),  # 16x16 -> 32x32
      upsample(64, 3),   # 32x32 -> 64x64
  ]
  def unet_model(output_channels):
      # This is the last layer of the model
      last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2,
          padding='same', activation='softmax')  #64x64 -> 128x128

      # Downsampling through the model
      inputs = tf.keras.layers.Input(shape=[128, 128, 3])
      skips = down_stack(inputs)

      x = skips[-1]
      skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
      for up, skip in zip(up_stack, skips):
          x = up(x)
          concat = tf.keras.layers.Concatenate()
          x = concat([x, skip])
      x = last(x)
      return tf.keras.Model(inputs=inputs, outputs=x)
  ```
  ```sh
  Epoch 00050: val_dice_loss improved from 0.01949 to 0.01888, saving model to /content/drive/My Drive/keras_checkpoints
  INFO:tensorflow:Assets written to: /content/drive/My Drive/keras_checkpoints/assets
  3623/3623 [==============================] - 699s 193ms/step - loss: 0.0470 - dice_loss: 0.0180 - val_loss: 0.0515 - val_dice_loss: 0.0189
  Epoch 51/100
  3622/3623 [============================>.] - ETA: 0s - loss: 0.0466 - dice_loss: 0.0179
  Epoch 00051: val_dice_loss did not improve from 0.01888
  3623/3623 [==============================] - 686s 189ms/step - loss: 0.0466 - dice_loss: 0.0179 - val_loss: 0.0521 - val_dice_loss: 0.0192
  Epoch 52/100
  3622/3623 [============================>.] - ETA: 0s - loss: 0.0464 - dice_loss: 0.0178
  Epoch 00052: val_dice_loss did not improve from 0.01888
  3623/3623 [==============================] - 686s 189ms/step - loss: 0.0464 - dice_loss: 0.0178 - val_loss: 0.0515 - val_dice_loss: 0.0190
  Epoch 53/100
   769/3623 [=====>........................] - ETA: 7:56 - loss: 0.0465 - dice_loss: 0.0178
  ```
  ```py
  def plot_image_predict(model, image_path, thresh=0.5, reverse=False, color='g'):
      aa = imread(image_path)
      bb = resize(aa, (128, 384))
      if reverse:
          bb = 1 - bb
      cc = rgb2gray(bb)
      dd = tf.convert_to_tensor([np.expand_dims(cc, -1)])
      ee = model.predict(dd)
      fig = plt.figure()
      plt.imshow(bb)
      plt.contour(ee[0, :, :, 0] > thresh, [0.5], colors=[color])
  ```
***
