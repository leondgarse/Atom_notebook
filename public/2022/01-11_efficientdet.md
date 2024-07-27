# ___2022 - 01 - 11 EfficientDet___
***

# Links
  - [Unet Plus Plus with EfficientNet Encoder](https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder)
  - [mask-rcnn with augmentation and multiple masks](https://www.kaggle.com/abhishek/mask-rcnn-with-augmentation-and-multiple-masks)
  - [Object detection: Bounding box regression with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)
  - [R-CNN object detection with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/)
  - [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)
  - [Github google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet)
  - [Github zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
  - [keras.io/examples Object Detection with RetinaNet](https://keras.io/examples/vision/retinanet/)
***

# COCO dataset
## dataset basic info
  ```py
  COCO_LABELS = """person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign,
      parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie,
      suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket,
      bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut,
      cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
      toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush"""
  COCO_LABEL_DICT = {id: ii.strip() for id, ii in enumerate(COCO_LABELS.split(","))}

  import tensorflow_datasets as tfds
  ds, info = tfds.load('coco/2017', with_info=True)
  aa = ds['train'].as_numpy_iterator().next()
  print(aa.keys())
  # dict_keys(['image', 'image/filename', 'image/id', 'objects'])
  print(aa['image'].shape)
  # (462, 640, 3)
  print(aa['objects'])
  # {'area': array([17821, 16942,  4344]),
  #  'bbox': array([[0.54380953, 0.13464062, 0.98651516, 0.33742186],
  #      [0.50707793, 0.517875  , 0.8044805 , 0.891125  ],
  #      [0.3264935 , 0.36971876, 0.65203464, 0.4431875 ]], dtype=float32),
  #  'id': array([152282, 155195, 185150]),
  #  'is_crowd': array([False, False, False]),
  #  'label': array([3, 3, 0])
  # }

  # id already processed starting from `0`, and `0` presents `person`.
  ee = []
  for ii in ds['train'].take(10000):
      ee.extend(ii['objects']['label'])
  ee = np.array(ee)
  print(f"{ee.min() = }, {ee.max() = }, {np.unique(ee).shape = }")
  # ee.min() = 0, ee.max() = 79, np.unique(ee).shape = (80,)

  from tqdm import tqdm
  bb = [ii['objects']['bbox'].shape[0] for ii in tqdm(dataset["train"])]
  pd.value_counts(bb).sort_index()
  # 0      1021
  # 1     13893
  # 2     21391
  # ...
  # 78        1
  # 80        1
  # 93        1
  pd.value_counts(bb).sort_index().plot()
  ```
## Show example
  ```py
  from keras_cv_attention_models.coco.data import COCO_LABEL_DICT

  def coco_show(sample, ax=None):
      if ax is None:
          fig, ax = plt.subplots()
      imm = sample['image']
      ax.imshow(imm)
      for bb, label, is_crowd in zip(sample['objects']['bbox'], sample['objects']['label'], sample['objects']['is_crowd']):
          if is_crowd:
              continue
          # bbox is [top, left, bottom, right]
          ss = np.array([bb[0] * imm.shape[0], bb[1] * imm.shape[1], bb[2] * imm.shape[0], bb[3] * imm.shape[1]])
          ax.plot(ss[[1, 1, 3, 3, 1]], ss[[0, 2, 2, 0, 0]])

          label = int(label)
          color = ax.lines[-1].get_color()
          ax.text(ss[1], ss[0] - 5, "{}, {}".format(label, COCO_LABEL_DICT[label]), color=color, fontweight="bold")
      ax.set_axis_off()
      plt.tight_layout()
      return sample['objects']

  import tensorflow_datasets as tfds
  ds, info = tfds.load('coco/2017', with_info=True)
  _ = coco_show(ds['train'].shuffle(100).as_numpy_iterator().next())
  ```
  ![](images/coco_example.png)
***

# Init dataset
## Anchors
  ```py
  sys.path.append('../automl/efficientdet/')
  from tf2 import anchors
  aa = anchors.Anchors(min_level=3, max_level=3, num_scales=3, aspect_ratios=[1.0, 2.0, 0.5], anchor_scale=4, image_size=8)
  bb = aa._generate_boxes()

  """ Show basic anchors """
  from keras_cv_attention_models.coco import data
  data.draw_bboxes(bb)

  """ All anchors should match """
  cc = anchors.Anchors(min_level=3, max_level=7, num_scales=3, aspect_ratios=[1.0, 2.0, 0.5], anchor_scale=4, image_size=512)._generate_boxes()
  dd = data.get_anchors()
  print(f"{cc.shape = }, {dd.shape = }, {np.allclose(cc, dd * 512) = }")
  # cc.shape = TensorShape([49104, 4]), dd.shape = TensorShape([49104, 4]), np.allclose(cc, dd * 512) = True
  ```
  ![](images/draw_bboxes.png)
  ```py
  import tensorflow_datasets as tfds
  ds, info = tfds.load('coco/2017', with_info=True)
  aa = ds['train'].as_numpy_iterator().next()
  imm, bboxes, labels = aa['image'], aa['objects']['bbox'], aa['objects']['label']

  from keras_cv_attention_models.coco import data
  anchors = data.get_anchors()
  rr = data.assign_anchor_classes_by_iou_with_bboxes(bboxes, anchors, labels)
  print(f"{rr.shape = }, {rr[rr[:, -1] >= 0].shape = }")
  # rr.shape = TensorShape([49104, 5]), rr[rr[:, -1] >= 0].shape = TensorShape([59, 5])

  def decode_bboxes(preds, anchors):
      bboxes, label = preds[:, :4], preds[:, 4:]
      anchors_wh = anchors[:, 2:] - anchors[:, :2]
      anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

      bboxes_center = bboxes[:, :2] * anchors_wh + anchors_center
      bboxes_wh = np.exp(bboxes[:, 2:]) * anchors_wh

      preds_left_top = bboxes_center - 0.5 * bboxes_wh
      pred_right_bottom = preds_left_top + bboxes_wh
      return np.concatenate([preds_left_top, pred_right_bottom, label], axis=-1)


  valid_encoded, valid_anchors = rr[rr[:, -1] >= 0], anchors[rr[:, -1] >= 0]
  valid_encoded = decode_bboxes(valid_encoded, valid_anchors)

  fig, axes = plt.subplots(1, 2)
  data.show_image_with_bboxes(imm, valid_anchors[:, :4], valid_encoded[:, -1], ax=axes[0])
  data.show_image_with_bboxes(imm, valid_encoded[:, :4], valid_encoded[:, -1], ax=axes[1])
  axes[0].set_title('raw anchors')
  axes[1].set_title('decoded anchors')
  fig.tight_layout()
  ```
  ![](images/decode_anchors.png)
## get feature size
  ```py
  # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
  def get_feat_sizes(image_size, max_level):
      """Get feat widths and heights for all levels.

      Args:
        image_size: A integer, a tuple (H, W), or a string with HxW format.
        max_level: maximum feature level.

      Returns:
        feat_sizes: a list of tuples (height, width) for each level.
      """
      feat_sizes = [(image_size[0], image_size[1])]
      feat_size = image_size
      for _ in range(1, max_level + 1):
          feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
          feat_sizes.append((feat_size[0], feat_size[1]))
      return feat_sizes
  feat_sizes = get_feat_sizes([228, 228], 7)
  strides = [feat_sizes[0][0] / feat_sizes[ii][0] for ii in range(3, 7 + 1)]
  print(f"{feat_sizes = }, {strides = }")
  # feat_sizes = [(228, 228), (114, 114), (57, 57), (29, 29), (15, 15), (8, 8), (4, 4), (2, 2)], strides = [7.862068965517241, 15.2, 28.5, 57.0, 114.0]
  ```
  ```py
  input_shape, pyramid_levels = (228, 228, 3), [3, 7]
  pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))

  feat_sizes = [input_shape[:2]]
  for _ in range(max(pyramid_levels)):
      pre_feat_size = feat_sizes[-1]
      feat_sizes.append(((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1))
  ```
## Random
  ```py
  def random_crop(height, width, jitter=0.3):
      new_area = width / height * np.random.uniform(1 - jitter, 1 + jitter) / np.random.uniform(1 - jitter, 1 + jitter)
      scale = np.random.uniform(0.25, 2)
      if new_area < 1:
          new_height = int(scale * height)
          new_width = int(new_height * new_area)
      else:
          new_width = int(scale * width)
          new_height = int(new_width / new_area)
      return new_height, new_width

  aa = np.array([random_crop(100, 100) for _ in range(100000)])
  hhs, wws = aa[:, 0], aa[:, 1]

  # scale: (scale * height) ** 2 * new_area, (scale * width) ** 2 * new_area,
  print("Scale range:", ((hhs * wws).min() / 1e4, (hhs * wws).max() / 1e4))
  # Scale range: (0.035, 3.9402)

  # ratio: new_area, 1 / new_area --> (0.7/1.3, 1.3/0.7) ~= (0.54, 1.85),
  print("Ratio range:", ((wws / hhs).min(), (wws / hhs).max()))
  # Ratio range: (0.5172413793103449, 1.9411764705882353)
  ```
## Dataset tests
  ```py
  import tensorflow_datasets as tfds
  from keras_cv_attention_models.coco import data
  data_name, input_shape, batch_size, buffer_size = "coco/2017", (224, 224), 16, 1000

  dataset, info = tfds.load(data_name, with_info=True)
  num_classes = info.features['objects']["label"].num_classes
  total_images = info.splits["train"].num_examples
  steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))

  AUTOTUNE = tf.data.AUTOTUNE
  anchors = data.get_anchors(input_shape[:2]).astype("float32")
  num_anchors = anchors.shape[0]
  empty_label = tf.concat([tf.zeros([num_anchors, 4]), tf.zeros([num_anchors, 1]) - 1], axis=-1)

  magnitude = 0
  train_process = data.RandomProcessImage(target_shape=input_shape, magnitude=magnitude)
  bbox_process = lambda bbox, label: tf.cond(
      tf.shape(bbox)[0] == 0,
      lambda: empty_label,
      lambda: data.assign_anchor_classes_by_iou_with_bboxes(bbox, anchors, label),
  )
  train_dataset = dataset["train"].map(train_process).map(lambda xx, yy: (xx, bbox_process(yy[0], yy[1])))
  train_dataset = train_dataset.batch(batch_size)

  rescale_mode = "tf"
  mean, std = data.init_mean_std_by_rescale_mode(rescale_mode)
  # rescaling = lambda xx: (tf.clip_by_value(xx, 0, 255) - mean) / std
  rescaling = lambda xx: (xx - mean) / std
  train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), yy), num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
  ```
  ```py
  from keras_cv_attention_models.coco import data
  import tensorflow_datasets as tfds
  data_name, input_shape, batch_size, buffer_size = "coco/2017", (224, 224), 16, 1000
  dataset, info = tfds.load(data_name, with_info=True)

  anchors = data.get_anchors(input_shape[:2]).astype("float32")
  num_anchors = anchors.shape[0]
  empty_label = tf.zeros([num_anchors, 5])
  train_process = data.RandomProcessImage(target_shape=input_shape, magnitude=5)
  bbox_process = lambda bbox, label: tf.cond(
      tf.shape(bbox)[0] == 0,
      lambda: empty_label,
      lambda: data.assign_anchor_classes_by_iou_with_bboxes(bbox, anchors, label),
  )
  for id, ii in enumerate(dataset['train']):
      print(id)
      # imm, bbox, label = ii['image'], ii['objects']['bbox'], ii['objects']['label']
      imm, (bbox, label) = train_process(ii)
      yy = bbox_process(bbox, label)
      if np.isnan(yy).sum() != 0:
          break
  ```
## automl training aspect aware random crop
  ```py
  def get_random_image_scale(source_shape, target_shape, scale_min=0.1, scale_max=2.0):
      random_scale_factor = tf.random.uniform([], scale_min, scale_max)
      scaled_y, scaled_x = random_scale_factor * target_shape[0], random_scale_factor * target_shape[1]
      height, width = tf.cast(source_shape[0], tf.float32), tf.cast(source_shape[1], tf.float32)
      return tf.minimum(scaled_y / height, scaled_x / width)

  def get_image_aspect_aware_random_scale_crop(source_shape, target_shape, scale_min=0.1, scale_max=2.0):
      """ https://github.com/google/automl/tree/master/efficientdet/dataloader.py#L67 """
      random_image_scale = get_random_image_scale(source_shape, target_shape, scale_min, scale_max)

      # Select non-zero random offset (x, y) if scaled image is larger than self._output_size.
      height, width = tf.cast(source_shape[0], tf.float32), tf.cast(source_shape[1], tf.float32)
      scaled_height, scaled_width = height * random_image_scale, width * random_image_scale
      offset_y, offset_x = tf.maximum(0.0, scaled_height - target_shape[0]), tf.maximum(0.0, scaled_width - target_shape[1])
      random_offset_y, random_offset_x = offset_y * tf.random.uniform([], 0, 1), offset_x * tf.random.uniform([], 0, 1)
      random_offset_y, random_offset_x = tf.cast(random_offset_y, tf.int32), tf.cast(random_offset_x, tf.int32)
      return random_image_scale, random_offset_y, random_offset_x

  def aspect_aware_resize_and_crop_image(image, target_shape, scale=-1, crop_y=0, crop_x=0, method="bilinear", antialias=False):
      if scale == -1:
          scale = tf.minimum(target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
      scaled_hh, scaled_ww = int(image.shape[0] * scale), int(image.shape[1] * scale)
      image = tf.image.resize(image, [scaled_hh, scaled_ww], method=method, antialias=antialias)
      image = image[crop_y : crop_y + target_shape[0], crop_x : crop_x + target_shape[1]]
      image = tf.image.pad_to_bounding_box(image, 0, 0, target_shape[0], target_shape[1])
      return image, scale
  ```
***

# EfficientDet build
## Automl definition
  ```py
  # cd automl/efficientdet/
  import hparams_config
  from tf2 import efficientdet_keras, util_keras
  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  model = efficientdet_keras.EfficientDetNet(config=config)
  model_name = config.name
  # model = efficientdet_keras.EfficientDetModel(config=config)
  model.build((None, config.image_size, config.image_size, 3))
  # model.load_weights(tf.train.latest_checkpoint(model_name))
  util_keras.restore_ckpt(model, tf.train.latest_checkpoint(model_name), skip_mismatch=False)
  model.save_weights(model_name + ".h5")

  inputs = keras.layers.Input([config.image_size, config.image_size, 3])
  mm = keras.models.Model(inputs, model.call(inputs, training=False))
  {ii.name: ii.shape for ii in mm.outputs}
  # {'class_net/class-predict/BiasAdd:0': TensorShape([None, 64, 64, 810]),
  #  'class_net/class-predict/BiasAdd_1:0': TensorShape([None, 32, 32, 810]),
  #  'class_net/class-predict/BiasAdd_2:0': TensorShape([None, 16, 16, 810]),
  #  'class_net/class-predict/BiasAdd_3:0': TensorShape([None, 8, 8, 810]),
  #  'class_net/class-predict/BiasAdd_4:0': TensorShape([None, 4, 4, 810]),
  #  'box_net/box-predict/BiasAdd:0': TensorShape([None, 64, 64, 36]),
  #  'box_net/box-predict/BiasAdd_1:0': TensorShape([None, 32, 32, 36]),
  #  'box_net/box-predict/BiasAdd_2:0': TensorShape([None, 16, 16, 36]),
  #  'box_net/box-predict/BiasAdd_3:0': TensorShape([None, 8, 8, 36]),
  #  'box_net/box-predict/BiasAdd_4:0': TensorShape([None, 4, 4, 36])}
  #   cls_out_list, box_out_list = model(inputs, training=False)

  """ Backbone """
  print(f"{model.backbone.name = }")
  # model.backbone.name = 'efficientnet-b0'
  bb = keras.models.Model(inputs, model.backbone.call(inputs, training=False))
  {ii.name: ii.shape for ii in bb.outputs}
  # {'head/dense/BiasAdd:0': TensorShape([None, 1000]),
  #  'blocks_0/Identity_1:0': TensorShape([None, 256, 256, 16]),
  #  'blocks_2/Add:0': TensorShape([None, 128, 128, 24]),
  #  'blocks_4/Add:0': TensorShape([None, 64, 64, 40]),
  #  'blocks_10/Add:0': TensorShape([None, 32, 32, 112]),
  #  'blocks_15/Identity_2:0': TensorShape([None, 16, 16, 320])}

  """ fpn_cell """
  inputs = [keras.layers.Input([ii, ii, cc]) for ii, cc in [[64, 40], [32, 112], [16, 320], [8, 64], [4, 64]]]
  feats = inputs
  fpn_cells = mm.get_layer('fpn_cells')
  cell = fpn_cells.cells[0]
  for fnode in cell.fnodes:
      feats = fnode.call(feats, training=True)
  cell_feats = feats
  min_level = fpn_cells.config.min_level
  max_level = fpn_cells.config.max_level

  feats = []
  for level in range(min_level, max_level + 1):
    for i, fnode in enumerate(reversed(fpn_cells.fpn_config.nodes)):
      if fnode['feat_level'] == level:
        feats.append(cell_feats[-1 - i])
        break
  cc = keras.models.Model(inputs, feats)
  {ii.name: ii.shape for ii in cc.outputs}
  # {'op_after_combine8/bn/FusedBatchNormV3:0': TensorShape([None, 64, 64, 64]),
  #  'op_after_combine9/bn/FusedBatchNormV3:0': TensorShape([None, 32, 32, 64]),
  #  'op_after_combine10/bn/FusedBatchNormV3:0': TensorShape([None, 16, 16, 64]),
  #  'op_after_combine11/bn/FusedBatchNormV3:0': TensorShape([None, 8, 8, 64]),
  #  'op_after_combine12/bn/FusedBatchNormV3:0': TensorShape([None, 4, 4, 64])}
  ```
## EfficientDet build
  ```py
  from keras_cv_attention_models import efficientnet
  mm = efficientnet.EfficientNetV1B0()

  """ Pick all stack output layers """
  dd = {}
  for ii in mm.layers:
      match = re.match("^stack_?(\\d+)_block_?(\\d+)_output$", ii.name)
      if match is not None:
          cur_stack = "stack_" + match[1] + "_output"
          dd.update({cur_stack: ii})

  """ Filter those have same downsample rate """
  ee = {str(vv.output_shape[1]): vv for kk, vv in dd.items()}
  {ii.name: ii.output_shape for ii in ee.values()}
  # {'stack_0_block0_output': (None, 112, 112, 16),
  #  'stack_1_block1_output': (None, 56, 56, 24),
  #  'stack_2_block1_output': (None, 28, 28, 40),
  #  'stack_4_block2_output': (None, 14, 14, 112),
  #  'stack_6_block0_output': (None, 7, 7, 320)}

  """ Selected features """
  features = list(ee.values())[1:]
  ```
  ```py
  def bi_fpn_5(features, output_channel, activation="swish", name=""):
      print(f">>>> bi_fpn: {[ii.shape for ii in features] = }")
      p3, p4, p5, p6, p7 = features
      p6_up = resample_fusion([p6, p7], output_channel, activation=activation, name=name + "p6_up_")
      p5_up = resample_fusion([p5, p6_up], output_channel, activation=activation, name=name + "p5_up_")
      p4_up = resample_fusion([p4, p5_up], output_channel, activation=activation, name=name + "p4_up_")
      p3_out = resample_fusion([p3, p4_up], output_channel, activation=activation, name=name + "p3_up_")

      p4_out = resample_fusion([p4, p4_up, p3_out], output_channel, activation=activation, name=name + "p4_out_")
      p5_out = resample_fusion([p5, p5_up, p4_out], output_channel, activation=activation, name=name + "p5_out_")
      p6_out = resample_fusion([p6, p6_up, p5_out], output_channel, activation=activation, name=name + "p6_out_")
      p7_out = resample_fusion([p7, p6_out], output_channel, activation=activation, name=name + "p7_out_")
      return [p3_out, p4_out, p5_out, p6_out, p7_out]
  ```
## Recompute grad
  - [reducing-memory-usage-when-training-efficientdets-on-gpu](https://github.com/google/automl/tree/master/efficientdet#11-reducing-memory-usage-when-training-efficientdets-on-gpu)
  ```py
  def recompute_grad(recompute=False):
      """Decorator determine whether use gradient checkpoint.
      """

      def _wrapper(f):
          if recompute:
              return tf.recompute_grad(f)
          return f

      return _wrapper
  ```
***

# Convert and predict
## Convert weights
  ```py
  from keras_cv_attention_models.efficientdet import convert_efficientdet as efficientdet
  from keras_cv_attention_models.model_surgery import model_surgery

  idx = 0
  # bb = efficientdet.EfficientNetV1B0(input_shape=(512, 512, 3), num_classes=0, output_conv_filter=0)
  # bb = efficientdet.EfficientDet(bb, num_classes=90)
  # bb = getattr(efficientdet, "EfficientDetD{}".format(idx))(pretrained=None)
  bb = getattr(efficientdet, "EfficientDetD{}".format(idx))(pretrained=None, rescale_mode='tf') # Det-AdvProp model
  # backbone = getattr(efficientdet, "EfficientNetV1B{}".format(idx))(input_shape=(320, 320, 3), output_conv_filter=0, se_ratios=[0] * 10, is_fused=False, num_classes=0, activation="relu6", pretrained=None)
  # bb = getattr(efficientdet, "EfficientDetD{}".format(idx))(backbone=backbone, activation="relu6", anchor_scale=3, use_weighted_sum=False, pretrained=None)
  # bb = getattr(efficientdet, "EfficientDetLite{}".format(idx))()

  target_names = [ii.name for ii in bb.layers if len(ii.weights) != 0]
  aa = {bb.get_layer(ii).name: [jj.shape.as_list() for jj in bb.get_layer(ii).weights] for ii in target_names}
  _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]

  """ Load h5 weights, flatten to dict format {layer_name: weights} """
  import h5py
  ff = h5py.File("efficientdet-d{}.h5".format(idx), mode="r")
  # ff = h5py.File("efficientdet-lite{}.h5".format(idx), mode="r")

  def get_weights_recursion(value, pre_name=[]):
      if not isinstance(value, h5py.Group):
          return [("/".join(pre_name), value)]
      else:
          aa = []
          for kk in value.keys():
              aa.extend(get_weights_recursion(value[kk], pre_name+[kk]))
          return aa

  aa = get_weights_recursion(ff["efficientnet-b{}".format(idx)]["efficientnet-b{}".format(idx)])
  # aa = get_weights_recursion(ff["efficientnet-lite{}".format(idx)]["efficientnet-lite{}".format(idx)])
  aa.extend(get_weights_recursion(ff['resample_p6']['resample_p6']))
  # {'bn': [(64,), (64,), (64,), (64,)], 'conv2d': [(64,), (1, 1, 320, 64)]}
  aa.extend(get_weights_recursion(ff['fpn_cells']['fpn_cells']))
  aa.extend(get_weights_recursion(ff['box_net']['box_net']))
  aa.extend(get_weights_recursion(ff['class_net']['class_net']))

  """ Stack weights by layer name """
  ss = {}
  for ii in aa:
      split_name = ii[0].split("/")
      layer_name, weight_name = "_".join(split_name[:-1]), split_name[-1]
      ss.setdefault(layer_name, {}).update({weight_name: ii[1]})
  {kk : [ss[kk][ii].shape for ii in ss[kk]] for kk in ss}

  """ Reload weights """
  for ii in target_names:
      print(ii)
      ww = ss[ii]
      tt = bb.get_layer(ii)
      if isinstance(tt, efficientdet.ReluWeightedSum):
          tt.set_weights([np.array([tf.convert_to_tensor(ww[ii]) for ii in ww])])
      else:
          tt.set_weights([tf.convert_to_tensor(ww[ii.name.split('/')[-1]]).numpy() for ii in tt.weights])

  bb.save(bb.name.lower() + ".h5")
  # bb.save("efficientdet_lite{}.h5".format(idx))

  """ Run prediction """
  from keras_cv_attention_models.coco import data
  from keras_cv_attention_models import test_images
  imm = test_images.dog()
  bbs, ccs = bb.decode_predictions(bb(bb.preprocess_input(imm))[0])
  data.show_image_with_bboxes(imm, bbs, ccs, num_classes=90)
  ```
  **Tests**
  ```py
  inputs = [keras.layers.Input([ii, ii, cc]) for ii, cc in [[64, 40], [32, 112], [16, 320], [8, 64], [4, 64]]]
  fpn_features = inputs
  for id in range(3):
      fpn_features = efficientdet.bi_fpn(fpn_features, 64, name="cell_{}_".format(id))
  bb = keras.models.Model(inputs, fpn_features)
  target_names = [ii.name for ii in bb.layers if len(ii.weights) != 0]
  aa = {bb.get_layer(ii).name: [jj.shape.as_list() for jj in bb.get_layer(ii).weights] for ii in target_names}
  _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
  ```
  ```py
  inputs = [keras.layers.Input([ii, ii, cc]) for ii, cc in [[64, 64], [32, 64], [16, 64], [8, 64], [4, 64]]]
  head_depth, num_anchors, activation = 3, 9, "swish"
  bbox_regressor = efficientdet.detector_head(inputs, 64, head_depth, 4, num_anchors, activation, head_activation=None, name="box-")
  bb = keras.models.Model(inputs, bbox_regressor)
  target_names = [ii.name for ii in bb.layers if len(ii.weights) != 0]
  aa = {bb.get_layer(ii).name: [jj.shape.as_list() for jj in bb.get_layer(ii).weights] for ii in target_names}
  _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
  ```
## Predict
  ```py
  from keras_cv_attention_models.coco import data
  from keras_cv_attention_models import test_images
  imm = test_images.dog()
  rr = bb(tf.expand_dims(keras.applications.imagenet_utils.preprocess_input(imm, mode='torch'), 0))[0].numpy()

  anchors = data.get_anchors()
  dd = data.decode_bboxes(rr, anchors).numpy()
  cc = dd[dd[:, 4:].max(-1) > 0.3]
  rr = tf.image.non_max_suppression(cc[:, :4], cc[:, 4:].max(-1), max_output_size=15, iou_threshold=0.5)
  cc_nms = tf.gather(cc, rr).numpy()
  bboxes, labels = cc_nms[:, :4], cc_nms[:, 4:].argmax(-1)
  data.show_image_with_bboxes(imm, bboxes, labels, num_classes=90)
  ```
  ```py
  from keras_cv_attention_models.coco import data
  from keras_cv_attention_models import test_images
  imm = test_images.dog()
  bbs, ccs = bb.decode_predictions(bb(bb.preprocess_input(imm))[0])
  data.show_image_with_bboxes(imm, bbs, ccs, num_classes=90)
  ```
## Reload
  ```py
  from keras_cv_attention_models.efficientdet import efficientdet
  idx = 0
  # mm = getattr(efficientdet, "EfficientDetD{}".format(idx))(pretrained=None)
  mm = getattr(efficientdet, "EfficientDetLite{}".format(idx))(pretrained=None)
  mm.load_weights(mm.name + '.h5')

  from keras_cv_attention_models.coco import data
  from keras_cv_attention_models import test_images
  imm = test_images.dog_cat()
  bboxs, lables, confidences = mm.decode_predictions(mm(mm.preprocess_input(imm)))[0]
  data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=90)

  mm.save("{}_{}_coco.h5".format(mm.name, mm.input_shape[1]))
  ```
## Comparing output
  ```py
  sys.path.append('../automl/efficientdet/')
  import hparams_config
  import inference
  from tf2 import efficientdet_keras

  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  # config.nms_configs.score_thresh = 0.4
  # model = efficientdet_keras.EfficientDetNet(config=config)
  model = efficientdet_keras.EfficientDetModel(config=config)
  model.build((None, 512, 512, 3))
  model.load_weights(tf.train.latest_checkpoint("../automl/efficientdet/efficientdet-d0/"))
  model.summary(expand_nested=True)
  model.save_weights("efficientdet-d0.h5")

  def merge_class_box_level_outputs(cls_outputs, box_outputs, min_level=3, max_level=7, num_classes=90):
      """Concatenates class and box of all levels into one tensor."""
      cls_outputs_all, box_outputs_all = [], []
      batch_size = tf.shape(cls_outputs[0])[0]
      for level in range(0, max_level - min_level + 1):
          cls_outputs_all.append( tf.reshape(cls_outputs[level], [batch_size, -1, num_classes]))
          box_outputs_all.append(tf.reshape(box_outputs[level], [batch_size, -1, 4]))
      ccs, bbs = tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)
      ccs = tf.nn.sigmoid(ccs)
      return tf.concat([bbs, ccs], axis=-1)

  aa = model.call(tf.ones([1, 512, 512, 3]), pre_mode=None, post_mode=None)
  out = merge_class_box_level_outputs(aa[0], aa[1]).numpy()

  # efficientdet_d0.h5 is converted from efficientdet-d0.h5
  from keras_cv_attention_models import efficientdet
  mm = efficientdet.EfficientDetD0(pretrained=None)
  mm.load_weights('efficientdet_d0.h5')
  bb = mm(tf.ones([1, 512, 512, 3]))
  np.allclose(bb, out, atol=1e-6)
  # True
  ```
***

# Train
  - **automl reported loss** 'box_loss': 0.003652954, 'cls_loss': 0.35299847, 'loss': 0.6351326
  ```py
  from keras_cv_attention_models import efficientnet, efficientdet
  from keras_cv_attention_models.coco import data, losses
  from keras_cv_attention_models.imagenet import train_func

  strategy = train_func.init_global_strategy()

  input_shape, batch_size, basic_save_name, initial_epoch = (256, 256, 3), 32, "effd0_test", 0
  lr_base_512, weight_decay, lr_decay_steps, lr_min, lr_decay_on_batch, lr_warmup = 8e-3, 0.02, 32, 1e-6, False, 1e-4
  warmup_steps, cooldown_steps = 3, 3
  train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
      input_shape=input_shape,
      batch_size=batch_size,
      resize_method="bicubic",
      resize_antialias=True,
      random_crop_min=0.08,
      magnitude=6,
  )

  backbone = efficientnet.EfficientNetV1B0(input_shape=input_shape, num_classes=0)
  model = efficientdet.EfficientDet(backbone, freeze_backbone=True, num_classes=80)

  lr_base = lr_base_512 * batch_size / 512
  lr_scheduler, lr_total_epochs = train_func.init_lr_scheduler(lr_base, lr_decay_steps, lr_min, lr_decay_on_batch, lr_warmup, warmup_steps, cooldown_steps)
  epochs = lr_total_epochs

  model = train_func.compile_model(model, "adamw", lr_base, weight_decay, loss=losses.FocalLossWithBbox(), metrics=losses.ClassAccuracyWithBbox())
  latest_save, hist = train_func.train(model, epochs, train_dataset, test_dataset, initial_epoch, lr_scheduler, basic_save_name)
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py -b 64 -p adamw --backbone efficientnet.EfficientNetV2B0
  ```
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "V1B0, bs 64": "checkpoints/EfficientDet_EfficientNetV1B0_256_coco_2017_adamw_batchsize_64_randaug_6_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
      "V2B0, bs 64": "checkpoints/EfficientDet_EfficientNetV2B0_256_coco_2017_adamw_batchsize_64_randaug_6_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
      # "V1B0, bs 8": "checkpoints/EfficientDet_EfficientNetV1B0_256_adamw_coco_2017_batchsize_8_randaug_6_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
      "V1B0, mosaic 0.5, bs 64": "checkpoints/EfficientDetD0_256_mosaic_05_bs_64_hist.json",
      "V1B0, mosaic 0.5, anchor free, bs 64": "checkpoints/EfficientDetD0_256_mosaic_05_bs_64_anchor_free_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=3, base_size=8)
  ```

  | Model            | Epoch   | Trian loss, cls_loss, bbox_loss | Train cls_acc | Val loss, cls_acc |
  | ---------------- | ------- | ------------------------------- | ------------- | ----------------- |
  | EfficientNetV1B0 | 96/105  | 0.4975, 0.43799, 0.005196       | 0.8616        | 0.5791, 0.8380    |
  | - mosaic 0.5     | 104/105 | 0.6095, 0.33107, 0.004444       | 0.7989        | 0.5911, 0.8419    |
  | EfficientNetV2B0 | 87/105  | 0.4419, 0.32617, 0.004364       | 0.8874        | 0.5865, 0.8420    |

  - **Test**
  ```py
  from keras_cv_attention_models import efficientdet, efficientnet, yolox

  # model = efficientdet.EfficientDetD0(pretrained="checkpoints/EfficientDet_EfficientNetV1B0_256_coco_2017_adamw_batchsize_64_randaug_6_RRC_0.08_lr512_0.008_wd_0.02_epoch_96_val_acc_0.8380.h5", num_classes=80, input_shape=[256, 256, 3])
  backbone = efficientnet.EfficientNetV2B0(input_shape=(256, 256, 3), num_classes=0)
  pretrained = "checkpoints/EfficientDet_EfficientNetV2B0_256_coco_2017_adamw_batchsize_64_randaug_6_RRC_0.08_lr512_0.008_wd_0.02_epoch_39_val_acc_0.7912.h5"
  model = efficientdet.EfficientDetD0(backbone=backbone, pretrained=pretrained, num_classes=80)
  # model = yolox.YOLOXTiny(pretrained='checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_8_randaug_6_mosaic_0.5_RRC_1.0_lr512_0.008_wd_0.02_epoch_19_val_loss_5.3121.h5', input_shape=(256, 256, 3), rescale_mode='torch')

  # Run prediction
  from keras_cv_attention_models import test_images
  imm = test_images.dog_cat()
  bboxs, lables, confidences = model.decode_predictions(model(model.preprocess_input(imm)))[0]

  # Show result
  from keras_cv_attention_models.coco import data
  data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
  ```py
  from keras_cv_attention_models import efficientdet

  model = efficientdet.EfficientDetD0(pretrained="checkpoints/EfficientDetD0_256_mosaic_05_bs_64_epoch_70_val_acc_0.8274.h5", num_classes=80, input_shape=[256, 256, 3])

  # Run prediction
  from keras_cv_attention_models import test_images
  imm = test_images.dog_cat()
  bboxs, lables, confidences = model.decode_predictions(model(model.preprocess_input(imm)))[0]

  # Show result
  from keras_cv_attention_models.coco import data
  data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
***

# Evaluation
## automl det model evaluation
  ```py
  import hparams_config
  import inference
  from tf2 import efficientdet_keras
  from tf2 import util_keras

  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  config.nms_configs.max_nms_inputs = 5000
  config.mean_rgb, config.stddev_rgb, config.scale_range = 0.0, 1.0, True
  model = efficientdet_keras.EfficientDetNet(config=config)
  # model = efficientdet_keras.EfficientDetModel(config=config)
  model.build((None, config.image_size, config.image_size, 3))
  # model.load_weights(tf.train.latest_checkpoint("efficientdet-d0/"))
  util_keras.restore_ckpt(model, tf.train.latest_checkpoint("efficientdet-d0/"), skip_mismatch=False)

  import coco_metric
  import dataloader
  from tf2 import postprocess

  evaluator = coco_metric.EvaluationMetric(filename="annotations/instances_val2017.json")
  batch_size = 1
  ds = dataloader.InputReader("tfrecord/val*", is_training=False, max_instances_per_image=100)(config, batch_size=batch_size)

  from tqdm import tqdm
  for i, (images, labels) in tqdm(enumerate(ds), total=5000):
      cls_outputs, box_outputs = model(images, training=False)
      detections = postprocess.generate_detections(config, cls_outputs, box_outputs, labels['image_scales'], labels['source_ids'])
      detections = postprocess.transform_detections(detections).numpy()
      evaluator.update_state(groundtruth_data=labels['groundtruth_data'].numpy(), detections=detections)
  metrics = evaluator.result()
  ```
## eval func
  ```py
  from keras_cv_attention_models.coco import eval_func
  from keras_cv_attention_models import efficientdet

  model = efficientdet.EfficientDetD0(pretrained="coco")
  eval_func.run_coco_evaluation(model)

  """ sub steps """
  input_shape = model.input_shape[1:-1]
  ds = eval_func.init_eval_dataset(target_shape=input_shape).take(10)
  pred_decoder = model.decode_predictions if hasattr(model, "decode_predictions") else DecodePredictions(input_shape)
  detection_results = eval_func.model_eval_results(model, ds, pred_decoder)
  eval_func.coco_evaluation(detection_results)
  ```
  ```py
  to_coco_json = lambda xx: {"image_id": int(xx[0]), "bbox": xx[1:5].tolist(), "score": float(xx[5]), "category_id": int(xx[6])}
  ```
## D0 evaluation results
  - **From ckpt**
  ```py
  """ automl main.py """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.358
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.451
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.475
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699

  """ mine from ckp, clip bbox, per_class, norm before resize, topk """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.359
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.452
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.477
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.703
  ```
  - **From h5**
  ```py
  """ mine from h5, norm before resize, clip bbox, per_class, topk """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.366
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.132
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.538
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.460
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.484
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.204
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710

  """ mine from h5, norm before resize, clip bbox, mode=per_class, topk=-1 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.339
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.361
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.394
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.534
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.441
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.548
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692

  """ mine from h5, norm before resize, clip bbox, mode=global, topk=-1 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.354
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.113
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.385
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.520
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.273
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.422
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.523
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656

  """ mine from h5, norm before resize, clip bbox, method=hard, mode=global, topk=-1 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.347
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.115
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.271
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.411
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.437
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.164
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643

  """ mine from h5, norm before resize, method=hard, mode=global, topk=-1 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.346
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.115
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.517
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.408
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.164
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.635

  """ mine from h5, norm before resize, score=0.1, method=hard, mode=global, topk=-1 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.326
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.501
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.344
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.112
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.383
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.513
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.268
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.394
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.144
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.485
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
  ```
## Self trained models
  ```py
  from keras_cv_attention_models.coco import eval_func
  from keras_cv_attention_models import efficientdet, efficientnet
  backbone = efficientnet.EfficientNetV1B0(input_shape=(256, 256, 3), num_classes=0, pretrained=None)
  pretrained = "checkpoints/EfficientDet_EfficientNetV1B0_256_coco_2017_adamw_batchsize_64_randaug_6_RRC_0.08_lr512_0.008_wd_0.02_latest.h5"
  mm = efficientdet.EfficientDetD0(backbone=backbone, pretrained=pretrained, num_classes=80)
  eval_func.run_coco_evaluation(mm, nms_score_threshold=0.001, nms_method='gaussian', nms_mode="per_class", nms_topk=5000, batch_size=2)
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m checkpoints/EfficientDet_EfficientNetV2B0_latest.h5 -d coco -b 8
  ```
  ```py
  """ trained EffD0 256 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.195
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.326
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.200
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.188
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.401
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.184
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.285
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.322
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.579

  """ trained EffV2D0 256 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.204
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.338
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.211
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.208
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.411
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.192
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.295
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.308
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.033
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.340
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582

  """ trained EffD0 256 mosaic 0.5 """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.200
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.343
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.203
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.206
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.392
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.185
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.286
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.335
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.566
  ```
## automl main results
  ```sh
  CUDA_VISIBLE_DEVICES='-1' python tf2/eval.py --model_name=efficientdet-d0 --model_dir=efficientdet-d0 --val_file_pattern=tfrecord/val* --val_json_file=annotations/instances_val2017.json --hparams "mean_rgb=0.0,stddev_rgb=1.0,scale_range=True"
  ```
  ```py
  """ D0 AdvProp """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.142
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.461
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.572
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.709

  """ D1 AdvProp """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.599
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.440
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.463
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.591
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.520
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.624
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.748
  ```
  ```py
  """ D0 AdvProp, model.load_weights """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.473
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.316
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.347
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.474
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.267
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.420
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.172
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.523
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667

  """ D0 AdvProp, util_keras.restore_ckpt """
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.142
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.461
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.572
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.709
  ```
## automl main log
  ```sh
  !python main.py --mode=eval --model_name=efficientdet-d0 --model_dir=efficientdet-d0 --val_file_pattern=tfrecord/val* --val_json_file=annotations/instances_val2017.json

  2022-02-08 13:03:38.747463: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
  I0208 13:03:38.747918 140138441946944 main.py:228] {'name': 'efficientdet-d0', 'act_type': 'swish', 'image_size': (512, 512), 'target_size': None, 'input_rand_hflip': True, 'jitter_min': 0.1, 'jitter_max': 2.0, 'autoaugment_policy': None, 'grid_mask': False, 'sample_image': None, 'map_freq': 5, 'num_classes': 90, 'seg_num_classes': 3, 'heads': ['object_detection'], 'skip_crowd_during_training': True, 'label_map': None, 'max_instances_per_image': 100, 'regenerate_source_id': False, 'min_level': 3, 'max_level': 7, 'num_scales': 3, 'aspect_ratios': [1.0, 2.0, 0.5], 'anchor_scale': 4.0, 'is_training_bn': True, 'momentum': 0.9, 'optimizer': 'sgd', 'learning_rate': 0.08, 'lr_warmup_init': 0.008, 'lr_warmup_epoch': 1.0, 'first_lr_drop_epoch': 200.0, 'second_lr_drop_epoch': 250.0, 'poly_lr_power': 0.9, 'clip_gradients_norm': 10.0, 'num_epochs': 300, 'data_format': 'channels_last', 'mean_rgb': [123.675, 116.28, 103.53], 'stddev_rgb': [58.395, 57.120000000000005, 57.375], 'scale_range': False, 'label_smoothing': 0.0, 'alpha': 0.25, 'gamma': 1.5, 'delta': 0.1, 'box_loss_weight': 50.0, 'iou_loss_type': None, 'iou_loss_weight': 1.0, 'weight_decay': 4e-05, 'strategy': None, 'mixed_precision': False, 'loss_scale': None, 'box_class_repeats': 3, 'fpn_cell_repeats': 3, 'fpn_num_filters': 64, 'separable_conv': True, 'apply_bn_for_resampling': True, 'conv_after_downsample': False, 'conv_bn_act_pattern': False, 'drop_remainder': True, 'nms_configs': {'method': 'gaussian', 'iou_thresh': None, 'score_thresh': 0.0, 'sigma': None, 'pyfunc': False, 'max_nms_inputs': 0, 'max_output_size': 100}, 'tflite_max_detections': 100, 'fpn_name': None, 'fpn_weight_method': None, 'fpn_config': None, 'survival_prob': None, 'img_summary_steps': None, 'lr_decay_method': 'cosine', 'moving_average_decay': 0.9998, 'ckpt_var_scope': None, 'skip_mismatch': True, 'backbone_name': 'efficientnet-b0',   'backbone_config': None, 'var_freeze_expr': None, 'use_keras_model': True, 'dataset_type': None, 'positives_momentum': None, 'grad_checkpoint': False, 'verbose': 1, 'save_freq': 'epoch', 'model_name': 'efficientdet-d0', 'iterations_per_loop': 1000, 'model_dir': 'efficientdet-d0', 'num_shards': 8, 'num_examples_per_epoch': 120000, 'backbone_ckpt': '', 'ckpt': None, 'val_json_file': 'annotations/instances_val2017.json', 'testdev_dir': None, 'profile': False, 'mode': 'eval'}
  INFO:tensorflow:Using config: {'_model_dir': 'efficientdet-d0', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': allow_soft_placement: true
  , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 1000, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_checkpoint_save_graph_def': True, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
  I0208 13:03:38.774948 140138441946944 estimator.py:202] Using config: {'_model_dir': 'efficientdet-d0', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': allow_soft_placement: true
  , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 1000, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_checkpoint_save_graph_def': True, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
  INFO:tensorflow:Using config: {'_model_dir': 'efficientdet-d0', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': allow_soft_placement: true
  , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 1000, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_checkpoint_save_graph_def': True, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
  I0208 13:03:38.775390 140138441946944 estimator.py:202] Using config: {'_model_dir': 'efficientdet-d0', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': 1000, '_save_checkpoints_secs': None, '_session_config': allow_soft_placement: true
  , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 1000, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_checkpoint_save_graph_def': True, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}
  INFO:tensorflow:Waiting for new checkpoint at efficientdet-d0
  I0208 13:03:38.775578 140138441946944 checkpoint_utils.py:136] Waiting for new checkpoint at efficientdet-d0
  INFO:tensorflow:Found new checkpoint at efficientdet-d0/model
  I0208 13:03:38.778743 140138441946944 checkpoint_utils.py:145] Found new checkpoint at efficientdet-d0/model
  I0208 13:03:38.778815 140138441946944 main.py:308] Starting to evaluate.
  INFO:tensorflow:Calling model_fn.
  I0208 13:03:39.212198 140138441946944 estimator.py:1173] Calling model_fn.
  I0208 13:03:39.212340 140138441946944 utils.py:600] use mixed precision policy name float32
  WARNING:tensorflow:From /home/leondgarse/workspace/samba/automl/efficientdet/utils.py:601: The name tf.keras.layers.enable_v2_dtype_behavior is deprecated. Please use tf.compat.v1.keras.layers.enable_v2_dtype_behavior instead.

  W0208 13:03:39.378921 140138441946944 module_wrapper.py:149] From /home/leondgarse/workspace/samba/automl/efficientdet/utils.py:601: The name tf.keras.layers.enable_v2_dtype_behavior is deprecated. Please use tf.compat.v1.keras.layers.enable_v2_dtype_behavior instead.

  I0208 13:03:39.381606 140138441946944 efficientnet_builder.py:215] global_params= GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=0.2, data_format='channels_last', num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, depth_divisor=8, min_depth=None, survival_prob=0.0, relu_fn=functools.partial(<function activation_fn at 0x7f740c3ce940>, act_type='swish'), batch_norm=<class 'utils.BatchNormalization'>, use_se=True, local_pooling=None, condconv_num_experts=None, clip_projection_output=False, blocks_args=['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25'], fix_head_stem=None, grad_checkpoint=False)
  I0208 13:03:42.295468 140138441946944 det_model_fn.py:81] LR schedule method: cosine
  I0208 13:03:42.497536 140138441946944 postprocess.py:92] use max_nms_inputs for pre-nms topk.
  I0208 13:03:42.966423 140138441946944 det_model_fn.py:476] Eval val with groudtruths annotations/instances_val2017.json.
  I0208 13:03:42.994841 140138441946944 det_model_fn.py:553] Load EMA vars with ema_decay=0.999800
  INFO:tensorflow:Done calling model_fn.
  I0208 13:03:43.446776 140138441946944 estimator.py:1175] Done calling model_fn.
  INFO:tensorflow:Starting evaluation at 2022-02-08T13:03:43
  I0208 13:03:43.458636 140138441946944 evaluation.py:250] Starting evaluation at 2022-02-08T13:03:43
  INFO:tensorflow:Graph was finalized.
  I0208 13:03:43.718523 140138441946944 monitored_session.py:243] Graph was finalized.
  INFO:tensorflow:Restoring parameters from efficientdet-d0/model
  I0208 13:03:43.721569 140138441946944 saver.py:1395] Restoring parameters from efficientdet-d0/model
  INFO:tensorflow:Running local_init_op.
  I0208 13:03:45.005094 140138441946944 session_manager.py:527] Running local_init_op.
  INFO:tensorflow:Done running local_init_op.
  I0208 13:03:45.100110 140138441946944 session_manager.py:530] Done running local_init_op.
  INFO:tensorflow:Evaluation [500/5000]
  I0208 13:04:56.714685 140138441946944 evaluation.py:163] Evaluation [500/5000]
  INFO:tensorflow:Evaluation [1000/5000]
  I0208 13:06:05.720834 140138441946944 evaluation.py:163] Evaluation [1000/5000]
  INFO:tensorflow:Evaluation [1500/5000]
  I0208 13:07:14.748428 140138441946944 evaluation.py:163] Evaluation [1500/5000]
  INFO:tensorflow:Evaluation [2000/5000]
  I0208 13:08:26.162289 140138441946944 evaluation.py:163] Evaluation [2000/5000]
  INFO:tensorflow:Evaluation [2500/5000]
  I0208 13:09:37.366406 140138441946944 evaluation.py:163] Evaluation [2500/5000]
  INFO:tensorflow:Evaluation [3000/5000]
  I0208 13:10:49.291777 140138441946944 evaluation.py:163] Evaluation [3000/5000]
  INFO:tensorflow:Evaluation [3500/5000]
  I0208 13:12:03.320028 140138441946944 evaluation.py:163] Evaluation [3500/5000]
  INFO:tensorflow:Evaluation [4000/5000]
  I0208 13:13:16.708130 140138441946944 evaluation.py:163] Evaluation [4000/5000]
  INFO:tensorflow:Evaluation [4500/5000]
  I0208 13:14:28.731283 140138441946944 evaluation.py:163] Evaluation [4500/5000]
  INFO:tensorflow:Evaluation [5000/5000]
  I0208 13:15:42.472607 140138441946944 evaluation.py:163] Evaluation [5000/5000]
  loading annotations into memory...
  Done (t=0.59s)
  creating index...
  index created!
  Loading and preparing results...
  Converting ndarray to lists...
  (500000, 7)
  0/500000
  DONE (t=2.49s)
  creating index...
  index created!
  Running per image evaluation...
  Evaluate annotation type *bbox*
  DONE (t=48.30s).
  Accumulating evaluation results...
  DONE (t=10.07s).
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
   Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
   Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.358
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
   Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
   Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.288
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.451
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.475
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
   Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
   Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699
  INFO:tensorflow:Inference Time : 781.78098s
  I0208 13:16:45.239724 140138441946944 evaluation.py:269] Inference Time : 781.78098s
  INFO:tensorflow:Finished evaluation at 2022-02-08-13:16:45
  I0208 13:16:45.240339 140138441946944 evaluation.py:271] Finished evaluation at 2022-02-08-13:16:45
  INFO:tensorflow:Saving dict for global step 0: AP = 0.33509377, AP50 = 0.514792, AP75 = 0.3583659, APl = 0.5259228, APm = 0.3860942, APs = 0.12527254, ARl = 0.69915473, ARm = 0.5572281, ARmax1 = 0.28827018, ARmax10 = 0.45066467, ARmax100 = 0.4748777, ARs = 0.19994779, box_loss = 0.0, cls_loss = 30.703037, global_step = 0, loss = 30.798128
  I0208 13:16:45.240440 140138441946944 estimator.py:2083] Saving dict for global step 0: AP = 0.33509377, AP50 = 0.514792, AP75 = 0.3583659, APl = 0.5259228, APm = 0.3860942, APs = 0.12527254, ARl = 0.69915473, ARm = 0.5572281, ARmax1 = 0.28827018, ARmax10 = 0.45066467, ARmax100 = 0.4748777, ARs = 0.19994779, box_loss = 0.0, cls_loss = 30.703037, global_step = 0, loss = 30.798128
  INFO:tensorflow:Saving 'checkpoint_path' summary for global step 0: efficientdet-d0/model
  I0208 13:16:46.018409 140138441946944 estimator.py:2143] Saving 'checkpoint_path' summary for global step 0: efficientdet-d0/model
  I0208 13:16:46.018846 140138441946944 main.py:316] efficientdet-d0/model has no global step info: stop!
  ```
***

# EfficientDet lite
  ```py
  from keras_cv_attention_models.efficientdet import convert_efficientdet as efficientdet
  backbone = efficientdet.EfficientNetV1B0(input_shape=(320, 320, 3), output_conv_filter=0, se_ratios=[0] * 10, is_fused=False, num_classes=0, activation="relu6", pretrained=None)
  bb = efficientdet.EfficientDetD0(backbone=backbone, activation="relu6", anchor_scale=3, use_weighted_sum=False, pretrained=None)
  ```
