## Troch roi align
  - [token labeling](https://www.cnblogs.com/dan-baishucaizi/p/14994269.html)
  ```py
  # RandomResizedCropAndInterpolationWithCoords, https://github.com/zihangJiang/TokenLabeling/blob/aa438eff9b9fc2daa8c8b4cc6bfaa6e3721f995e/tlt/data/label_transforms_factory.py#L41
  i, j, h, w = self.get_params(img, self.scale, self.ratio)
  coords = (i / img.size[1],
            j / img.size[0],
            h / img.size[1],
            w / img.size[0])
  coords_map = torch.zeros_like(label_map[0:1])
  # trick to store coords_map is label_map
  coords_map[0,0,0,0],coords_map[0,0,0,1],coords_map[0,0,0,2],coords_map[0,0,0,3] = coords
  label_map = torch.cat([label_map, coords_map])
  if isinstance(self.interpolation, (tuple, list)):
      interpolation = random.choice(self.interpolation)
  else:
      interpolation = self.interpolation
  return torchvision_F.resized_crop(img, i, j, h, w, self.size, interpolation), label_map
  ```
  ```py
  from torchvision.ops import roi_align
  import torch

  def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
      x = x.long().view(-1, 1)
      return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

  def get_featuremaps(label_maps_topk, num_classes, device='cuda'):
      label_maps_topk_sizes = label_maps_topk[0].size()
      label_maps = torch.full([label_maps_topk.size(0), num_classes, label_maps_topk_sizes[2],
                                label_maps_topk_sizes[3]], 0, dtype=torch.float32 ,device=device)
      for _label_map, _label_topk in zip(label_maps, label_maps_topk):
          _label_map = _label_map.scatter_(
              0,
              _label_topk[1][:, :, :].long(),
              _label_topk[0][:, :, :].float()
          )
      return label_maps

  def get_label(label_maps, batch_coords,label_size=1,device='cuda'):
      num_batches = label_maps.size(0)
      print(">>>> boxes:", torch.cat([torch.arange(num_batches).view(num_batches, 1).float().to(device), batch_coords.float() * label_maps.size(3) - 0.5], 1))
      target_label = roi_align(
          input=label_maps,
          boxes=torch.cat([torch.arange(num_batches).view(num_batches, 1).float().to(device), batch_coords.float() * label_maps.size(3) - 0.5], 1),
          output_size=(label_size, label_size))
      if label_size>1:
          target_label_cls = roi_align(
              input=label_maps,
              boxes=torch.cat([torch.arange(num_batches).view(num_batches, 1).float().to(device), batch_coords.float() * label_maps.size(3) - 0.5], 1),
              output_size=(1, 1))
          B,C,H,W = target_label.shape
          target_label = target_label.view(B,C,H*W)
          target_label = torch.cat([target_label_cls.view(B,C,1),target_label],dim=2)
      # target_label = torch.nn.functional.softmax(target_label.squeeze(), 1)
      return target_label

  def get_labelmaps_with_coords(label_maps_topk, num_classes, on_value=1., off_value=0.,label_size=1, device='cuda'):
      '''
      Adapted from https://github.com/naver-ai/relabel_imagenet/blob/main/utils/relabel_functions.py
      Generate the target label map for training from the given bbox and raw label map
      '''
      # trick to get coords_map from label_map
      random_crop_coords = label_maps_topk[:,2,0,0,:4].view(-1, 4)
      random_crop_coords[:, 2:] += random_crop_coords[:, :2]
      random_crop_coords = random_crop_coords.to(device)

      # trick to get ground truth from label_map
      ground_truth = label_maps_topk[:,2,0,0,5].view(-1).to(dtype=torch.int64)
      ground_truth = one_hot(ground_truth, num_classes, on_value=on_value, off_value=off_value, device=device)

      # get full label maps from raw topk labels
      label_maps = get_featuremaps(label_maps_topk=label_maps_topk, num_classes=num_classes,device=device)

      # get token-level label and ground truth
      token_label = get_label(label_maps=label_maps, batch_coords=random_crop_coords, label_size=label_size, device=device)
      B,C = token_label.shape[:2]
      token_label = token_label*on_value+off_value
      if label_size==1:
          return torch.cat([ground_truth.view(B,C,1),token_label.view(B,C,1)],dim=2)
      else:
          return torch.cat([ground_truth.view(B,C,1),token_label],dim=2)

  label_map = np.concatenate([np.random.uniform(size=(1, 5, 18, 18)), np.random.choice(10, size=[1, 5, 18, 18])], 0)

  label_map = torch.from_numpy(label_map)
  coords_map = torch.zeros_like(label_map[0:1])
  coords_map[0, 0, 0, :4] = torch.tensor([0, 0, 1, 1])
  label_map = torch.cat([label_map, coords_map])

  label_maps_topk = torch.from_numpy(np.expand_dims(label_map, 0))
  num_classes, on_value, off_value, label_size, device = 20, 1, 0, 14, 'cpu'
  aa = get_labelmaps_with_coords(label_maps_topk, num_classes, label_size=label_size, device=device).numpy()
  ```
## TF crop and resize
  ```py
  def token_label_preprocess(token_label, num_classes=10, num_pathes=14):
      if token_label.shape[-1] != num_classes:
          ibb = np.zeros([*token_label.shape[1:-1], num_classes])
          ipp_ids = tf.cast(tf.reshape(token_label[0], (-1, token_label.shape[-1])), tf.int32)
          ipp_scores = tf.reshape(token_label[1], (-1, token_label.shape[-1]))
          for ii, id, score in zip(ibb.reshape(-1, ibb.shape[-1]), ipp_ids, ipp_scores):
              ii[id] = score
          token_label = ibb

      cur_patches = token_label.shape[0]
      if num_pathes != cur_patches:
          # token_label = tf.image.resize(token_label, (num_pathes, num_pathes))
          # token_label = tf.gather(token_label, pick, axis=0)        
          # token_label = tf.gather(token_label, pick, axis=1)
          pick = np.clip(np.arange(0, cur_patches, cur_patches / num_pathes), 0, cur_patches - 1)
          pick_floor = np.floor(pick).astype('int')
          pick_ceil = np.ceil(pick).astype('int')
          pick_val = tf.reshape(tf.cast(pick - pick_floor, token_label.dtype), [-1, 1, 1])

          token_label = tf.gather(token_label, pick_floor, axis=0) * pick_val + tf.gather(token_label, pick_ceil, axis=0) * (1 - pick_val)
          pick_val = tf.transpose(pick_val, [1, 0, 2])
          token_label = tf.gather(token_label, pick_floor, axis=1) * pick_val + tf.gather(token_label, pick_ceil, axis=1) * (1 - pick_val)
      token_label = tf.reshape(token_label, (num_pathes * num_pathes, token_label.shape[-1]))
      return token_label
  ```
  ```py
  batch_size = 1
  num_boxes = 5
  crop_size = (24, 24)

  from skimage.data import chelsea
  image = chelsea()

  target_patches = 7
  xx, yy = np.meshgrid(np.arange(0, target_patches), np.arange(0, target_patches))
  xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
  zz = np.concatenate([xx, yy, xx + 1, yy + 1], axis=-1) / target_patches
  output = tf.image.crop_and_resize(np.expand_dims(image, 0), zz, [0] * (target_patches * target_patches), crop_size)

  out = output.numpy().reshape(target_patches, target_patches, *crop_size, image.shape[-1])

  fig, axes = plt.subplots(1, 2)
  axes[0].imshow(image)
  axes[1].imshow(np.hstack([np.vstack(out[ii]) for ii in range(target_patches)]) / 255)
  ```
  ```py
  image = tf.random.uniform([7, 7, 3])
  crop_size = (1, 1)
  target_patches = 14
  xx, yy = np.meshgrid(np.arange(0, target_patches), np.arange(0, target_patches))
  xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
  zz = np.concatenate([xx, yy, xx + 1, yy + 1], axis=-1) / target_patches
  output = tf.image.crop_and_resize(np.expand_dims(image, 0), zz, [0] * (target_patches * target_patches), crop_size)

  out = output.numpy().reshape(target_patches, target_patches, *crop_size, image.shape[-1])

  fig, axes = plt.subplots(1, 3)
  axes[0].imshow(image)
  axes[1].imshow(np.hstack([np.vstack(out[ii]) for ii in range(target_patches)]))

  imm = token_label_preprocess(image, num_classes=image.shape[-1], num_pathes=target_patches).numpy()
  imm = imm.reshape(target_patches, target_patches, image.shape[-1])
  axes[2].imshow(imm)
  ```
  ```py
  def token_label_preprocess(token_label, num_classes=10, num_pathes=14):
      if token_label.shape[-1] != num_classes:  # To one_hot like
          ipp_ids = tf.cast(tf.reshape(token_label[0], (-1, token_label.shape[-1], 1)), tf.int32)
          ipp_scores = tf.cast(tf.reshape(token_label[1], (-1, token_label.shape[-1])), tf.float32)

          iaa = tf.zeros(num_classes)
          ibb = tf.stack([tf.tensor_scatter_nd_update(iaa, ipp_ids[ii], ipp_scores[ii]) for ii in range(ipp_ids.shape[0])])
          # hhww = token_label.shape[1] * token_label.shape[2]
          # id_indexes = tf.expand_dims(tf.range(hhww), 1) + tf.expand_dims(tf.range(token_label.shape[-1]), 0)
          # indexed_ids = tf.concat([tf.expand_dims(id_indexes, -1), ipp_ids], -1)
          # ibb = tf.zeros([hhww, num_classes])
          # tf.print(ibb.shape, indexed_ids.shape, ipp_scores.shape)
          # ibb = tf.tensor_scatter_nd_update(ibb, indexed_ids, ipp_scores)
          token_label = tf.reshape(ibb, [token_label.shape[1], token_label.shape[2], ibb.shape[-1]])

      cur_patches = token_label.shape[0]
      if num_pathes != cur_patches:
          xx, yy = np.meshgrid(np.arange(0, num_pathes), np.arange(0, num_pathes))
          xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
          boxes = np.concatenate([xx, yy, xx + 1, yy + 1], axis=-1) / num_pathes
          box_indices = [0] * (num_pathes * num_pathes)
          token_label = tf.image.crop_and_resize(tf.expand_dims(token_label, 0), boxes, box_indices, crop_size=(1, 1))
      token_label = tf.reshape(token_label, (num_pathes * num_pathes, token_label.shape[-1]))
      return token_label

  token_label = np.concatenate([np.random.choice(10, size=[1, 7, 7, 5]), np.random.uniform(size=(1, 7, 7, 5))], 0)
  token_label_preprocess(token_label, num_classes=10).shape
  # TensorShape([196, 10])

  def load_cifar10_token_label(label_token_file, num_classes=10, batch_size=1024, image_shape=(32, 32), num_pathes=14):
      import tensorflow_datasets as tfds
      AUTOTUNE = tf.data.experimental.AUTOTUNE

      train_ds = tfds.load("cifar10", split="train")

      token_label_data = np.load(label_token_file)
      token_label_ds = tf.data.Dataset.from_tensor_slices(token_label_data)
      token_label_train_ds = tf.data.Dataset.zip((train_ds, token_label_ds))

      image_preprocess = lambda data: tf.image.resize(data["image"], image_shape[:2]) / 255.0
      label_preprocess = lambda data: tf.one_hot(data["label"], depth=num_classes)

      train_preprocessing = lambda data, token_label: (image_preprocess(data), (label_preprocess(data), token_label_preprocess(token_label, num_classes, num_pathes)))
      token_label_train_ds = token_label_train_ds.shuffle(buffer_size=batch_size * 100).map(train_preprocessing, num_parallel_calls=AUTOTUNE)
      token_label_train_ds = token_label_train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

      # Load test dataset
      test_preprocessing = lambda data: (image_preprocess(data), label_preprocess(data))
      test_ds = tfds.load("cifar10", split="test").map(test_preprocessing, num_parallel_calls=AUTOTUNE).batch(batch_size)
      return token_label_train_ds, test_ds

  token_label_train_ds, test_ds = load_cifar10_token_label("cifar10_token_label_patch_50000_2_7_7_5.npy", batch_size=32, image_shape=(224, 224))
  dd = token_label_train_ds.as_numpy_iterator().next()
  ```
## Token labeling

  ```py
  from skimage.data import chelsea
  img = chelsea()[:224, :224]
  num_patches = 14
  patch_size = 224 // 14
  iaa = img.reshape(num_patches, patch_size, num_patches, patch_size, 3).transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, 3)

  from skimage.transform import resize
  mm = keras.applications.EfficientNetB1(input_shape=(240, 240, 3))
  ibb = np.array([resize(ii, mm.input_shape[1:3]) for ii in iaa])
  ipp = mm(ibb).numpy()
  for id, ii in enumerate(np.argsort(ipp, axis=-1)):
      ipp[id][ii[:-5]] = 0  # Keep the top 5
  print(f"{ipp.shape = }, {(ipp != 0).sum(1) = }")
  # ipp.shape = (196, 1000), (ipp != 0).sum(1) = array([5, 5, 5, 5, ...])
  ```
  ```py
  import numpy as np
  import tensorflow as tf

  def token_label_preprocessing(image, teacher_model, num_patches=14, top_k=5, return_one_hot=True):
      patch_size = image_shape[0] // num_patches
      iaa = image.numpy().reshape(num_patches, patch_size, num_patches, patch_size, 3).transpose(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, 3)
      ibb = tf.stack([tf.image.resize(ii, teacher_model.input_shape[1:3]) for ii in iaa]) # (0, 255)
      if teacher_model.layers[-1].activation.__name__ == "softmax":
          ipp = teacher_model(ibb).numpy()
      else:
          ipp = tf.nn.softmax(teacher_model(ibb)).numpy()

      if return_one_hot:
          for id, ii in enumerate(np.argsort(ipp, axis=-1)):
              ipp[id][ii[:-top_k]] = 0  # Keep the top top_k
          return ipp
      else:
          ipp_ids = np.argsort(ipp, axis=-1)[:, -top_k:]
          ipp_scores = np.stack([ii[jj] for ii, jj in zip(ipp, ipp_ids)])
          return ipp_ids, ipp_scores

  import tensorflow_datasets as tfds
  train_ds = tfds.load("cifar10", split="train")
  data = train_ds.as_numpy_iterator().next()
  image_shape = (224, 224)
  image = tf.image.resize(data["image"], image_shape[:2])

  # Return encoded one-hot labels
  model = keras.models.load_model(tf.io.gfile.glob("ef_b0_cifar10_*.h5")[0])
  ipp = token_label_preprocessing(image, model)
  print(f"{ipp.shape = }")
  # ipp.shape = (196, 10)

  # Return ids and their scores if num_classes is large
  num_classes = 10
  ipp_ids, ipp_scores = token_label_preprocessing(image, model, return_one_hot=False)
  print(f"{ipp_ids.shape = }, {ipp_scores.shape = }")
  ibb = np.zeros([ipp_ids.shape[0], num_classes])
  for ii, id, score in zip(ibb, ipp_ids, ipp_scores):
      ii[id] = score

  print(f"{np.allclose(ibb, ipp) = }")
  # np.allclose(ibb, ipp) = True

  from tqdm import tqdm
  aa = [token_label_preprocessing(tf.image.resize(data["image"], image_shape[:2]), model) for data in tqdm(train_ds)]
  np.save('cifar10_label_token_{}.npy'.format(aa[0].shape[0]), np.stack(aa))

  label_token = np.load("cifar10_label_token_patch_196.npy")
  label_token_ds = tf.data.Dataset.from_tensor_slices(label_token)
  label_token_train_ds = tf.data.Dataset.zip((train_ds, label_token_ds))
  bb = label_token_train_ds.as_numpy_iterator().next()
  print(f"{bb[0]['image'].shape = }, {bb[0]['label'] = }, {bb[1].shape = }")
  # bb[0]['image'].shape = (32, 32, 3), bb[0]['label'] = 7, bb[1].shape = (196, 10)

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  image_shape = (224, 224, 3)
  batch_size = 32
  data_preprocessing = lambda data: (tf.image.resize(data["image"], image_shape[:2]), data["label"])
  label_token_preprocessing = lambda data, label_token: (tf.image.resize(data["image"], image_shape[:2]), (data["label"], label_token))
  label_token_train_ds = label_token_train_ds.shuffle(buffer_size=batch_size * 100).map(label_token_preprocessing, num_parallel_calls=AUTOTUNE)
  label_token_train_ds = label_token_train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
  bb = label_token_train_ds.as_numpy_iterator().next()
  print(f"{bb[0].shape = }, {bb[1][0].shape = }, {bb[1][1].shape = }")
  # bb[0].shape = (32, 224, 224, 3), bb[1][0].shape = (32,), bb[1][1].shape = (32, 196, 10)

  test_ds = tfds.load("cifar10", split="test").map(data_preprocessing, num_parallel_calls=AUTOTUNE).batch(batch_size)
  bb = test_ds.as_numpy_iterator().next()
  print(f"{bb[0].shape = }, {bb[1].shape = }")
  # bb[0].shape = (32, 224, 224, 3), bb[1].shape = (32,)
  ```
***

# Model
```py
def convert_to_token_label_model(model, pool_layer_id="auto"):
    # Search pool layer id
    num_total_layers = len(model.layers)
    if pool_layer_id == "auto":
        for header_layer_id, layer in enumerate(model.layers[::-1]):
            header_layer_id = num_total_layers - header_layer_id - 1
            print("[Search pool layer] header_layer_id = {}, layer.name = {}".format(header_layer_id, layer.name))
            if isinstance(layer, keras.layers.GlobalAveragePooling2D):
                break
        pool_layer_id = header_layer_id

    nn = model.layers[pool_layer_id - 1].output # layer output before pool layer

    # Add header layers w/o pool layer
    for header_layer_id in range(pool_layer_id + 1, num_total_layers):
        aa = model.layers[header_layer_id]
        print("[Build new layer] header_layer_id = {}, layer.name = {}".format(header_layer_id, aa.name))
        config = aa.get_config()
        config["name"] = config["name"] + "_token_label"
        bb = aa.__class__.from_config(config)
        bb.build(aa.input_shape)
        bb.set_weights(aa.get_weights())
        nn = bb(nn)
    token_label_model = keras.models.Model(model.inputs[0], [*model.outputs, nn])
    print("token_label_model.output_shape =", token_label_model.output_shape)
    return token_label_model

def extract_token_label_batch(image_batch, model, top_k=5):
    predictions = model(image_batch)[-1]
    if model.layers[-1].activation.__name__ != "softmax":
        predictions = tf.nn.softmax(predictions, axis=-1)

    prediction_scores, prediction_ids = tf.math.top_k(predictions, k=top_k)
    return tf.stack([tf.cast(prediction_ids, prediction_scores.dtype), prediction_scores], axis=1)


from keras_cv_attention_models import visualizing

def show_patches(image, label, resize_shape=(160, 160)):
    height, width = image.shape[:2]
    num_height_patch, num_width_patch = label.shape[1], label.shape[2]
    height_patch, width_patch = int(tf.math.ceil(height / num_height_patch)), int(tf.math.ceil(width / num_width_patch))
    # fig, axes = plt.subplots(num_height_patch, num_width_patch)

    image_pathes, labels = [], []
    for hh_id in range(num_height_patch):
        hh_image = image[hh_id * height_patch: (hh_id + 1) * height_patch]
        for ww_id in range(num_width_patch):
            image_patch = hh_image[:, ww_id * width_patch: (ww_id + 1) * width_patch]
            image_pathes.append(tf.image.resize(image_patch, resize_shape).numpy())
            scores = ",".join(["{:.1f}".format(ii * 100) for ii in label[1, hh_id, ww_id]])
            classes = ",".join(["{:d}".format(ii) for ii in label[0, hh_id, ww_id].astype('int')])
            labels.append(classes + "\n" + scores)
    visualizing.stack_and_plot_images(image_pathes, labels)

import kecam
# mm = kecam.efficientnet.EfficientNetV2S()
mm = keras.models.load_model('keras_cv_attention_models/volo/ef_b0_cifar10_0.9297.h5')
nn = convert_to_token_label_model(mm)
tt = kecam.imagenet.data.init_dataset("cifar10", seed=0)[0]
aa, bb = tt.as_numpy_iterator().next()
cc = extract_token_label_batch(aa, nn)
print(f"{aa.shape = }, {bb.shape = }, {cc.shape = }")
# aa.shape = (64, 224, 224, 3), bb.shape = (64, 10), cc.shape = (64, 2, 7, 7, 5)

image, label, resize = aa[0], cc[0], (100, 100)
show_patches(image / 2 + 0.5, label)
```
```py
class TokenLabelAlign:
    def __init__(self, num_classes=10, target_num_pathes=14, align_method="bilinear"):
        self.num_classes, self.align_method = num_classes, align_method
        target_num_pathes = target_num_pathes[:2] if isinstance(target_num_pathes, (list, tuple)) else (target_num_pathes, target_num_pathes)
        self.target_num_pathes_h, self.target_num_pathes_w = target_num_pathes
        self.built = False

    def build(self, token_label_shape):
        # To one-hot
        self.source_patch_h, self.source_patch_w, num_topk = token_label_shape[1], token_label_shape[2], token_label_shape[3]
        hh, ww = tf.meshgrid(range(self.source_patch_h), range(self.source_patch_w), indexing='ij')
        hhww = tf.concat([tf.reshape(hh, [-1, 1, 1]), tf.reshape(ww, [-1, 1, 1])], axis=-1)
        self.one_hot_hhww = tf.repeat(hhww, num_topk, axis=1)

        # Align to target shape
        hh, ww = tf.meshgrid(range(0, self.target_num_pathes_h), range(0, self.target_num_pathes_w), indexing='ij')
        hh, ww = tf.reshape(hh, [-1, 1]), tf.reshape(ww, [-1, 1])
        boxes = tf.concat([hh, ww, hh + 1, ww + 1], axis=-1)
        self.boxes = tf.cast(boxes, "float32") / [self.target_num_pathes_h, self.target_num_pathes_w, self.target_num_pathes_h, self.target_num_pathes_w]
        self.box_indices = [0] * (self.target_num_pathes_h * self.target_num_pathes_w)  # 0 is indicating which batch
        self.need_align = self.target_num_pathes_h != self.source_patch_h or self.target_num_pathes_w != self.source_patch_w

    def __call__(self, token_label):
        if not self.built:
            self.build(token_label.shape)
        label_pos, label_score = tf.cast(token_label[0], "int32"), tf.cast(token_label[1], "float32")
        label_position = tf.concat([tf.reshape(self.one_hot_hhww, [-1, 2]), tf.reshape(label_pos, [-1, 1])], axis=-1)
        token_label_one_hot = tf.zeros([self.source_patch_h, self.source_patch_w, self.num_classes])
        token_label_one_hot = tf.tensor_scatter_nd_update(token_label_one_hot, label_position, tf.reshape(label_score, -1))

        if self.need_align:
            token_label_one_hot = tf.expand_dims(token_label_one_hot, 0)  # Expand a batch dimension, required by crop_and_resize
            token_label_one_hot = tf.image.crop_and_resize(token_label_one_hot, self.boxes, self.box_indices, crop_size=(1, 1), method=self.align_method)
        return tf.reshape(token_label_one_hot, (self.target_num_pathes_h, self.target_num_pathes_w, self.num_classes))

dd = TokenLabelAlign(target_num_pathes=7)
print(f"{np.allclose(tf.math.top_k(dd(cc[0]), 5)[0], cc[0][1]) = }")
print(f"{np.allclose(tf.math.top_k(dd(cc[0]), 5)[1], cc[0][0]) = }")
```
