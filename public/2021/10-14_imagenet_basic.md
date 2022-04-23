# ___2021 - 10 - 14 ImageNet Basic___
***
# TOC
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2021 - 10 - 14 ImageNet Basic___](#2021-10-14-imagenet-basic)
  - [TOC](#toc)
  - [Init dataset](#init-dataset)
  - [Timm Imagenet](#timm-imagenet)
  	- [Timm Training](#timm-training)
  	- [Timm validation](#timm-validation)
  	- [inception crop](#inception-crop)
  	- [Timm randaug and RRC](#timm-randaug-and-rrc)
  - [TF Imagenet Basic](#tf-imagenet-basic)
  	- [TF Training](#tf-training)
  	- [Pretrained Model Validation](#pretrained-model-validation)

  <!-- /TOC -->
***

# Init dataset
  - [ILSVRC2012_img_train.tar](magnet:?xt=urn:btih:umdds7gptqxk2jyvlgb4evbcpqh5sohc)
  - [ILSVRC2012_img_val.tar](magnet:?xt=urn:btih:lvwq357nqhx5jhfjt2shg7qk4xr2l4xf)
  - [Kaggle imagenet object localization patched 2019](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
    ```sh
    unzip imagenet-object-localization-challenge.zip -d imagenet-object-localization-challenge
    cd imagenet-object-localization-challenge
    tar xvf imagenet_object_localization_patched2019.tar.gz

    cd ILSVRC/Data/CLS-LOC/train/
    ls -1 | xargs -I '{}' tar cvf {}.tar {}
    tar cvf ILSVRC2019_img_train.tar ./*.tar

    cd ../test
    ls > foo && tar cvf ILSVRC2019_img_test.tar -T foo  # exclude the directory `./`
    cd ../val
    tar cvf ILSVRC2019_img_val.tar $(ls *.JPEG)  # exclude the directory `./`
    cd ..

    DATASET_PATH='/media/SD/tdtest/imagenet_2019'
    mkdir $DATASET_PATH
    mv train/ILSVRC2019_img_train.tar test/ILSVRC2019_img_test.tar val/ILSVRC2019_img_val.tar $DATASET_PATH

    mkdir -p ~/tensorflow_datasets/downloads/manual
    rm -f ~/tensorflow_datasets/downloads/manual/ILSVRC2012_img_*.tar
    ln -s $DATASET_PATH/ILSVRC2019_img_train.tar ~/tensorflow_datasets/downloads/manual/ILSVRC2012_img_train.tar
    ln -s $DATASET_PATH/ILSVRC2019_img_val.tar ~/tensorflow_datasets/downloads/manual/ILSVRC2012_img_val.tar
    ln -s $DATASET_PATH/ILSVRC2019_img_test.tar ~/tensorflow_datasets/downloads/manual/ILSVRC2012_img_test.tar

    mv ~/tensorflow_datasets/imagenet2012 ~/tensorflow_datasets/imagenet2012.bak
    ```
    ```py
    import tensorflow_datasets as tfds
    aa = tfds.image_classification.Imagenet2012()
    aa.download_and_prepare()
    train_ds, val_ds = aa.as_dataset(split='train'), aa.as_dataset(split='validation')
    ```
    ```py
    import tensorflow_datasets as tfds
    train_ds, val_ds = tfds.load('imagenet2012', split='train'), tfds.load('imagenet2012', split='validation')

    data = val_ds.as_numpy_iterator().next()
    print(data.keys())
    # dict_keys(['file_name', 'image', 'label'])
    plt.imshow(data['image'])
    ```
***

# Timm Imagenet
## Timm Training
  ```sh
  pipi torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  pipi torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

  git clone https://github.com.cnpmjs.org/NVIDIA/apex.git && cd apex/
  vi setup.py
  # Comment off `RuntimeError` after `if (bare_metal_major != torch_binary_major) or (bare_metal_minor != torch_binary_minor):`
  sudo touch /usr/local/cuda-11.2/targets/x86_64-linux/include/cuda_profiler_api.h
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

  CUDA_VISIBLE_DEVICES='1' python train_tf.py FOO -c _79_83-fusedlamb-cosine-lr0.00500-wd0.020000-n0-rand-m7-mstd0.5-inc1-m0.1-sd0.1-d0.0-ls0.0-301-299-resnet50-args.yaml.txt

  CUDA_VISIBLE_DEVICES='0' python train.py $HOME/tensorflow_datasets/ -c _79_83-fusedlamb-cosine-lr0.00500-wd0.020000-n0-rand-m7-mstd0.5-inc1-m0.1-sd0.1-d0.0-ls0.0-301-299-resnet50-args.yaml.txt --batch-size 256 --lr 4e-3

  CUDA_VISIBLE_DEVICES='0' python validate.py $HOME/tensorflow_datasets/ --dataset tfds/imagenet2012 --img-size 224 --crop-pct 0.95 --model resnet50 --checkpoint ./output/train/20211109-111121-resnet50-160/checkpoint-99.pth.tar

  CUDA_VISIBLE_DEVICES='0' python validate_tf.py $HOME/tensorflow_datasets/ --dataset tfds/imagenet2012 --img-size 224 --crop-pct 0.95 --model resnet50 --channels-last --checkpoint ../keras_cv_attention_models/checkpoints/aotnet50_imagenet2012_batch_size_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_LAMB_lr0.008_wd0.02_epoch_102_val_acc_0.7494.h5
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='0' python validate.py $HOME/tensorflow_datasets/ --dataset tfds/imagenet2012 --img-size 224 --crop-pct 0.95 --model resnet50 --checkpoint ./output/train/20211115-150528-resnet50-160/checkpoint-96.pth.tar
  # * Acc@1 78.206 (21.794) Acc@5 93.878 (6.122)
  ```
  ```py
  def parse_timm_log(log_filee):
      with open(log_filee, 'r') as ff:
          aa = ff.readlines()

      train_epoch_started, train_epoch_end_pattern, previous_line = False, "", ""
      for ii in aa:
          if ii.startswith("Train:"):
              train_epoch_started = True
              previous_line = ii
          elif train_epoch_started and not ii.startswith("Train:"):
              train_epoch_end_pattern = previous_line.split("[")[1].split("]")[0]
              break

      test_epoch_started, test_epoch_end_pattern, previous_line = False, "", ""
      for ii in aa:
          if ii.startswith("Test:"):
              test_epoch_started = True
              previous_line = ii
          elif test_epoch_started and not ii.startswith("Test:"):
              test_epoch_end_pattern = previous_line.split("[")[1].split("]")[0]
              break

      train_loss = [float(ii.split('Loss: ')[1].split(" ")[1][1:-1]) for ii in aa if train_epoch_end_pattern in ii]
      lr = [float(ii.split('LR: ')[1].split(" ")[0]) for ii in aa if train_epoch_end_pattern in ii]
      val_loss = [float(ii.split('Loss: ')[1].strip().split(" ")[1][1:-1]) for ii in aa if test_epoch_end_pattern in ii]
      val_acc = [float(ii.split('Acc@1: ')[1].strip().split("Acc@5:")[0].split("(")[1].split(")")[0]) for ii in aa if test_epoch_end_pattern in ii]

      # print(f"{len(train_loss) = }, {len(lr) = }, {len(val_loss) = }, {len(val_acc) = }")
      return {"loss": train_loss, "lr": lr, "val_loss": val_loss, "val_acc": val_acc}

  aa = parse_timm_log('log.foo')
  from keras_cv_attention_models.imagenet import plot_hists
  plot_hists(aa, 'timm resnet50')
  ```
## Timm validation
  ```py
  from validate_tf import *
  setup_default_logging()
  args = parser.parse_args('/home/tdtest/tensorflow_datasets/ --dataset tfds/imagenet2012 --img-size 224'.split(' '))
  data = '/home/tdtest/tensorflow_datasets/'
  dataset = "tfds/imagenet2012"
  dataset = create_dataset(root=data, name=dataset, split='validation', load_bytes=False, class_map="")

  model = create_model("resnet50", pretrained=False, num_classes=1000, in_chans=3)
  data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
  loader = create_loader(
          dataset,
          input_size=data_config['input_size'],
          batch_size=256,
          use_prefetcher=True,
          interpolation=data_config['interpolation'],
          mean=data_config['mean'],
          std=data_config['std'],
          num_workers=4,
          crop_pct=0.95,
          pin_memory=False,
          tf_preprocessing=False)
  ```
  ```py
  sys.path.append('../keras_cv_attention_models/')
  from keras_cv_attention_models import aotnet
  mm = aotnet.AotNet50(input_shape=(224, 224, 3), classifier_activation=None)
  mm.load_weights('../keras_cv_attention_models/aotnet50_160_imagenet.h5')
  y_true, y_pred_top_1, y_pred_top_5 = [], [], []
  for ii in range(int(np.ceil(50000/256))):
      input = np.load('imagenet_eval/input_{}.npy'.format(ii))
      target = np.load('imagenet_eval/target_{}.npy'.format(ii))
      print(ii, input.shape, target.shape)
      predicts = mm(input).numpy()
      pred_argsort = predicts.argsort(-1)
      y_pred_top_1.extend(pred_argsort[:, -1])
      y_pred_top_5.extend(pred_argsort[:, -5:])
      y_true.extend(target)

  y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
  accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
  accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
  print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
  # >>>> Accuracy top1: 0.78066 top5: 0.9368
  ```
  ```py
  import tensorflow_datasets as tfds

  def evaluation_process_resize_crop(datapoint, target_shape=(224, 224), central_crop=1.0, resize_method="bilinear"):
      image = datapoint["image"]
      shape = tf.shape(image)
      height, width = shape[0], shape[1]
      min_border = tf.cast(tf.minimum(height, width), tf.float32)
      scale_size = tf.cast(tf.minimum(*target_shape), tf.float32) / central_crop
      hh_scale = tf.cast(tf.floor(tf.cast(height, tf.float32) * scale_size / min_border), tf.int32)
      ww_scale = tf.cast(tf.floor(tf.cast(width, tf.float32) * scale_size / min_border), tf.int32)
      image = tf.image.resize(image, (hh_scale, ww_scale), method=resize_method)

      y, x = (hh_scale - target_shape[0]) // 2, (ww_scale - target_shape[1]) // 2
      image = tf.image.crop_to_bounding_box(image, y, x, target_shape[0], target_shape[1])
      image = tf.clip_by_value(image, 0, 255)

      label = datapoint["label"]
      return image, label

  data_name, input_shape, eval_central_crop, resize_method = "imagenet2012", (224, 224), 0.95, "bicubic"
  mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
  std = tf.constant([0.229, 0.224, 0.225]) * 255.0
  rescaling = lambda xx: (xx - mean) / std
  as_one_hot = lambda yy: tf.one_hot(yy, num_classes)

  dataset, info = tfds.load(data_name, with_info=True)
  test_process = lambda xx: evaluation_process_resize_crop(xx, input_shape[:2], eval_central_crop, resize_method)  # timm
  test_dataset = dataset["validation"].map(test_process)
  test_dataset = test_dataset.batch(batch_size).map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))
  ```
  ```py
  import math
  import torch
  from torchvision import transforms
  from PIL import Image
  from tqdm import tqdm
  import tensorflow_datasets as tfds

  img_size, crop_pct, batch_size = 224, 0.95, 64
  tfl = [
      transforms.Resize(int(math.floor(img_size / crop_pct)), interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(img_size),
      transforms.ToTensor(),
      transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
  ]
  tfl = transforms.Compose(tfl)

  test_process = lambda image: np.array(tfl(Image.fromarray(image)).permute([1, 2, 0]))
  ds = tfds.load("imagenet2012", with_info=False)["validation"]

  from keras_cv_attention_models import aotnet
  mm = aotnet.AotNet50(input_shape=(img_size, img_size, 3), classifier_activation=None)
  mm.load_weights('aotnet50_160_imagenet.h5')

  total = len(ds)
  id = 0
  inputs = []
  y_true, y_pred_top_1, y_pred_top_5 = [], [], []
  mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
  std = tf.constant([0.229, 0.224, 0.225]) * 255.0
  for data_point in tqdm(ds.as_numpy_iterator(), total=len(ds)):
  # for data_point in ds.as_numpy_iterator():
      imm, label = test_process(data_point['image']), data_point['label']
      imm = (imm - mean) / std
      id += 1
      inputs.append(imm)
      y_true.append(label)
      if id % batch_size != 0 and id != total:
          continue
      # print(id, np.shape(inputs), np.shape(y_true))

      predicts = mm(np.array(inputs)).numpy()
      pred_argsort = predicts.argsort(-1)
      y_pred_top_1.extend(pred_argsort[:, -1])
      y_pred_top_5.extend(pred_argsort[:, -5:])
      inputs = []
  y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
  accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
  accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
  print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
  # >>>> Accuracy top1: 0.78066 top5: 0.9368
  ```
  ```py
  from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

  def test_process(image, target_shape=(224, 224), central_crop=0.95, resize_method="bilinear"):
      shape = tf.shape(image)
      height, width = shape[0], shape[1]
      min_border = tf.cast(tf.minimum(height, width), tf.float32)
      scale_size = tf.cast(tf.minimum(*target_shape), tf.float32) / central_crop
      hh_scale = tf.cast(tf.floor(tf.cast(height, tf.float32) * scale_size / min_border), tf.int32)
      ww_scale = tf.cast(tf.floor(tf.cast(width, tf.float32) * scale_size / min_border), tf.int32)
      # image = tf.convert_to_tensor(np.array(Image.fromarray(image).resize((ww_scale, hh_scale), resample=Image.BICUBIC)))
      resize_func = lambda xx: img_to_array(array_to_img(xx).resize((ww_scale, hh_scale), resample=Image.BICUBIC)).astype('float32')
      image = tf.numpy_function(resize_func, [image], tf.float32)
      # image = tf.image.resize(image, (hh_scale, ww_scale), method=resize_method)

      y, x = (hh_scale - target_shape[0]) // 2, (ww_scale - target_shape[1]) // 2
      image = tf.image.crop_to_bounding_box(image, y, x, target_shape[0], target_shape[1])
      # image = tf.clip_by_value(image, 0, 255)
      return tf.cast(image, 'float32')

  # >>>> Accuracy top1: 0.78058 top5: 0.93622
  # >>>> Accuracy top1: 0.78068 top5: 0.93622
  ```
  ```py
  mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
  std = tf.constant([0.229, 0.224, 0.225]) * 255.0
  rescaling = lambda xx: (xx - mean) / std

  test_dataset = dataset["validation"].map(lambda data_point: (test_process(data_point['image']), data_point['label']))
  test_dataset = test_dataset.batch(batch_size).map(lambda xx, yy: (rescaling(xx), yy))

  y_true, y_pred_top_1, y_pred_top_5 = [], [], []
  for img_batch, true_labels in tqdm(test_dataset.as_numpy_iterator(), "Evaluating", total=len(test_dataset)):
      predicts = np.array(model_interf(img_batch))
      pred_argsort = predicts.argsort(-1)
      y_pred_top_1.extend(pred_argsort[:, -1])
      y_pred_top_5.extend(pred_argsort[:, -5:])
      y_true.extend(np.array(true_labels).argmax(-1))
  y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
  accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
  accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
  print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
  return y_true, y_pred_top_1, y_pred_top_5
  ```
## inception crop
  ```py
  if inception_crop:
      channels = im.shape[-1]
      begin, size, _ = tf.image.sample_distorted_bounding_box(
          tf.shape(im),
          tf.zeros([0, 0, 4], tf.float32),
          area_range=(0.05, 1.0),
          min_object_covered=0,  # Don't enforce a minimum area.
          use_image_if_no_bounding_boxes=True)
      im = tf.slice(im, begin, size)
      # Unfortunately, the above operation loses the depth-dimension. So we
      # need to restore it the manual way.
      im.set_shape([None, None, channels])
      im = tf.image.resize(im, [crop_size, crop_size])
  else:
      im = tf.image.resize(im, [resize_size, resize_size])
      im = tf.image.random_crop(im, [crop_size, crop_size, 3])
  if tf.random.uniform(shape=[]) > 0.5:
      im = tf.image.flip_left_right(im)
  ```
## Timm randaug and RRC
  ```py
  import timm.data.auto_augment as timm_auto_augment
  from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

  PIL_INTERP_DICT = {"bicubic": 3, "bilinear": 2}
  target_shape = (224, 224)
  resize_method = "bicubic"
  magnitude = 6

  aa_params = {
      "translate_const": int(min(target_shape) * 0.45),
      "img_mean": (124, 117, 104),    # MUST be tuple
      "interpolation": PIL_INTERP_DICT.get(resize_method, PIL_INTERP_DICT["bilinear"]),
  }
  auto_augment = "rand-m{}-mstd0.5-inc1".format(magnitude)
  rr = timm_auto_augment.rand_augment_transform(auto_augment, aa_params)
  process = lambda img: img_to_array(rr(array_to_img(img)))

  from skimage.data import chelsea
  plt.imshow(np.vstack([np.hstack([process(chelsea()) for _ in range(10)]) for _ in range(10)]) / 255)
  ```
  ```py
  from keras_cv_attention_models.imagenet import augment
  from skimage.data import chelsea
  aa = augment.RandAugment(num_layers=2, magnitude=6, translate_const=0.45)
  image = tf.convert_to_tensor(chelsea().astype('float32'))
  plt.imshow(np.vstack([np.hstack([aa(image) for _ in range(10)]) for _ in range(10)]) / 255)
  ```
  ```py
  from keras_cv_attention_models.imagenet import data
  train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset('cifar10', input_shape=(224, 224), batch_size=64, random_crop_min=0.6, mixup_alpha=0.1, cutmix_alpha=1.0, rescale_mode="tf", magnitude=6)
  aa, bb = train_dataset.as_numpy_iterator().next()
  plt.imshow(np.vstack([np.hstack(aa[ii * 8 : (ii + 1) * 8]) for ii in range(8)]) / 2 + 0.5)
  ```
  ```py
  def get_params(img, scale, ratio):
      height, width = img.size[0], img.size[1]
      area = height * width

      for attempt in range(10):
          target_area = random.uniform(*scale) * area
          log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
          aspect_ratio = math.exp(random.uniform(*log_ratio))

          w = int(round(math.sqrt(target_area * aspect_ratio)))
          h = int(round(math.sqrt(target_area / aspect_ratio)))

          if w <= width and h <= height:
              i = random.randint(0, height - h)
              j = random.randint(0, width - w)
              return i, j, h, w

      # Fallback to central crop
      in_ratio = width / height
      if in_ratio < min(ratio):
          w = width
          h = int(round(w / min(ratio)))
      elif in_ratio > max(ratio):
          h = height
          w = int(round(h * max(ratio)))
      else:  # whole image
          w = width
          h = height
      i = (img.size[1] - h) // 2
      j = (img.size[0] - w) // 2
      return i, j, h, w

  def get_params_2(img, scale, ratio, log_distribute=True):
      width, height = img.size[0], img.size[1]  # img.size is (width, height)
      area = height * width
      scale_max = min(height * height * ratio[1] / area, width * width / ratio[0] / area, scale[1])
      target_area = random.uniform(scale[0], scale_max) * area

      ratio_min = max(target_area / (height * height), ratio[0])
      ratio_max = min(width * width / target_area, ratio[1])
      if log_distribute:  # More likely to select a smaller value
          log_ratio = (math.log(ratio_min), math.log(ratio_max))
          aspect_ratio = math.exp(random.uniform(*log_ratio))
      else:
          aspect_ratio = random.uniform(ratio_min, ratio_max)

      ww = int(round(math.sqrt(target_area * aspect_ratio)))
      hh = int(round(math.sqrt(target_area / aspect_ratio)))

      top = random.randint(0, height - hh)
      left = random.randint(0, width - ww)
      return top, left, hh, ww
  ```
  **Show results**
  ```py
  import math, random
  from PIL import Image

  img = Image.fromarray(np.zeros([100, 100, 3], 'uint8'))
  aa = np.array([get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.3333333)) for _ in range(100000)])
  hhs, wws = aa[:, 2], aa[:, 3]
  print("Scale range:", ((hhs * wws).min() / 1e4, (hhs * wws).max() / 1e4))
  # Scale range: (0.075, 0.9801)
  print("Ratio range:", ((wws / hhs).min(), (wws / hhs).max()))
  # Ratio range: (0.7272727272727273, 1.375)

  fig, axes = plt.subplots(4, 1, figsize=(6, 8))
  pp = {
      "ratio distribute": wws / hhs,
      "scale distribute": wws * hhs / 1e4,
      "height distribute": hhs,
      "width distribute": wws,
  }
  for ax, kk in zip(axes, pp.keys()):
      _ = ax.hist(pp[kk], bins=1000, label=kk)
      ax.set_title("[with attempt] " + kk)
  fig.tight_layout()
  ```
  **TF function**
  ```py
  def random_crop_fraction_timm(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), compute_dtype="float32"):
      size = tf.shape(image)
      height, width = tf.cast(size[0], dtype=compute_dtype), tf.cast(size[1], dtype=compute_dtype)
      area = height * width
      in_ratio = width / height

      target_areas = tf.random.uniform((10,), scale[0], scale[1]) * area
      log_min, log_max = tf.math.log(ratio[0]), tf.math.log(ratio[1])
      aspect_ratios = tf.random.uniform((10,), log_min, log_max, dtype=compute_dtype)
      aspect_ratios = tf.math.exp(aspect_ratios)

      ww_crops, hh_crops = tf.sqrt(target_areas * aspect_ratios), tf.sqrt(target_areas / aspect_ratios)
      pick = tf.argmax(tf.logical_and(hh_crops <= height, ww_crops <= width))
      hh_crop = tf.cast(tf.math.floor(hh_crops[pick]), "int32")
      ww_crop = tf.cast(tf.math.floor(ww_crops[pick]), "int32")
      # return hh_crop, ww_crop
      return tf.cond(
          tf.logical_and(hh_crop <= size[0], ww_crop <= size[1]),
          lambda: tf.image.random_crop(image, (hh_crop, ww_crop, 3)),
          lambda: tf.image.central_crop(image, tf.minimum(tf.minimum(ratio[1] / in_ratio, in_ratio * ratio[0]), scale[1])),
      )
  ```
***

# TF Imagenet Basic
## TF Training
  ```py
  keras.mixed_precision.set_global_policy("mixed_float16")
  strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

  from keras_cv_attention_models.imagenet import data
  from keras_cv_attention_models.imagenet import callbacks
  from keras_cv_attention_models import model_surgery
  from keras_cv_attention_models import aotnet, coatnet, cmt
  import tensorflow_addons as tfa

  input_shape = (160, 160, 3)
  batch_size = 256 * strategy.num_replicas_in_sync
  lr_base_512 = 8e-3
  l2_weight_decay = 0
  optimizer_wd_mul = 0.02
  label_smoothing = 0
  lr_decay_steps = 100 # [30, 60, 90] for constant decay
  lr_warmup = 5
  lr_min = 1e-6
  epochs = 105
  initial_epoch = 0
  basic_save_name = None

  data_name = "imagenet2012"
  magnitude = 6
  mixup_alpha = 0.1
  cutmix_alpha = 1.0
  central_crop = 1
  random_crop_min = 0.08

  train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
      data_name=data_name,
      input_shape=input_shape,
      batch_size=batch_size,
      mixup_alpha=mixup_alpha,
      cutmix_alpha=cutmix_alpha,
      rescale_mode="torch",
      central_crop=central_crop,
      random_crop_min=random_crop_min,
      resize_method="bicubic",
      random_erasing_prob=0.0,
      magnitude=magnitude,
  )

  # model = coatnet.CoAtNet0(num_classes=1000, activation='gelu', drop_connect_rate=0.2, drop_rate=0.2)
  # model = cmt.CMTTiny(input_shape=input_shape, num_classes=num_classes, drop_connect_rate=0.1, drop_rate=0.1)
  model = aotnet.AotNet50(num_classes=num_classes, input_shape=input_shape)
  # model = keras.models.load_model('checkpoints/resnet50_imagenet2012_batch_size_256_randaug_5_mixup_0.1_cutmix_1.0_RRC_0.08_LAMB_lr0.002_wd0.02_latest.h5')

  lr_base = lr_base_512 * batch_size / 512
  if isinstance(lr_decay_steps, list):
      constant_lr_sch = lambda epoch: callbacks.constant_scheduler(epoch, lr_base=lr_base, lr_decay_steps=lr_decay_steps, warmup=lr_warmup)
      lr_scheduler = keras.callbacks.LearningRateScheduler(constant_lr_sch)
      epochs = epochs if epochs != 0 else lr_decay_steps[-1] + lr_decay_steps[0] + lr_warmup   # 124 for lr_decay_steps=[30, 60, 90], lr_warmup=4
  else:
      lr_scheduler = callbacks.CosineLrScheduler(lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, warmup=lr_warmup, steps_per_epoch=-1)
      # lr_scheduler = callbacks.CosineLrSchedulerEpoch(lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, warmup=lr_warmup)
      epochs = epochs if epochs != 0 else lr_decay_steps * 3 + lr_warmup  # 94 for lr_decay_steps=30, lr_warmup=4

  if model.optimizer is None:
      if l2_weight_decay != 0:
          model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=l2_weight_decay, apply_to_batch_normal=False)

      if optimizer_wd_mul > 0:
          # optimizer = tfa.optimizers.AdamW(learning_rate=lr_base, weight_decay=lr_base * optimizer_wd_mul)
          optimizer = tfa.optimizers.LAMB(learning_rate=lr_base, weight_decay_rate=optimizer_wd_mul)
      else:
          optimizer = keras.optimizers.SGD(learning_rate=lr_base, momentum=0.9)
      model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), metrics=["acc"])

  compiled_opt = model.optimizer
  compiled_opt = compiled_opt.inner_optimizer if isinstance(compiled_opt, keras.mixed_precision.LossScaleOptimizer) else compiled_opt
  if basic_save_name is None:
      basic_save_name = "{}_{}_batch_size_{}".format(model.name, data_name, batch_size)
      basic_save_name += "_randaug_{}_mixup_{}_cutmix_{}_RRC_{}".format(magnitude, mixup_alpha, cutmix_alpha, random_crop_min)
      basic_save_name += "_{}_lr{}_wd{}".format(compiled_opt.__class__.__name__, lr_base_512, optimizer_wd_mul or l2_weight_decay)
  print(">>>> basic_save_name =", basic_save_name)

  """ imagenet.train.train """
  if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch == -1:
      lr_scheduler.build(steps_per_epoch)
  is_lr_on_batch = True if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch > 0 else False

  # ckpt_path = os.path.join("checkpoints", basic_save_name + "epoch_{epoch:03d}_val_acc_{val_acc:.4f}.h5")
  # cur_callbacks = [keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)]
  # cur_callbacks = [keras.callbacks.ModelCheckpoint(os.path.join("checkpoints", basic_save_name + ".h5"))]
  cur_callbacks = [callbacks.MyCheckpoint(basic_save_name, monitor='val_acc')]
  hist_file = os.path.join("checkpoints", basic_save_name + "_hist.json")
  if initial_epoch == 0 and os.path.exists(hist_file):
      os.remove(hist_file)
    cur_callbacks.append(callbacks.MyHistory(initial_file=hist_file))
    cur_callbacks.append(keras.callbacks.TerminateOnNaN())
    if lr_scheduler is not None:
        cur_callbacks.append(lr_scheduler)

    if lr_scheduler is not None and isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
        print(">>>> Append weight decay callback...")
        lr_base, wd_base = model.optimizer.lr.numpy(), model.optimizer.weight_decay.numpy()
        wd_callback = callbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=is_lr_on_batch)
        cur_callbacks.append(wd_callback)  # should be after lr_scheduler

    model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=cur_callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        use_multiprocessing=True,
        workers=4,
    )
    ```
    ```sh
    TF_XLA_FLAGS="--tf_xla_auto_jit=2" CUDA_VISIBLE_DEVICES='0' ./train_script.py --random_crop_min 0
    CUDA_VISIBLE_DEVICES='0' ./train_script.py -r checkpoints/aotnet50_imagenet2012_batch_size_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_LAMB_lr0.008_wd0.02_latest.h5 --initial_epoch 86 --random_crop_min 0.08
    CUDA_VISIBLE_DEVICES='1' ./train_script.py -r checkpoints/aotnet50_imagenet2012_batch_size_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.0_LAMB_lr0.008_wd0.02_latest.h5 --initial_epoch 95 --random_crop_min 0
    ```
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0 --resize_antialias --bce_threshold 0.2 --input_shape 224 --lr_base_512 5e-3 --magnitude 7 --batch_size 128 --lr_decay_steps 300 --epochs 305 -s aotnet.AotNet50_A2 --additional_model_kwargs '{"drop_connect_rate": 0.05}'
    ```
    ```sh
    watch -n 10 sh -c "nvidia-smi > /dev/null || ssh leondgarse@192.168.16.189 -C 'notify-send --urgency=low \"[83 Error] nvidia fall\"'"
    ```
    ```sh
    while true; do if [ $(nvidia-smi | grep MiB | sed -n '1p' | cut -d '|' -f 3 | cut -d 'M' -f 1) -lt 5000 ]; then notifyme "CUDA0_done"; CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py -m coatnet.CoAtNet0 --seed 0 --resize_antialias --bce_threshold 0.2 --batch_size 128 -s aaa ; break ; fi; sleep 10m; done
    while true; do if [ $(nvidia-smi | grep MiB | sed -n '2p' | cut -d '|' -f 3 | cut -d 'M' -f 1) -lt 5000 ]; then notifyme "CUDA1_done"; CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py -m coatnet.CoAtNet0 --seed 0 --resize_antialias --bce_threshold 0.2 --batch_size 128 -s aaa ; break ; fi; sleep 10m; done
    ```
## Pretrained Model Validation
  ```py
  sys.path.append('../automl/efficientnetv2/')
  import datasets as orign_datasets
  import effnetv2_model as orign_effnetv2_model

  dataset = "imagenetft"
  model_type = "s"
  load_weights = "imagenet21k-ft1k"
  cc = orign_datasets.get_dataset_config(dataset)
  if cc.get("model", None):
      cc.model.num_classes = cc.data.num_classes
  else:
      cc['model'] = None
  model = orign_effnetv2_model.get_model('efficientnetv2-{}'.format(model_type), model_config=cc.model, weights=load_weights)

  from keras_cv_attention_models import imagenet
  imagenet.evaluation(model, input_shape=(384, 384), rescale_mode="tf", resize_method='bicubic', central_crop=0.99)
  ```
  ```py
  import timm
  mm = timm.models.regnetz_d(pretrained=True)
  _ = mm.evaluation()
  from keras_cv_attention_models import imagenet
  imagenet.evaluation(mm, input_shape=(256, 256, 3), rescale_mode="tf", resize_method="bicubic", central_crop=0.95)
  ```
  ```py
  import keras_cv_attention_models
  from keras_cv_attention_models import imagenet
  mm = keras_cv_attention_models.regnet.RegNetZD()
  imagenet.evaluation(mm, input_shape=(256, 256, 3), rescale_mode="tf", resize_method="bicubic", central_crop=0.95)
  ```
## EfficientNetV2 eval results
  | model   | Input Resolution | rescale_mode | resize_method     | central_crop | top 1       | top 5       |
  | ------- | ---------------- | ------------ | ----------------- | ------------ | ----------- | ----------- |
  | EffV2B0 | 224              | tf           | bilinear          | 0.875        | 0.75304     | 0.92606     |
  | EffV2B0 | 224              | tf           | bicubic           | 0.875        | 0.7581      | 0.92858     |
  | EffV2B0 | 224              | torch        | bilinear          | 0.875        | 0.78594     | 0.94262     |
  | EffV2B0 | 224              | torch        | bicubic           | 0.875        | **0.78748** | 0.94386     |
  | EffV2B0 | 224              | torch        | bicubic antialias | 0.875        | 0.7833      | 0.94038     |
  | EffV2B0 | 224              | torch        | bicubic           | 0.87         | 0.787       | 0.9436      |
  | EffV2B0 | 224              | torch        | bicubic           | 0.95         | 0.78732     | **0.94412** |
  | EffV2B0 | 224              | torch        | bicubic antialias | 0.95         | 0.78288     | 0.93998     |
  | EffV2B0 | 224              | torch        | bicubic           | 1.0          | 0.78536     | 0.94404     |

  | model       | Input Resolution | rescale_mode | resize_method     | central_crop | top 1       | top 5       |
  | ----------- | ---------------- | ------------ | ----------------- | ------------ | ----------- | ----------- |
  | EffV2B2,21k | 260              | torch        | bicubic           | 0.95         | 0.19478     | 0.31922     |
  | EffV2B2,21k | 260              | tf           | bicubic           | 0.95         | 0.79118     | 0.94922     |
  | EffV2B2,21k | 260              | tf           | bicubic antialias | 0.95         | 0.79254     | 0.94992     |
  | EffV2B2,21k | 260              | tf           | bicubic antialias | 0.99         | **0.79482** | **0.95022** |
  |             |                  |              |                   |              |             |             |
  | EffV2B1,21k | 240              | tf           | bicubic antialias | 0.99         | **0.79034** | **0.94978** |
  | EffV2B1,21k | 240              | tf           | bicubic antialias | 0.87         | 0.7819      | 0.9436      |
  |             |                  |              |                   |              |             |             |
  | EffV2B0,21k | 224              | tf           | bicubic antialias | 0.95         | 0.77418     | 0.94134     |
  | EffV2B0,21k | 224              | tf           | bicubic antialias | 0.875        | 0.76786     | 0.93682     |
  | EffV2B0,21k | 224              | tf           | bicubic antialias | 0.99         | **0.7755**  | **0.94208** |
  |             |                  |              |                   |              |             |             |
  | EffV2B3,21k | 224              | tf           | bicubic antialias | 0.99         | **0.82462** | **0.96524** |

  | model     | Input Resolution | rescale_mode | resize_method     | central_crop     | top 1       | top 5       |
  | --------- | ---------------- | ------------ | ----------------- | ---------------- | ----------- | ----------- |
  | EffV2B1   | 240              | torch        | bicubic           | 0.87             | 0.79722     | **0.94938** |
  | EffV2B1   | 240              | torch        | bicubic           | 240/272 (0.882)  | 0.79788     | 0.94924     |
  | EffV2B1   | 240              | torch        | bicubic           | 0.95             | **0.7987**  | 0.94936     |
  |           |                  |              |                   |                  |             |             |
  | EffV2B2   | 260              | torch        | bicubic           | 260/292 (0.890)  | 0.80428     | 0.95184     |
  | EffV2B2   | 260              | torch        | bicubic           | 0.95             | **0.80642** | **0.95262** |
  |           |                  |              |                   |                  |             |             |
  | EffV2B3   | 300              | torch        | bicubic           | 300/332 (0.9036) | 0.81974     | 0.95818     |
  | EffV2B3   | 300              | torch        | bicubic           | 0.95             | **0.82098** | **0.95896** |
  | EffV2B3   | 300              | torch        | bicubic           | 1.0              | 0.82        | 0.9587      |
  |           |                  |              |                   |                  |             |             |
  | EffV2T    | 288              | torch        | bicubic           | 0.99             | 0.82186     | 0.96112     |
  | EffV2T    | 288              | torch        | bicubic antialias | 0.99             | 0.82324     | 0.96114     |
  | EffV2T    | 288              | torch        | bicubic antialias | 1.0              | 0.82272     | **0.96206** |
  | EffV2T    | 288              | torch        | bicubic antialias | 0.95             | **0.82338** | 0.96072     |
  | EffV2T    | 320              | torch        | bicubic antialias | 0.99             | 0.82588     | **0.96238** |
  | EffV2T    | 320              | torch        | bicubic antialias | 0.95             | **0.82594** | 0.962       |
  |           |                  |              |                   |                  |             |             |
  | EffV2T_GC | 288              | torch        | bicubic antialias | 0.95             | 0.82404     | 0.96236     |
  | EffV2T_GC | 288              | torch        | bicubic antialias | 0.99             | 0.82394     | 0.96282     |
  | EffV2T_GC | 288              | torch        | bicubic antialias | 1.0              | **0.82458** | **0.96306** |
  | EffV2T_GC | 320              | torch        | bicubic antialias | 1.0              | **0.82676** | **0.96372** |

  | model       | Input Resolution | rescale_mode | resize_method     | central_crop | top 1       | top 5       |
  | ----------- | ---------------- | ------------ | ----------------- | ------------ | ----------- | ----------- |
  | EffV2S      | 384              | tf           | bicubic           | 0.95         | 0.83788     | 0.96602     |
  | EffV2S      | 384              | torch        | bicubic           | 0.95         | 0.29858     | 0.46904     |
  | EffV2S      | 384              | tf           | bicubic           | 0.99         | 0.8386      | 0.967       |
  | EffV2S      | 384              | tf           | bicubic antialias | 0.99         | 0.8387      | 0.967       |
  | EffV2S      | 384              | tf           | bicubic antialias | -1           | 0.83764     | 0.96652     |
  | EffV2S      | 384              | [128, 128]   | bicubic antialias | 0.99         | 0.8389      | 0.96712     |
  | EffV2S      | 384              | [128, 128]   | bicubic antialias | 1.0          | **0.83892** | **0.96726** |
  |             |                  |              |                   |              |             |             |
  | EffV2S ft1k | 384              | tf           | bicubic           | 0.94         | 0.84006     | 0.97118     |
  | EffV2S ft1k | 384              | torch        | bicubic           | 0.94         | 0.13104     | 0.21656     |
  | EffV2S ft1k | 384              | tf           | bicubic           | 0.95         | 0.84076     | 0.97134     |
  | EffV2S ft1k | 384              | tf           | bicubic           | 0.96         | 0.8417      | 0.97176     |
  | EffV2S ft1k | 384              | tf           | bicubic           | 0.97         | 0.84188     | 0.972       |
  | EffV2S ft1k | 384              | tf           | bicubic           | 0.98         | 0.84302     | 0.9727      |
  | EffV2S ft1k | 384              | tf           | bicubic           | 0.99         | 0.84328     | 0.97254     |
  | EffV2S ft1k | 384              | tf           | bicubic antialias | 0.99         | 0.84336     | 0.97256     |
  | EffV2S ft1k | 384              | tf           | bicubic           | 1.0          | 0.84312     | 0.97292     |
  | EffV2S ft1k | 384              | tf           | bicubic antialias | 1.0          | 0.84348     | 0.97288     |
  | EffV2S ft1k | 384              | tf           | bicubic           | -1           | 0.84794     | 0.97494     |
  | EffV2S ft1k | 384              | [128, 128]   | bicubic           | -1           | 0.84796     | 0.9751      |
  | EffV2S ft1k | 384              | tf           | bicubic antialias | -1           | 0.84802     | 0.97504     |
  | EffV2S ft1k | 384              | [128, 128]   | bicubic antialias | -1           | **0.84804** | **0.97514** |

  | model        | Input Resolution | rescale_mode | resize_method     | central_crop | top 1       | top 5       |
  | ------------ | ---------------- | ------------ | ----------------- | ------------ | ----------- | ----------- |
  | EffV2M       | 480              | tf           | bicubic           | 0.99         | **0.8509**  | 0.973       |
  | EffV2M       | 480              | tf           | bicubic antialias | 0.99         | 0.85086     | 0.9731      |
  | EffV2M       | 480              | [128, 128]   | bicubic antialias | 0.99         | 0.8508      | **0.97314** |
  | EffV2M       | 480              | [128, 128]   | bicubic antialias | 1.0          | 0.8503      | 0.9728      |
  |              |                  |              |                   |              |             |             |
  | EffV2L       | 480              | tf           | bicubic           | 0.99         | **0.855**   | **0.97324** |
  | EffV2L       | 480              | tf           | bicubic antialias | 0.99         | 0.85478     | 0.9732      |
  |              |                  |              |                   |              |             |             |
  | EffV2M ft1k  | 480              | tf           | bicubic           | 0.99         | 0.85606     | 0.9775      |
  | EffV2M ft1k  | 480              | tf           | bicubic           | -1           | 0.86082     | 0.9795      |
  | EffV2M ft1k  | 480              | [128, 128]   | bicubic           | -1           | **0.8609**  | **0.97952** |
  | EffV2M ft1k  | 480              | [128, 128]   | bicubic antialias | -1           | 0.86068     | 0.97952     |
  |              |                  |              |                   |              |             |             |
  | EffV2L ft1k  | 480              | tf           | bicubic           | 0.99         | 0.86294     | 0.9799      |
  | EffV2L ft1k  | 480              | tf           | bicubic           | -1           | 0.86824     | **0.98162** |
  | EffV2L ft1k  | 480              | tf           | bicubic antialias | -1           | 0.86824     | 0.98148     |
  | EffV2L ft1k  | 480              | [128, 128]   | bicubic           | -1           | 0.86818     | 0.98162     |
  | EffV2L ft1k  | 480              | [128, 128]   | bicubic antialias | -1           | **0.8683**  | 0.9815      |
  |              |                  |              |                   |              |             |             |
  | EffV2XL ft1k | 512              | tf           | bicubic           | 0.99         | 0.86532     | 0.97866     |
  | EffV2XL ft1k | 512              | tf           | bicubic           | -1           | 0.86772     | **0.98066** |
  | EffV2XL ft1k | 512              | [128, 128]   | bicubic           | -1           | 0.86782     | 0.9806      |
  | EffV2XL ft1k | 512              | [128, 128]   | bicubic antialias | -1           | **0.86788** | 0.9806      |
## HaloNet AotNet RegNet eval results
  | model        | rescale_mode        | resize_method | central_crop | top 1   | top 5   | Reported |
  | ------------ | ------------------- | ------------- | ------------ | ------- | ------- | -------- |
  | halonet26t   | torch               | bicubic       | 0.95         | 0.78588 | 0.94078 | 79.134   |
  | halonet26t   | tf                  | bicubic       | 0.95         | 0.7766  | 0.93588 |          |
  | halonet50t   | torch               | bicubic       | 0.94         | 0.81134 | 0.95202 | 81.350   |
  | halonet50t   | tf                  | bicubic       | 0.94         | 0.8068  | 0.9493  |          |
  | aotnet50     | torch (resize-crop) | bicubic       | 0.95         | 0.79292 | 0.94126 | 80.4     |
  | aotnet50     | torch (crop-resize) | bicubic       | 0.95         | 0.79384 | 0.94164 |          |
  | aotnet50     | torch (resize-crop) | bicubic       | 0.875        | 0.7938  | 0.94148 |          |
  | aotnet50     | torch (crop-resize) | bicubic       | 0.875        | 0.79444 | 0.94222 |          |
  | aotnet50     | tf                  | bicubic       | 0.95         | 0.78208 | 0.93604 |          |
  | aotnet50 160 | torch               | bicubic       | 0.95         | 0.72594 | 0.90654 |          |
  | aotnet50 160 | torch               | bicubic       | 0.875        | 0.73132 | 0.91042 |          |
  | aotnet50 160 | torch (224)         | bicubic       | 0.95         | 0.7668  | 0.93066 | 78.1     |
  | aotnet50 160 | torch (224)         | bicubic       | 0.875        | 0.7674  | 0.93    |          |
  | aotnet50 160 | tf (224)            | bicubic       | 0.95         | 0.73926 | 0.9148  |          |
  | regnetzd 256 | tf                  | bicubic       | 0.95         | 0.83462 | 0.96658 | 83.422   |
  | regnetzd 256 | torch               | bicubic       | 0.95         | 0.80944 | 0.95612 |          |
  | regnetzd 160 | tf                  | bicubic       | 0.95         | 0.78004 | 0.93908 |          |

  | model      | rescale_mode | resize_method | central_crop | top 1   | top 5   | Reported |
  | ---------- | ------------ | ------------- | ------------ | ------- | ------- | -------- |
  | RegNetY032 | torch (288)  | bicubic       | 0.875        | 0.82446 | 0.96248 | 82.722   |
  | RegNetY040 | torch        | bicubic       | 0.875        | 0.80568 | 0.9512  | 81.5     |
  | RegNetY040 | tf           | bicubic       | 0.875        | 0.80108 | 0.9478  |          |
  | RegNetY080 | torch        | bicubic       | 0.875        | 0.8148  | 0.94876 | 82.2     |
  | RegNetY160 | torch        | bicubic       | 0.875        | 0.81432 | 0.94456 | 82.0     |
  | RegNetY320 | torch        | bicubic       | 0.875        | 0.81738 | 0.94152 | 82.5     |

  | model              | rescale_mode        | resize_method      | central_crop | top 1   | top 5   |
  | ------------------ | ------------------- | ------------------ | ------------ | ------- | ------- |
  | timm resnet50 (CE) | torch (clip 255)    | bicubic antialias  | 0.95         | 0.77    | 0.93722 |
  | timm resnet50 (CE) | torch               | bicubic antialias  | 0.95         | 0.77026 | 0.93716 |
  | timm resnet50 (CE) | torch               | bilinear antialias | 0.95         | 0.76898 | 0.93746 |
  | timm resnet50 (CE) | torch               | bilinear           | 0.95         | 0.76404 | 0.93412 |
  | AotNet50 (CE)      | torch               | bilinear           | 0.95         | 0.7694  | 0.93704 |
  | AotNet50 (CE)      | torch               | bicubic            | 0.95         | 0.77004 | 0.93702 |
  | AotNet50 (CE)      | torch               | bicubic antialias  | 0.95         | 0.7647  | 0.9335  |
  | AotNet50 (CE)      | torch (clip 255)    | bicubic            | 0.95         | 0.76996 | 0.9371  |
  | AotNet50 (CE)      | torch (resize_crop) | bicubic            | 0.95         | 0.76956 | 0.93696 |
***
