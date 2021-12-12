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

# OCR
  ```sh
  docker run -it -p 8866:8866 paddleocr:cpu bash
  git clone https://gitee.com/PaddlePaddle/PaddleOCR
  sed -i 's/ch_det_mv3_db/ch_det_r50_vd_db/' deploy/hubserving/ocr_system/params.py
  sed -i 's/ch_rec_mv3_crnn/ch_rec_r34_vd_crnn_enhance/' deploy/hubserving/ocr_system/params.py
  export PYTHONPATH=. && hub uninstall ocr_system; hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system

  OCR_DID=`docker ps -a | sed -n '2,2p' | cut -d ' ' -f 1`
  docker cp ch_det_r50_vd_db_infer/* $OCR_DID:/PaddleOCR/inference/
  docker cp ch_rec_r34_vd_crnn_enhance_infer/* $OCR_DID:/PaddleOCR/inference/

  ```
  ```sh
  IMG_STR=`base64 -w 0 $TEST_PIC`
  echo "{\"images\": [\"`base64 -w 0 Selection_261.png`\"]}" > foo
  curl -H "Content-Type:application/json" -X POST --data "{\"images\": [\"填入图片Base64编码(需要删除'data:image/jpg;base64,'）\"]}" http://localhost:8866/predict/ocr_system
  curl -H "Content-Type:application/json" -X POST --data "{\"images\": [\"`base64 -w 0 Selection_101.png`\"]}" http://localhost:8866/predict/ocr_system
  curl -H "Content-Type:application/json" -X POST --data foo http://localhost:8866/predict/ocr_system
  echo "{\"images\": [\"`base64 -w 0 Selection_261.png`\"]}" | curl -v -X PUT -H 'Content-Type:application/json' -d @- http://localhost:8866/predict/ocr_system
  ```
  ```py
  import requests
  import base64
  import json
  from matplotlib.font_manager import FontProperties

  class PaddleOCR:
      def __init__(self, url, font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc'):
          self.url = url
          self.font = FontProperties(fname=font)
      def __call__(self, img_path, thresh=0.9, show=2):
          with open(img_path, 'rb') as ff:
              aa = ff.read()
          bb = base64.b64encode(aa).decode()
          rr = requests.post(self.url, headers={"Content-type": "application/json"}, data='{"images": ["%s"]}' % bb)
          dd = json.loads(rr.text)

          imm = imread(img_path)
          if show == 0:
              return dd
          if show == 1:
              fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
              axes = [axes, axes]
          elif show == 2:
              fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
              axes[1].invert_yaxis()
          axes[0].imshow(imm)
          for ii in dd['results'][0]:
              jj = np.array(ii['text_region'])
              kk = np.vstack([jj, jj[0]])
              axes[0].plot([ii[0] for ii in kk], [ii[1] for ii in kk], 'r')
              axes[1].text(jj[-1, 0], jj[-1, 1], ii['text'], fontproperties=font, fontsize=(jj[-1, 1]-jj[0, 1])/2)
          # plt.tight_layout()
          return dd

  pp = PaddleOCR("http://localhost:8866/predict/ocr_system")
  pp("./Selection_261.png")
  ```
  ```sh
  python3 tools/infer/predict_system.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/"  --rec_model_dir="./inference/rec_crnn/"
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/elementwise_layers.py
  def convert_reciprocal(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert element-wise division
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """     
      logger = logging.getLogger('onnx2keras:reciprocal')
      print(layers[node.input[0]])

      if len(node.input) != 1:
          assert AttributeError('Not 1 input for reciprocal layer.')

      layers[node_name] = 1 / layers[node.input[0]]
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/upsampling_layers.py
  def convert_resize(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert upsample.
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """
      logger = logging.getLogger('onnx2keras:resize')
      logger.warning('!!! EXPERIMENTAL SUPPORT (resize) !!!')
      print([layers[ii] for ii in node.input])

      if len(node.input) != 1:
          if node.input[-1] in layers and isinstance(layers[node.input[-1]], np.ndarray):
              params['scales'] = layers[node.input[-1]]
          else:
              raise AttributeError('Unsupported number of inputs')

      if params['mode'].decode('utf-8') != 'nearest':
          logger.error('Cannot convert non-nearest upsampling.')
          raise AssertionError('Cannot convert non-nearest upsampling')

      scale = np.uint8(params['scales'][-2:])

      upsampling = keras.layers.UpSampling2D(
          size=scale, name=keras_name
      )   

      layers[node_name] = upsampling(layers[node.input[0]])
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/operation_layers.py
  def convert_clip(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert clip layer
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """
      logger = logging.getLogger('onnx2keras:clip')
      if len(node.input) != 1:
          assert AttributeError('More than 1 input for clip layer.')

      input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
      print(node.input, [layers[ii] for ii in node.input], node_name, params)
      if len(node.input == 3):
          params['min'] = layers[node.input[1]]
          params['max'] = layers[node.input[2]]

      if params['min'] == 0:
          logger.debug("Using ReLU({0}) instead of clip".format(params['max']))
          layer = keras.layers.ReLU(max_value=params['max'], name=keras_name)
      else:
          def target_layer(x, vmin=params['min'], vmax=params['max']):
              import tensorflow as tf
              return tf.clip_by_value(x, vmin, vmax)
          layer = keras.layers.Lambda(target_layer, name=keras_name)
          lambda_func[keras_name] = target_layer

      layers[node_name] = layer(input_0)
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/activation_layers.py
  def convert_hard_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert Sigmoid activation layer
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """
      if len(node.input) != 1:
          assert AttributeError('More than 1 input for an activation layer.')

      input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
      hard_sigmoid = keras.layers.Activation(keras.activations.hard_sigmoid, name=keras_name)
      layers[node_name] = hard_sigmoid(input_0)
  ```
  ```py
  from .activation_layers import convert_hard_sigmoid
  from .elementwise_layers import convert_reciprocal
  from .upsampling_layers import convert_resize
  ```
***
## 图神经网络
  - [Github /dmlc/dgl](https://github.com/dmlc/dgl)
***

# Imagenet
## Timm
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
CUDA_VISIBLE_DEVICES='0' python validate.py $HOME/tensorflow_datasets/ --dataset tfds/imagenet2012 --img-size 224 --crop-pct 0.95 --model resnet50 --checkpoint ./output/train/20211109-111121-resnet50-160/checkpoint-99.pth.tar
# * Acc@1 78.066 (21.934) Acc@5 93.680 (6.320)
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
## Init dataset
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
## Training
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
watch -n 10 sh -c "nvidia-smi > /dev/null || ssh leondgarse@192.168.16.189 -C 'notify-send --urgency=low \"[83 Error] nvidia fall\"'"
```
## Validation
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

| model   | Input Resolution | rescale_mode | resize_method | central_crop     | top 1       | top 5       |
| ------- | ---------------- | ------------ | ------------- | ---------------- | ----------- | ----------- |
| EffV2B1 | 240              | torch        | bicubic       | 0.87             | 0.79722     | **0.94938** |
| EffV2B1 | 240              | torch        | bicubic       | 240/272 (0.882)  | 0.79788     | 0.94924     |
| EffV2B1 | 240              | torch        | bicubic       | 0.95             | **0.7987**  | 0.94936     |
| EffV2B2 | 260              | torch        | bicubic       | 260/292 (0.890)  | 0.80428     | 0.95184     |
| EffV2B2 | 260              | torch        | bicubic       | 0.95             | **0.80642** | **0.95262** |
| EffV2B3 | 300              | torch        | bicubic       | 300/332 (0.9036) | 0.81974     | 0.95818     |
| EffV2B3 | 300              | torch        | bicubic       | 0.95             | **0.82098** | **0.95896** |
| EffV2B3 | 300              | torch        | bicubic       | 1.0              | 0.82        | 0.9587      |
| EffV2T  | 288              | torch        | bicubic       | 0.99             | 0.82186     | 0.96112     |
| EffV2T  | 320              | torch        | bicubic       | 0.99             | **0.82506** | **0.96228** |

| model       | Input Resolution | rescale_mode | resize_method     | central_crop | top 1       | top 5       |
| ----------- | ---------------- | ------------ | ----------------- | ------------ | ----------- | ----------- |
| EffV2S      | 384              | tf           | bicubic           | 0.95         | 0.83788     | 0.96602     |
| EffV2S      | 384              | torch        | bicubic           | 0.95         | 0.29858     | 0.46904     |
| EffV2S      | 384              | tf           | bicubic           | 0.99         | **0.8386**  | **0.967**   |
| EffV2S      | 384              | tf           | bicubic antialias | 0.99         | 0.8387      | 0.967       |
| EffV2S ft1k | 384              | tf           | bicubic           | 0.94         | 0.84006     | 0.97118     |
| EffV2S ft1k | 384              | torch        | bicubic           | 0.94         | 0.13104     | 0.21656     |
| EffV2S ft1k | 384              | tf           | bicubic           | 0.95         | 0.84076     | 0.97134     |
| EffV2S ft1k | 384              | tf           | bicubic           | 0.96         | 0.8417      | 0.97176     |
| EffV2S ft1k | 384              | tf           | bicubic           | 0.97         | 0.84188     | 0.972       |
| EffV2S ft1k | 384              | tf           | bicubic           | 0.98         | 0.84302     | 0.9727      |
| EffV2S ft1k | 384              | tf           | bicubic           | 0.99         | 0.84328     | 0.97254     |
| EffV2S ft1k | 384              | tf           | bicubic antialias | 0.99         | 0.84336     | 0.97256     |
| EffV2S ft1k | 384              | tf           | bicubic           | 1.0          | 0.84312     | **0.97292** |
| EffV2S ft1k | 384              | tf           | bicubic antialias | 1.0          | **0.84348** | 0.97288     |

| model        | Input Resolution | rescale_mode | resize_method     | central_crop | top 1   | top 5   |
| ------------ | ---------------- | ------------ | ----------------- | ------------ | ------- | ------- |
| EffV2M       | 480              | tf           | bicubic           | 0.99         | 0.8509  | 0.973   |
| EffV2M       | 480              | tf           | bicubic antialias | 0.99         | 0.85086 | 0.9731  |
| EffV2L       | 480              | tf           | bicubic           | 0.99         | 0.855   | 0.97324 |
| EffV2L       | 480              | tf           | bicubic antialias | 0.99         |         |         |
| EffV2M ft1k  | 480              | tf           | bicubic           | 0.99         | 0.85606 | 0.9775  |
| EffV2M ft1k  | 480              | tf           | bicubic           | 0.99         | 0.85606 | 0.9775  |
| EffV2XL ft1k | 512              | tf           | bicubic           | 0.99         | 0.86532 | 0.97866 |

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
| regnetzd 256 | tf                  | bicubic       | 0.95         | 0.83244 | 0.96636 | 84.034   |
| regnetzd 256 | torch               | bicubic       | 0.95         | 0.80944 | 0.95612 |          |

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

  ```py
  import sys, os, time, re, gc
  from pathlib import Path
  from glob import glob

  import tensorflow as tf
  import numpy as np
  import matplotlib.pyplot as plt

  from tensorflow.keras import backend as K
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.applications import vgg16, vgg19, resnet_v2

  path_imagenet_val_dataset = Path("data/") # path/to/data/
  dir_images = Path("data/val") # path/to/images/directory
  path_labels = Path("data/ILSVRC2012_validation_ground_truth.txt")
  path_synset_words = Path("data/synset_words.txt")
  path_meta = Path("data/meta.mat")

  x_val_paths = glob(str(path_imagenet_val_dataset / "x_val*.npy"))

  # Sort filenames in ascending order
  x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
  y_val = np.load(str(path_imagenet_val_dataset / "y_val.npy"))
  y_val_one_hot = to_categorical(y_val, 1000)
  def top_k_accuracy(y_true, y_pred, k=1, tf_enabled=True):
      if tf_enabled:
          argsorted_y = tf.argsort(y_pred)[:,-k:]
          matches = tf.cast(tf.math.reduce_any(tf.transpose(argsorted_y) == tf.argmax(y_true, axis=1, output_type=tf.int32), axis=0), tf.float32)
          return tf.math.reduce_mean(matches).numpy()
      else:
          argsorted_y = np.argsort(y_pred)[:,-k:]
          return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()

  K.clear_session()
  # model = vgg19.VGG19()
  import keras_efficientnet_v2
  model = keras_efficientnet_v2.EfficientNetV2B0()

  y_pred = None
  for i, x_val_path in enumerate(x_val_paths):
      x_val = np.load(x_val_path).astype('float32') # loaded as RGB
      x_val = vgg19.preprocess_input(x_val) # converted to BGR
      y_pred_sharded = model.predict(x_val, verbose=0, use_multiprocessing=True, batch_size=16, callbacks=None)

      try:
          y_pred = np.concatenate([y_pred, y_pred_sharded])
      except ValueError:
          y_pred = y_pred_sharded

      del x_val
      gc.collect()

      completed_percentage = (i + 1) * 100 / len(x_val_paths)
      if completed_percentage % 5 == 0:
          print("{:5.1f}% completed.".format(completed_percentage))

  print(top_k_accuracy(y_val_one_hot, y_pred, k=1))
  # 0.71248
  print(top_k_accuracy(y_val_one_hot, y_pred, k=5))
  ```
```py
class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, first_restart_step, steps_per_epoch, m_mul=0.5, t_mul=2.0, lr_min=1e-5, warmup=0):
        super(CosineLrScheduler, self).__init__()
        self.warmup = warmup * steps_per_epoch
        self.first_restart_step = first_restart_step * steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.init_step_num, self.cur_epoch = 0, 0

        if lr_min == lr_base * m_mul: # Without restart
            self.schedule = keras.experimental.CosineDecay(lr_base, self.first_restart_step, alpha=lr_min / lr_base)
        else:
            self.schedule = keras.experimental.CosineDecayRestarts(lr_base, self.first_restart_step, t_mul=t_mul, m_mul=m_mul, alpha=lr_min / lr_base)

        if warmup != 0:
            self.warmup_lr_func = lambda ii: lr_min + (lr_base - lr_min) * ii / self.warmup

    def on_train_batch_begin(self, cur_epoch, logs=None):
        self.init_step_num = int(self.steps_per_epoch * cur_epoch)
        self.cur_epoch = cur_epoch

    def on_train_batch_begin(self, iterNum, logs=None):
        global_iterNum = iterNum + self.init_step_num
        if global_iterNum < self.warmup:
            lr = self.warmup_lr_func(global_iterNum)
        else:
            lr = self.schedule(global_iterNum - self.warmup)

        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        if iterNum == 0:
            print("\nLearning rate for iter {} is {}".format(self.cur_epoch + 1, lr))
        return lr

def constant_scheduler(epoch, lr_base, lr_decay_steps, decay_rate=0.1, lr_min=0, warmup=0):
    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
    else:
        epoch -= warmup
        lr = lr_base * decay_rate ** np.sum(epoch >= np.array(lr_decay_steps))
        lr = lr if lr > lr_min else lr_min
    print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
    return lr

def exp_scheduler(epoch, lr_base=0.1, decay_step=1, decay_rate=0.9, lr_min=0, warmup=0):
    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
    else:
        epoch -= warmup
        lr = lr_base * decay_rate ** (epoch / decay_step)
        lr = lr if lr > lr_min else lr_min
    # print("Learning rate for iter {} is {}".format(epoch + 1, lr))
    return lr

from keras_cv_attention_models.imagenet.callbacks import exp_scheduler, constant_scheduler, CosineLrScheduler
epochs = np.arange(60)
plt.figure(figsize=(14, 6))
plt.plot(epochs, [exp_scheduler(ii, 0.1, 0.9) for ii in epochs], label="lr=0.1, decay=0.9")
plt.plot(epochs, [exp_scheduler(ii, 0.1, 0.7) for ii in epochs], label="lr=0.1, decay=0.7")
plt.plot(epochs, [constant_scheduler(ii, 0.1, [10, 20, 30, 40], 0.1) for ii in epochs], label="Constant, lr=0.1, decay_steps=[10, 20, 30, 40], decay_rate=0.1")

steps_per_epoch = 100
batchs = np.arange(60 * steps_per_epoch)
aa = CosineLrScheduler(0.1, first_restart_step=50, lr_min=1e-6, warmup=0, m_mul=1e-5, steps_per_epoch=steps_per_epoch)
plt.plot(batchs / steps_per_epoch, [aa.on_train_batch_begin(ii) for ii in batchs], label="Cosine, first_restart_step=50, min=1e-6, m_mul=1e-3")

bb = CosineLrScheduler(0.1, first_restart_step=16, lr_min=1e-7, warmup=1, m_mul=0.4, steps_per_epoch=steps_per_epoch)
plt.plot(batchs / steps_per_epoch, [bb.on_train_batch_begin(ii) for ii in batchs], label="Cosine restart, first_restart_step=16, min=1e-7, warmup=1, m_mul=0.4")

plt.xlim(0, 60)
plt.legend()
# plt.grid()
plt.tight_layout()
```
```py
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
xx = np.arange(0, 350)
ax.plot(xx, [exp_scheduler(ii, lr_base=0.256, decay_step=2.4, decay_rate=0.97) for ii in xx])
xx = np.arange(0, 52)
ax.twiny().plot(xx, [exp_scheduler(ii, lr_base=0.256, decay_step=1, decay_rate=0.915, warmup=2) for ii in xx], color='r')
```
```py
from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models import model_surgery

with tf.distribute.MirroredStrategy().scope():
    keras.mixed_precision.set_global_policy("mixed_float16")
    input_shape = (224, 224, 3)
    batch_size = 64
    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(batch_size=batch_size, input_shape=input_shape)
    model = keras.applications.ResNet50V2(input_shape=input_shape, weights=None)
    model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=5e-4, apply_to_batch_normal=False)

    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    # callbacks = myCallbacks.basic_callbacks(checkpoint="keras_checkpoints.h5", lr=0.1, lr_decay=0.1, lr_min=1e-5, lr_decay_steps=[20, 30, 40])
    lr_schduler = CosineLrScheduler(0.1, first_restart_step=16, m_mul=0.5, t_mul=2.0, lr_min=1e-05, warmup=2, steps_per_epoch=steps_per_epoch)
    callbacks = [lr_schduler, keras.callbacks.ModelCheckpoint(model.name + ".h5", monitor='val_loss', save_best_only=True)]

    model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(), metrics=['acc'])
    model.fit(
        train_dataset,
        epochs=50,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=0,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        use_multiprocessing=True,
        workers=4
    )
```
```py
hhs = {
    # "resnet50v2_imagenet_batch_size_128_magnitude_5": "checkpoints/saved/resnet50v2_imagenet_batch_size_128_magnitude_5_hist.json",
    # "coatnet0_imagenet_batch_size_64_magnitude_5": "checkpoints/saved/coatnet0_imagenet_batch_size_64_magnitude_5_hist.json",
    # "aotnet50_swish_preact_avg_down_drop02_mixup_0_imagenet_batch_size_512_magnitude_5": "checkpoints/saved/aotnet50_swish_preact_avg_down_drop02_mixup_0_imagenet_batch_size_512_magnitude_5_hist.json",
    # "coatnet0_imagenet2012_batch_size_32_randaug_10_mixup_0_lr0.01_wd0.1": "checkpoints/coatnet0_imagenet2012_batch_size_32_randaug_10_mixup_0_lr0.01_wd0.1_hist.json",
    # "cmt_tiny_imagenet2012_batch_size_128_randaug_10_mixup_0_lr0.01_wd0.1": "checkpoints/cmt_tiny_imagenet2012_batch_size_128_randaug_10_mixup_0_lr0.01_wd0.1_hist.json",
    # "aotnet50_swish_preact_avg_down_drop02_imagenet2012_batch_size_128_randaug_10_mixup_0_SGD_lr0.1_wd0.0005": "checkpoints/aotnet50_swish_preact_avg_down_drop02_imagenet2012_batch_size_128_randaug_10_mixup_0_SGD_lr0.1_wd0.0005_hist.json",
    # "aotnet50_swish_preact_avg_down_drop02_imagenet2012_batch_size_128_randaug_10_mixup_0_AdamW_lr0.01_wd0.1": "checkpoints/aotnet50_swish_preact_avg_down_drop02_imagenet2012_batch_size_128_randaug_10_mixup_0_AdamW_lr0.01_wd0.1_hist.json",
    "aotnet50, E20, randaug_10_mixup_0_SGD_lr0.1_wd0.0005": "checkpoints/aotnet50_swish_preact_avg_down_drop02_E20_imagenet2012_batch_size_128_randaug_10_mixup_0_SGD_lr0.1_wd0.0005_hist.json",
    "aotnet50, E20, randaug_10_mixup_0_AdamW_lr0.001_wd0.": "checkpoints/aotnet50_swish_preact_avg_down_drop02_E20_imagenet2012_batch_size_128_randaug_10_mixup_0_AdamW_lr0.001_wd0.1_hist.json",
    "aotnet50, E20, randaug_5_mixup_0_AdamW_lr0.001_wd0.1_on_epoch": "checkpoints/aotnet50_swish_preact_avg_down_drop02_E20_imagenet2012_batch_size_128_randaug_5_mixup_0_AdamW_lr0.001_wd0.1_on_epoch_hist.json",
    "aotnet50, E20, randaug_5_mixup_0_SGD_lr0.1_wd0.0005": "checkpoints/aotnet50_swish_preact_avg_down_drop02_E20_imagenet2012_batch_size_128_randaug_5_mixup_0_SGD_lr0.1_wd0.0005_hist.json",
}

fig = imagenet.plot_hists(hhs.values(), list(hhs.keys()), addition_plots=None)

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
***
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
