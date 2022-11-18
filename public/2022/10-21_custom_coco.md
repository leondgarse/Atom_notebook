# ___2022 - 10 - 21 Custom COCO___
***

下载:
https://packages.ubuntu.com/focal/amd64/libmbedcrypto3/download
https://packages.ubuntu.com/bionic/amd64/libmbedcrypto1/download
https://packages.ubuntu.com/focal/amd64/libmbedx509-0/download
https://packages.ubuntu.com/focal/amd64/libmbedtls12/download
https://packages.ubuntu.com/focal/amd64/libgit2-28/download

http://de.archive.ubuntu.com/ubuntu/pool/universe/libg/libgit2/libgit2-28_0.28.4+dfsg.1-2_amd64.deb

安装:
sudo dpkg -i libmbedcrypto3_2.16.4-1ubuntu2_amd64.deb
sudo dpkg -i libmbedcrypto1_2.8.0-1_amd64.deb  # 这个可能不需要
sudo dpkg -i libmbedx509-0_2.16.4-1ubuntu2_amd64.deb
sudo dpkg -i libmbedtls12_2.16.4-1ubuntu2_amd64.deb
sudo dpkg -i libgit2-28_0.28.4+dfsg.1-2_amd64.deb
sudo dpkg -i git-mark_0.7.6_amd64_focal.deb

## coco tiny
  ```py
  !mkdir -p coco_tiny/train2017
  !mkdir -p coco_tiny/val2017

  COCO_PATH = "../coco"

  import json

  with open(os.path.join(COCO_PATH, 'annotations/instances_train2017.json'), 'r') as ff:
      aa = json.load(ff)

  image_info_dict = {ii['id']: ii for ii in aa['images']}
  rr = {}
  target_ids = [2]
  for ii in aa['annotations']:
      if ii['category_id'] in target_ids:
          image_info = image_info_dict[ii['image_id']]
          bbox = ii['bbox']
          left = bbox[0] / image_info["width"]
          top = bbox[1] / image_info["height"]
          right = (bbox[0] + bbox[2]) / image_info["width"]
          bottom = (bbox[1] + bbox[3]) / image_info["height"]
          rr.setdefault(image_info["file_name"], []).append([top, left, bottom, right])
  print(len(rr))

  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(list(rr.keys()), list(rr.values()), test_size=0.1, random_state=42)

  train_image_path = os.path.abspath("./coco_tiny/train2017")
  train = [{"image": os.path.join(train_image_path, xx), "objects": {"label": [0] * len(yy), "bbox": yy}} for xx, yy in zip(x_train, y_train)]
  val_image_path = os.path.abspath("./coco_tiny/val2017")
  test = [{"image": os.path.join(val_image_path, xx), "objects": {"label": [0] * len(yy), "bbox": yy}} for xx, yy in zip(x_test, y_test)]

  with open("coco_tiny/coco_tiny.json", "w") as ff:
      json.dump({"train": train, "test": test, "info": {"num_classes": len(target_ids)}}, ff)

  import shutil

  train_image_path = os.path.abspath(os.path.join(COCO_PATH, 'images/train2017/'))
  for ii in x_train:
      shutil.copy2(os.path.join(train_image_path, ii), './coco_tiny/train2017/')
  for ii in x_test:
      shutil.copy2(os.path.join(train_image_path, ii), './coco_tiny/val2017/')
  ```
## coco tiny for using custom_dataset_script.py
  ```py
  !mkdir -p coco_tiny/train2017/images
  !mkdir -p coco_tiny/train2017/labels
  !mkdir -p coco_tiny/val2017/images
  !mkdir -p coco_tiny/val2017/labels

  COCO_PATH = "coco"

  import json

  with open(os.path.join(COCO_PATH, 'annotations/instances_train2017.json'), 'r') as ff:
      aa = json.load(ff)

  image_info_dict = {ii['id']: ii for ii in aa['images']}
  rr = {}
  target_ids = [2]
  for ii in aa['annotations']:
      if ii['iscrowd'] != 0:
          continue
      if ii['category_id'] in target_ids:
          image_info = image_info_dict[ii['image_id']]
          bbox = ii['bbox']
          left = bbox[0] / image_info["width"]
          top = bbox[1] / image_info["height"]
          right = (bbox[0] + bbox[2]) / image_info["width"]
          bottom = (bbox[1] + bbox[3]) / image_info["height"]

          label = target_ids.index(ii['category_id'])
          rr.setdefault(image_info["file_name"], []).append([label, top, left, bottom, right])
  print(len(rr))

  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(list(rr.keys()), list(rr.values()), test_size=0.1, random_state=42)

  import shutil

  source_image_path = os.path.abspath(os.path.join(COCO_PATH, 'images/train2017'))
  for xx, yy in zip(x_train, y_train):
      shutil.copy2(os.path.join(source_image_path, xx), './coco_tiny/train2017/images')
      label_file = os.path.splitext(xx)[0] + ".txt"
      with open(os.path.join("coco_tiny/train2017/labels", label_file), 'w') as ff:
          ff.write('\n'.join([' '.join([str(jj) for jj in ii]) for ii in yy]))

  for xx, yy in zip(x_test, y_test):
      shutil.copy2(os.path.join(source_image_path, xx), './coco_tiny/val2017/images')
      label_file = os.path.splitext(xx)[0] + ".txt"
      with open(os.path.join("coco_tiny/val2017/labels", label_file), 'w') as ff:
          ff.write('\n'.join([' '.join([str(jj) for jj in ii]) for ii in yy]))
  ```
  ```sh
  python3 custom_dataset_script.py --train_images coco_tiny/train2017/images/ --train_labels coco_tiny/train2017/labels/ \
  --test_images coco_tiny/val2017/images/ --test_labels coco_tiny/val2017/labels/ -s coco_tiny/coco_tiny.json
  ```
## COCO eval annotations format
  ```py
  aa = {
      'images': [{'id': 'foo'}],
      'annotations': [
          {"bbox": [1, 2, 3, 4], "category_id": 12, "image_id": "foo", "id": 0},
          {"bbox": [1, 2, 3, 4], "category_id": 12, "image_id": "foo", "id": 'goo'}
      ]
  }
  with open('foo.json', 'w') as ff:
      json.dump(aa, ff)

  from pycocotools.coco import COCO
  bb = COCO('foo.json')
  bb.anns
  ```
## From custom_dataset_script.py generated json to annotations
  ```py
  aa = {...}

  rr = []
  for ii in aa['train']:
      for bb, label in zip(ii['objects']['bbox'], ii['objects']['label']):
          rr.append({'bbox': bb, "category_id": label, "image_id": ii["image"], "id": len(rr)})

  dd = {}
  dd['images'] = [{'id': ii['image']} for ii in aa['train']]
  dd['annotations'] = rr

  with open('foo.json', 'w') as ff:
      json.dump(dd, ff)
  bb = COCO('foo.json')
  bb.anns
  ```
***

# COCO decode
  ```py
  from keras_cv_attention_models import efficientdet, test_images
  model = efficientdet.EfficientDetD0()
  preds = model(model.preprocess_input(test_images.dog()))

  # Decode and NMS
  from keras_cv_attention_models import coco
  input_shape = model.input_shape[1:-1]
  anchors = coco.get_anchors(input_shape=input_shape, pyramid_levels=[3, 7], anchor_scale=4)
  dd = coco.decode_bboxes(preds[0], anchors).numpy()
  rr = tf.image.non_max_suppression(dd[:, :4], dd[:, 4:].max(-1), score_threshold=0.3, max_output_size=15, iou_threshold=0.5)
  dd_nms = tf.gather(dd, rr).numpy()
  bboxes, labels, scores = dd_nms[:, :4], dd_nms[:, 4:].argmax(-1), dd_nms[:, 4:].max(-1)
  print(f"{bboxes = }, {labels = }, {scores = }")
  # bboxes = array([[0.433231  , 0.54432285, 0.8778939 , 0.8187578 ]], dtype=float32), labels = array([17]), scores = array([0.85373735], dtype=float32)
  ```
  ```py
  +        preds = preds if len(preds.shape) == 3 else preds[None]
  +        output_type = (tf.float32, tf.int64, tf.float32)
  +        __func__ = lambda xx: self.__decode_single__(pred, score_threshold, iou_or_sigma, max_output_size, method, mode, topk, input_shape)
  +        return tf.map_fn(self.__decode_single__, preds, fn_output_signature=output_type)
  +        # return [self.__decode_single__(pred, score_threshold, iou_or_sigma, max_output_size, method, mode, topk, input_shape) for pred in preds]

  tf.tensor_scatter_nd_update(tf.zeros([3, 94]), tf.range(2)[:, None], tf.ones([2, 94]))

  class Decoder(keras.layers.Layer):
      def __init__(self, model_input_shape, pyramid_levels=[3, 7], anchor_scale=4):
          super().__init__()
          self.anchors = coco.get_anchors(input_shape=model_input_shape, pyramid_levels=[3, 7], anchor_scale=4)
          self.model_input_shape, self.pyramid_levels, self.anchor_scale = model_input_shape, pyramid_levels, anchor_scale

      def __decode_single__(self, pred):
          dd = coco.decode_bboxes(pred, self.anchors)
          bboxes, labels = tf.split(dd, [4, -1], axis=-1)
          rr = tf.image.non_max_suppression(bboxes, tf.reduce_max(labels, -1), score_threshold=0.3, max_output_size=15, iou_threshold=0.5)
          dd_nms = tf.gather(dd, rr)
          return tf.tensor_scatter_nd_update(tf.zeros([100, dd.shape[-1]]), tf.range(tf.shape(rr)[0])[:, None], dd_nms)
          # return dd_nms

      def call(self, preds):
          return tf.map_fn(self.__decode_single__, preds)
          # return dd_nms
          # bboxes, labels = tf.split(dd_nms, [4, -1], axis=-1)

      def get_config(self):
          config = super().get_config()
          config.update({"model_input_shape": self.model_input_shape, "pyramid_levels": self.pyramid_levels, "anchor_scale": self.anchor_scale})
          return config
  ```
