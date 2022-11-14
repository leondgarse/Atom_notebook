# ___YOLOX___

***
# YOLOX
## PyTorch YOLOX
  ```py
  import torch
  from exps.default import yolox_s
  yolo_model = yolox_s.Exp()
  torch_model = yolo_model.get_model()
  _ = torch_model.eval()
  weight = torch.load('../keras_cv_attention_models/yolox_s.pth', map_location=torch.device('cpu'))["model"]
  torch_model.load_state_dict(weight)
  torch_model.head.decode_in_inference = False
  torch_pred = torch_model(torch.ones([1, 3, 640, 640]))

  """
  yolox_x, depth = 1.33, width = 1.25
  yolox_l, depth = 1.0, width = 1.0
  yolox_m, depth = 0.67, width = 0.75
  yolox_s, depth = 0.33, width = 0.50
  yolox_tiny, depth = 0.33, width = 0.375
      self.input_scale = (416, 416)
      self.mosaic_scale = (0.5, 1.5)
      self.random_size = (10, 20)
      self.test_size = (416, 416)
      self.enable_mixup = False
  yolox_nano, depth = 0.33, width = 0.25
      self.input_size = (416, 416)
      self.random_size = (10, 20)
      self.mosaic_scale = (0.5, 1.5)
      self.test_size = (416, 416)
      self.mosaic_prob = 0.5
      self.enable_mixup = False
      eps = 1e-3
      momentum = 0.03
      depthwise = True
  """
  ```
## Reload torch yolox weights
  ```py
  # conv1 -> deep, conv2 -> short, conv3 -> output
  tail_split_position = [1, 2] # 1 for backbone, 2 for pafpn
  tail_align_dict = [
    {"short_conv": "block1_1_conv", "short_bn": "block1_1_bn"},
    {"short_conv": "block1_1_conv", "short_bn": "block1_1_bn"},
  ]

  # nano
  full_name_align_dict_nano = [
    {
      "stack1_output_conv": "stack1_block1_2_dw_conv", "stack1_output_bn": "stack1_block1_2_dw_bn",
      "stack2_output_conv": "stack2_block1_2_dw_conv", "stack2_output_bn": "stack2_block1_2_dw_bn",
      "stack3_output_conv": "stack3_block1_2_dw_conv", "stack3_output_bn": "stack3_block1_2_dw_bn",
      "stack4_output_conv": "stack4_block1_2_dw_conv", "stack4_output_bn": "stack4_block1_2_dw_bn",
    },
    {
      "pafpn_c3p4_output_conv": "pafpn_c3p4_block1_2_dw_conv", "pafpn_c3p4_output_bn": "pafpn_c3p4_block1_2_dw_bn",
      "pafpn_c3p3_output_conv": "pafpn_c3p3_block1_2_dw_conv", "pafpn_c3p3_output_bn": "pafpn_c3p3_block1_2_dw_bn",
      "pafpn_c3n3_output_conv": "pafpn_c3n3_block1_2_dw_conv", "pafpn_c3n3_output_bn": "pafpn_c3n3_block1_2_dw_bn",
      "pafpn_c3n4_output_conv": "pafpn_c3n4_block1_2_dw_conv", "pafpn_c3n4_output_bn": "pafpn_c3n4_block1_2_dw_bn",
    },
  ]

  # tiny, s
  full_name_align_dict_s = [
    {
      "stack1_output_conv": "stack1_block1_2_conv", "stack1_output_bn": "stack1_short_conv",
      "stack2_output_conv": "stack2_block1_2_conv", "stack2_output_bn": "stack2_block1_2_bn",
      "stack3_output_conv": "stack3_block1_2_conv", "stack3_output_bn": "stack3_block1_2_bn",
      "stack4_output_conv": "stack4_short_conv", "stack4_output_bn": "stack4_block1_2_conv",
    },
    {
      "pafpn_c3p4_output_conv": "pafpn_c3p4_short_conv", "pafpn_c3p4_output_bn": "pafpn_c3p4_block1_2_conv",
      "pafpn_c3p3_output_conv": "pafpn_c3p3_short_conv", "pafpn_c3p3_output_bn": "pafpn_c3p3_block1_2_conv",
      "pafpn_c3n3_output_conv": "pafpn_c3n3_short_conv", "pafpn_c3n3_output_bn": "pafpn_c3n3_block1_2_conv",
      "pafpn_c3n4_output_conv": "pafpn_c3n4_short_conv", "pafpn_c3n4_output_bn": "pafpn_c3n4_block1_2_conv",
    },
  ]

  # m, l, x
  full_name_align_dict_m = [
    {
      "stack1_output_conv": "stack1_block1_2_conv", "stack1_output_bn": "stack1_block1_2_bn",
      "stack2_output_conv": "stack2_block1_2_conv", "stack2_output_bn": "stack2_block1_2_bn",
      "stack3_output_conv": "stack3_block1_2_conv", "stack3_output_bn": "stack3_block1_2_bn",
      "stack4_output_conv": "stack4_block1_2_conv", "stack4_output_bn": "stack4_block1_2_bn",
    },
    {
      "pafpn_c3p4_output_conv": "pafpn_c3p4_block1_2_conv", "pafpn_c3p4_output_bn": "pafpn_c3p4_block1_2_bn",
      "pafpn_c3p3_output_conv": "pafpn_c3p3_block1_2_conv", "pafpn_c3p3_output_bn": "pafpn_c3p3_block1_2_bn",
      "pafpn_c3n3_output_conv": "pafpn_c3n3_block1_2_conv", "pafpn_c3n3_output_bn": "pafpn_c3n3_block1_2_bn",
      "pafpn_c3n4_output_conv": "pafpn_c3n4_block1_2_conv", "pafpn_c3n4_output_bn": "pafpn_c3n4_block1_2_bn",
    },
  ]

  # tiny, s, m, l, x
  headers = [
    'head_1_cls_1_conv', 'head_1_cls_1_bn', 'head_1_cls_2_conv', 'head_1_cls_2_bn',
    'head_2_cls_1_conv', 'head_2_cls_1_bn', 'head_2_cls_2_conv', 'head_2_cls_2_bn',
    'head_3_cls_1_conv', 'head_3_cls_1_bn', 'head_3_cls_2_conv', 'head_3_cls_2_bn',
    'head_1_reg_1_conv', 'head_1_reg_1_bn', 'head_1_reg_2_conv', 'head_1_reg_2_bn',
    'head_2_reg_1_conv', 'head_2_reg_1_bn', 'head_2_reg_2_conv', 'head_2_reg_2_bn',
    'head_3_reg_1_conv', 'head_3_reg_1_bn', 'head_3_reg_2_conv', 'head_3_reg_2_bn',
    'head_1_class_out', 'head_2_class_out', 'head_3_class_out',
    'head_1_regression_out', 'head_2_regression_out', 'head_3_regression_out',
    'head_1_object_out', 'head_2_object_out', 'head_3_object_out',
    'head_1_stem_conv', 'head_1_stem_bn', 'head_2_stem_conv', 'head_2_stem_bn', 'head_3_stem_conv', 'head_3_stem_bn',
  ]

  headers_nano = [
    'head_1_cls_1_dw_conv', 'head_1_cls_1_dw_bn', 'head_1_cls_1_conv', 'head_1_cls_1_bn',
    'head_1_cls_2_dw_conv', 'head_1_cls_2_dw_bn', 'head_1_cls_2_conv', 'head_1_cls_2_bn',
    'head_2_cls_1_dw_conv', 'head_2_cls_1_dw_bn', 'head_2_cls_1_conv', 'head_2_cls_1_bn',
    'head_2_cls_2_dw_conv', 'head_2_cls_2_dw_bn', 'head_2_cls_2_conv', 'head_2_cls_2_bn',
    'head_3_cls_1_dw_conv', 'head_3_cls_1_dw_bn', 'head_3_cls_1_conv', 'head_3_cls_1_bn',
    'head_3_cls_2_dw_conv', 'head_3_cls_2_dw_bn', 'head_3_cls_2_conv', 'head_3_cls_2_bn',
    'head_1_reg_1_dw_conv', 'head_1_reg_1_dw_bn', 'head_1_reg_1_conv', 'head_1_reg_1_bn',
    'head_1_reg_2_dw_conv', 'head_1_reg_2_dw_bn', 'head_1_reg_2_conv', 'head_1_reg_2_bn',
    'head_2_reg_1_dw_conv', 'head_2_reg_1_dw_bn', 'head_2_reg_1_conv', 'head_2_reg_1_bn',
    'head_2_reg_2_dw_conv', 'head_2_reg_2_dw_bn', 'head_2_reg_2_conv', 'head_2_reg_2_bn',
    'head_3_reg_1_dw_conv', 'head_3_reg_1_dw_bn', 'head_3_reg_1_conv', 'head_3_reg_1_bn',
    'head_3_reg_2_dw_conv', 'head_3_reg_2_dw_bn', 'head_3_reg_2_conv', 'head_3_reg_2_bn',
    'head_1_class_out', 'head_2_class_out', 'head_3_class_out',
    'head_1_regression_out', 'head_2_regression_out', 'head_3_regression_out',
    'head_1_object_out', 'head_2_object_out', 'head_3_object_out',
    'head_1_stem_conv', 'head_1_stem_bn', 'head_2_stem_conv', 'head_2_stem_bn', 'head_3_stem_conv', 'head_3_stem_bn',
  ]

  specific_match_func = lambda tt: tt[:- len(headers)] + headers
  specific_match_func_nano = lambda tt: tt[:- len(headers_nano)] + headers_nano

  from keras_cv_attention_models import download_and_load
  from keras_cv_attention_models.yolox import yolox
  mm = yolox.YOLOXS()

  download_and_load.keras_reload_from_torch_model(
      torch_model=mm.name + ".pth",
      keras_model=mm,
      input_shape=mm.input_shape[1:-1],
      tail_align_dict=tail_align_dict,
      full_name_align_dict=full_name_align_dict_s,
      tail_split_position=tail_split_position,
      specific_match_func=specific_match_func,
      save_name=mm.name + "_coco.h5",
      do_convert=True,
  )
  ```
  **Convert bboxes output `[left, top, right, bottom]` -> `top, left, bottom, right`**
  ```py
  from keras_cv_attention_models.yolox import yolox
  mm = yolox.YOLOXS(pretrained="coco")
  for ii in range(1, 4):
      ss = mm.get_layer('head_{}_regression_out'.format(ii))
      ss.set_weights([ss.get_weights()[0][:, :, :, [1, 0, 3, 2]], ss.get_weights()[1][1, 0, 3, 2]])

  from keras_cv_attention_models import test_images, coco
  imm = test_images.dog_cat()
  preds = mm(mm.preprocess_input(imm))
  bboxs, lables, confidences = mm.decode_predictions(preds)[0]
  coco.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)

  mm.save(mm.name + "_coco.h5")
  ```
## YOLOX post process
  ```sh
  ./tools/demo.py image --name yolox-s --ckpt ../keras_cv_attention_models/yolox_s.pth --path ../keras_cv_attention_models/aa.jpg  --save_result
  ```
  ```py
  import torch
  import torchvision

  from keras_cv_attention_models import test_images
  # img = tf.expand_dims(tf.image.resize(keras.applications.imagenet_utils.preprocess_input(test_images.dog(), mode='torch'), [640, 640]), 0)
  img = test_images.dog()
  input_image = tf.expand_dims(tf.image.resize(img, [640, 640]), 0)
  input_image = torch.from_numpy(input_image.numpy()).permute([0, 3, 1, 2])

  sys.path.append('../YOLOX/')
  from exps.default import yolox_s as yolox
  yolo_model = yolox.Exp()
  torch_model = yolo_model.get_model()
  _ = torch_model.eval()
  weight = torch.load('yolox_s.pth', map_location=torch.device('cpu'))["model"]
  torch_model.load_state_dict(weight)
  torch_model.head.decode_in_inference = False
  torch_pred = torch_model(input_image)
  hw = torch_model.head.hw

  # hw = [x.shape[-2:] for x in torch_pred]
  # [batch, n_anchors_all, 85]
  # outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)

  def decode_outputs(hw, outputs, strides=[8, 16, 32], dtype="float32"):
      grids = []
      new_strides = []
      for (hsize, wsize), stride in zip(hw, strides):
          yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
          grid = torch.stack((xv, yv), 2).view(1, -1, 2)
          grids.append(grid)
          shape = grid.shape[:2]
          new_strides.append(torch.full((*shape, 1), stride))

      grids = torch.cat(grids, dim=1).type(dtype)
      new_strides = torch.cat(new_strides, dim=1).type(dtype)

      outputs[:, :, :2] = (outputs[:, :, :2] + grids) * new_strides
      outputs[:, :, 2:4] = torch.exp(outputs[:, :, 2:4]) * new_strides
      return outputs

  aa = decode_outputs(torch_model.head.hw, torch_pred, dtype=torch_pred.type())

  """ postprocess, https://github.com/Megvii-BaseDetection/YOLOX/tree/master/yolox/utils/boxes.py#L32 """
  box_corner = aa.new(aa.shape)
  box_corner[:, :, 0] = aa[:, :, 0] - aa[:, :, 2] / 2
  box_corner[:, :, 1] = aa[:, :, 1] - aa[:, :, 3] / 2
  box_corner[:, :, 2] = aa[:, :, 0] + aa[:, :, 2] / 2
  box_corner[:, :, 3] = aa[:, :, 1] + aa[:, :, 3] / 2
  aa[:, :, :4] = box_corner[:, :, :4]

  num_classes, conf_thre, nms_thre, class_agnostic = 80, 0.3, 0.3, True
  bb = [None for _ in range(len(aa))]
  for i, image_pred in enumerate(aa):
      # If none are remaining => process next image
      if not image_pred.size(0):
          continue
      # Get score and class with highest confidence
      class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

      conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
      # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
      detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
      detections = detections[conf_mask]
      if not detections.size(0):
          continue

      if class_agnostic:
          nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
      else:
          nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)

      detections = detections[nms_out_index]
      bb[i] = detections if bb[i] is None else torch.cat((bb[i], detections))

  from keras_cv_attention_models.coco import data
  cc = bb[0].detach().numpy()
  data.show_image_with_bboxes(img, cc[:, [1, 0, 3, 2]] / 640, cc[:, -1], cc[:, 5])
  ```
## YOLOX TF post process
  ```py
  from keras_cv_attention_models import yolox
  mm = yolox.YOLOXS()

  from keras_cv_attention_models import test_images, coco
  img = tf.expand_dims(tf.image.resize(test_images.dog(), [640, 640]), 0)
  preds = mm(img)
  # print([ii.shape.as_list() for ii in aa])
  # [[1, 80, 80, 85], [1, 40, 40, 85], [1, 20, 20, 85]]
  # preds = tf.concat([tf.reshape(ii, [-1, ii.shape[1] * ii.shape[2], ii.shape[3]]) for ii in aa], axis=1).numpy()

  YOLO_ANCHORS_PARAM = {"pyramid_levels": [3, 5], "aspect_ratios": [1], "num_scales" :1, "anchor_scale": 1, "grid_zero_start": True}
  anchors = coco.get_anchors([640, 640], **YOLO_ANCHORS_PARAM)

  """ coco.decode_bboxes """
  bboxes, label = preds[0][:, :4], preds[0][:, 4:]
  anchors_wh = anchors[:, 2:] - anchors[:, :2]
  anchors_center = (anchors[:, :2] + anchors[:, 2:]) * 0.5

  bboxes_center = bboxes[:, :2] * anchors_wh + anchors_center
  bboxes_wh = tf.math.exp(bboxes[:, 2:]) * anchors_wh

  dd = coco.decode_bboxes(preds[0], anchors)

  bbs, ccs, labels = dd[:, :4], tf.reduce_max(dd[:, 4:-1], axis=-1) * dd[:, -1], tf.argmax(dd[:, 4:-1], -1)
  rr, nms_scores = tf.image.non_max_suppression_with_scores(bbs, ccs, 100, 0.3, 0.3, 0.0)
  bbs, labels, ccs = tf.gather(bbs, rr).numpy(), tf.gather(labels, rr).numpy(), nms_scores.numpy()
  coco.show_image_with_bboxes(test_images.dog(), bbs, labels, ccs)
  ```
  ```py
  from keras_cv_attention_models import test_images, coco
  imm = test_images.dog_cat()
  preds = mm(mm.preprocess_input(imm))
  bboxs, lables, confidences = mm.decode_predictions(preds)[0]
  coco.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
## YOLOXTiny eval
  | nms_score_threshold | nms_iou_or_sigma | clip_bbox | nms_method | nms_mode      | nms_topk | others          | Val AP |
  | ------------------- | ---------------- | --------- | ---------- | ------------- | -------- | --------------- | ------ |
  | 0.001               | 0.5              | True      | gaussian   | per_class     | 5000     |                 | 0.289  |
  | 0.001               | 0.65             | True      | hard       | torch_batched | 5000     |                 | 0.291  |
  | 0.001               | 0.65             | True      | hard       | per_class     | 5000     |                 | 0.291  |
  | 0.001               | 0.65             | True      | hard       | torch_batched | 5000     | antialias False | 0.290  |
  | 0.001               | 0.65             | True      | hard       | torch_batched | 5000     | input 640       | 0.305  |
  | 0.001               | 0.65             | True      | hard       | torch_batched | 0        | input 640       | 0.303  |
  | 0.001               | 0.65             | True      | hard       | torch_batched | 5000     | BGR             | 0.329  |

  ```py
  CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m yolox.YOLOXTiny --nms_method hard --nms_iou_or_sigma 0.65 --use_bgr_input --use_anchor_free_mode
   # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
   # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.504
   # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.349
   # Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.138
   # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.360
   # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.499
   # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.287
   # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.458
   # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
   # Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
   # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.549
   # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
  ```
***

# YOLOX training
## Mosaic mix
  ```py
  from keras_cv_attention_models import test_images
  images = tf.stack([test_images.dog(), test_images.cat(), test_images.dog_cat()], axis=0)

  def random_largest_crop_and_resize_images(images, target_shape, method="bilinear", antialias=False):
      height, width = tf.cast(tf.shape(images)[1], "float32"), tf.cast(tf.shape(images)[2], "float32")
      target_height, target_width = tf.cast(target_shape[0], "float32"), tf.cast(target_shape[1], "float32")
      scale = tf.maximum(target_height / height, target_width / width)
      scaled_hh, scaled_ww = tf.cast(int(height * scale), "int32"), tf.cast(int(width * scale), "int32")
      images = tf.image.resize(images, [scaled_hh, scaled_ww], method=method, antialias=antialias)

      print(target_shape)
      # crop_hh = tf.cond(scaled_hh > target_height, lambda: tf.random.uniform((), 0, scaled_hh - target_height, dtype='int32'), lambda: 0)
      # crop_ww = tf.cond(scaled_ww > target_width, lambda: tf.random.uniform((), 0, scaled_ww - target_width, dtype='int32'), lambda: 0)
      crop_hh = tf.random.uniform((), 0, tf.maximum(scaled_hh - target_shape[0], 1), dtype='int32')
      crop_ww = tf.random.uniform((), 0, tf.maximum(scaled_ww - target_shape[1], 1), dtype='int32')
      print(scaled_hh, scaled_ww, target_shape[0], target_shape[1], crop_hh, crop_ww)
      images = images[:, crop_hh : crop_hh + target_shape[0], crop_ww : crop_ww + target_shape[1]]
      return images, scale, crop_hh, crop_ww

  batch_size, hh, ww, _ = images.shape
  split_hh = tf.cast(tf.random.uniform((), 0.25 * hh, 0.75 * hh), "int32")
  split_ww = tf.cast(tf.random.uniform((), 0.25 * ww, 0.75 * ww), "int32")

  # top_left, top_right, bottom_left, bottom_right
  sub_hh_wws = [[split_hh, split_ww], [split_hh, ww - split_ww], [hh - split_hh, split_ww], [hh - split_hh, ww - split_ww]]
  mixed_images, mixed_labels = [], []
  for sub_hh, sub_ww in sub_hh_wws:
      pick_indices = tf.random.shuffle(tf.range(batch_size))
      cur_images = tf.gather(images, pick_indices)
      cur_images, scale, crop_hh, crop_ww = random_largest_crop_and_resize_images(cur_images, [sub_hh, sub_ww])
      mixed_images.append(cur_images)
      print(f"{cur_images.shape = }")

      # cur_labels = tf.gather(labels, pick_indices)
      # cur_labels = resize_and_crop_bboxes(cur_labels, (hh, ww), scale, crop_hh, crop_ww)
      # cur_labels = refine_bboxes_labels(cur_labels, keep_shape=True)
      # mixed_labels.append(cur_labels)

  top = tf.concat([mixed_images[0], mixed_images[1]], axis=2)
  bottom = tf.concat([mixed_images[2], mixed_images[3]], axis=2)
  mix = tf.concat([top, bottom], axis=1)
  print(f"{top.shape = }, {bottom.shape = }, {mix.shape = }")
  ```
## Train
  ```py
   def get_losses(imgs, num_classes, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
       bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4], decoded. origin_preds is encoded preds
       obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
       cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

       # calculate targets
       nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

       total_num_anchors = outputs.shape[1]
       x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
       y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
       expanded_strides = torch.cat(expanded_strides, 1)
       cls_targets = []
       reg_targets = []
       l1_targets = []
       obj_targets = []
       fg_masks = []

       num_fg = 0.0
       num_gts = 0.0

       for batch_idx in range(outputs.shape[0]):
           num_gt = int(nlabel[batch_idx])
           num_gts += num_gt

           gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
           gt_classes = labels[batch_idx, :num_gt, 0]
           bboxes_preds_per_image = bbox_preds[batch_idx]

           gt_matched_classes, is_in_boxes_anchor, pred_ious_this_matching, matched_gt_inds, num_fg_img = get_assignments(
               batch_idx,
               num_classes,
               num_gt,
               total_num_anchors,
               gt_bboxes_per_image,
               gt_classes,
               bboxes_preds_per_image,
               expanded_strides,
               x_shifts,
               y_shifts,
               cls_preds,
               bbox_preds,
               obj_preds,
               labels,
               imgs,
           )

           num_fg += num_fg_img

           cls_target = F.one_hot(gt_matched_classes.to(torch.int64), num_classes) * pred_ious_this_matching.unsqueeze(-1)
           obj_target = is_in_boxes_anchor.unsqueeze(-1)
           reg_target = gt_bboxes_per_image[matched_gt_inds]

           cls_targets.append(cls_target)
           reg_targets.append(reg_target)
           obj_targets.append(obj_target.to(dtype))
           fg_masks.append(is_in_boxes_anchor)

       cls_targets = torch.cat(cls_targets, 0)
       reg_targets = torch.cat(reg_targets, 0)
       obj_targets = torch.cat(obj_targets, 0)
       fg_masks = torch.cat(fg_masks, 0)

       num_fg = max(num_fg, 1)
       return cls_targets, reg_targets, obj_targets, fg_masks, num_fg, bbox_preds.view(-1, 4)[fg_masks], cls_preds.view(-1, num_classes)[fg_masks]

   def get_assignments(
       batch_idx,
       num_classes,
       num_gt,
       total_num_anchors,
       gt_bboxes_per_image,
       gt_classes,
       bboxes_preds_per_image,
       expanded_strides,
       x_shifts,
       y_shifts,
       cls_preds,
       bbox_preds,
       obj_preds,
       labels,
       imgs,
       mode="gpu",
   ):
       # is_in_boxes_anchor, is_in_boxes_and_center
       is_in_boxes_anchor, is_in_boxes_and_center = get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)

       bboxes_preds_per_image = bboxes_preds_per_image[is_in_boxes_anchor]
       cls_preds_ = cls_preds[batch_idx][is_in_boxes_anchor]
       obj_preds_ = obj_preds[batch_idx][is_in_boxes_anchor]
       num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

       pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
       pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

       gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
       cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
       pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)

       cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center)
       num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, is_in_boxes_anchor)

       return gt_matched_classes, is_in_boxes_anchor, pred_ious_this_matching, matched_gt_inds, num_fg

   def get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
       expanded_strides_per_image = expanded_strides[0]
       x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
       y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
       x_centers_per_image = (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)  # [n_anchor] -> [n_gt, n_anchor]
       y_centers_per_image = (y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

       gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
       gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
       gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
       gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)

       b_l = x_centers_per_image - gt_bboxes_per_image_l
       b_r = gt_bboxes_per_image_r - x_centers_per_image
       b_t = y_centers_per_image - gt_bboxes_per_image_t
       b_b = gt_bboxes_per_image_b - y_centers_per_image
       bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

       is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
       is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
       # in fixed center

       center_radius = 2.5

       gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
       gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
       gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
       gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

       c_l = x_centers_per_image - gt_bboxes_per_image_l
       c_r = gt_bboxes_per_image_r - x_centers_per_image
       c_t = y_centers_per_image - gt_bboxes_per_image_t
       c_b = gt_bboxes_per_image_b - y_centers_per_image
       center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
       is_in_centers = center_deltas.min(dim=-1).values > 0.0
       is_in_centers_all = is_in_centers.sum(dim=0) > 0

       # in boxes and in centers
       is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

       is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
       return is_in_boxes_anchor, is_in_boxes_and_center

   def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, is_in_boxes_anchor):
       matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

       n_candidate_k = min(10, pair_wise_ious.size(1))
       topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
       dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
       dynamic_ks = dynamic_ks.tolist()
       for gt_idx in range(num_gt):
           _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
           matching_matrix[gt_idx][pos_idx] = 1

       anchor_matching_gt = matching_matrix.sum(0)
       if (anchor_matching_gt > 1).sum() > 0:
           _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
           matching_matrix[:, anchor_matching_gt > 1] *= 0
           matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
       fg_mask_inboxes = matching_matrix.sum(0) > 0
       num_fg = fg_mask_inboxes.sum().item()

       is_in_boxes_anchor[is_in_boxes_anchor.clone()] = fg_mask_inboxes

       matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
       gt_matched_classes = gt_classes[matched_gt_inds]

       pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
       return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
  ```
  ```py
  # [num_bboxes, num_anchors]
  pair_wise_ious, cost = torch.rand([3, 12]), torch.rand([3, 12])
  matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
  n_candidate_k = min(10, pair_wise_ious.size(1))
  topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1) # [num_bboxes, 10]
  dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # [num_bboxes]
  num_gt = dynamic_ks.size(0)
  dynamic_ks = dynamic_ks.tolist()
  for gt_idx in range(num_gt):
      _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
      matching_matrix[gt_idx][pos_idx] = 1
  ```
  ```py
  from keras_cv_attention_models.coco import anchors_func
  aa = anchors_func.AnchorFreeAssignMatching()

  # Fake data
  batch_size, num_bboxes, num_classes = 4, 32, 10
  bboxes_true = tf.random.uniform([batch_size, num_bboxes, 4])
  labels_true = tf.one_hot(tf.random.uniform([batch_size, num_bboxes], 0, num_classes, dtype=tf.int32), num_classes)
  valid_bboxes_pick = tf.cast(tf.random.uniform([batch_size, num_bboxes, 1]) > 0.5, tf.float32)
  bbox_labels_true = tf.concat([bboxes_true, labels_true, valid_bboxes_pick], axis=-1)
  bbox_labels_pred = tf.random.uniform([batch_size, aa.num_anchors, 4 + num_classes + 1])

  """ torch run """
  import torch
  bbox_labels_true_1 = tf.gather(bbox_labels_true[0], tf.where(bbox_labels_true[0, :, -1] > 0)[:, 0])
  bboxes_true, labels_true = bbox_labels_true_1[:, :4], bbox_labels_true_1[:, 4:-1]

  gt_bboxes_per_image = np.concatenate(anchors_func.corners_to_center_xywh_nd(bboxes_true), axis=-1)[:, [1, 0, 3, 2]]
  gt_bboxes_per_image = torch.from_numpy(gt_bboxes_per_image)

  center_xy, wh = anchors_func.corners_to_center_xywh_nd(aa.anchors)
  expanded_strides = torch.from_numpy(wh[:, 0].numpy().reshape(1, -1))
  x_shifts = torch.from_numpy(center_xy[:, 1].numpy().reshape(1, -1))
  y_shifts = torch.from_numpy(center_xy[:, 0].numpy().reshape(1, -1))

  is_in_boxes_anchor, is_in_boxes_and_center = get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, expanded_strides.shape[1], bboxes_true.shape[0])
  ```
## Anchor free test
  ```py
  from keras_cv_attention_models import yolox, test_images
  from keras_cv_attention_models.coco import anchors_func, data
  mm = yolox.YOLOXS()
  img = test_images.dog_cat()
  pred = mm(mm.preprocess_input(img))

  aa = anchors_func.AnchorFreeAssignMatching()
  bbs, lls, ccs = mm.decode_predictions(pred)[0]
  bbox_labels_true = tf.concat([bbs, tf.one_hot(lls, 80), tf.ones([bbs.shape[0], 1])], axis=-1)
  bboxes_true, labels_true, object_true, bboxes_pred, labels_pred = aa(tf.expand_dims(bbox_labels_true, 0), pred)
  data.show_image_with_bboxes(img, bboxes_pred, labels_pred.numpy().argmax(-1))
  ```
## Loss
  ```py
  class IOUloss(nn.Module):
      def __init__(self, reduction="none", loss_type="iou"):
          super(IOUloss, self).__init__()
          self.reduction = reduction
          self.loss_type = loss_type

      def forward(self, pred, target):
          assert pred.shape[0] == target.shape[0]

          pred = pred.view(-1, 4)
          target = target.view(-1, 4)
          tl = torch.max((pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2))
          br = torch.min((pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2))

          area_p = torch.prod(pred[:, 2:], 1)
          area_g = torch.prod(target[:, 2:], 1)

          en = (tl < br).type(tl.type()).prod(dim=1)
          area_i = torch.prod(br - tl, 1) * en
          area_u = area_p + area_g - area_i
          iou = (area_i) / (area_u + 1e-16) # Only iou type

          return 1 - iou ** 2
  ```
  ```py
  for ...:
      if self.use_l1:
          l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds], expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask])

      if self.use_l1:
          l1_targets.append(l1_target)

  cls_targets = torch.cat(cls_targets, 0)
  reg_targets = torch.cat(reg_targets, 0)
  obj_targets = torch.cat(obj_targets, 0)
  fg_masks = torch.cat(fg_masks, 0)
  if self.use_l1:
      l1_targets = torch.cat(l1_targets, 0)

  num_fg = max(num_fg, 1)
  loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
  loss_obj = self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
  loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg
  if self.use_l1:
      loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
  else:
      loss_l1 = 0.0

  reg_weight = 5.0
  loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

  return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1)

  self.l1_loss = nn.L1Loss(reduction="none")
  self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
  self.iou_loss = IOUloss(reduction="none")

  def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
      l1_target[:, 0] = gt[:, 0] / stride - x_shifts
      l1_target[:, 1] = gt[:, 1] / stride - y_shifts
      l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
      l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
      return l1_target
  ```
  ```py
  output = torch.cat([reg_output, obj_output, cls_output], 1)
  output, grid = self.get_output_and_grid(
      output, k, stride_this_level, xin[0].type()
  )
  x_shifts.append(grid[:, :, 0])
  y_shifts.append(grid[:, :, 1])
  expanded_strides.append(
      torch.zeros(1, grid.shape[1])
      .fill_(stride_this_level)
      .type_as(xin[0])
  )
  if self.use_l1:
      batch_size = reg_output.shape[0]
      hsize, wsize = reg_output.shape[-2:]
      reg_output = reg_output.view(
          batch_size, self.n_anchors, 4, hsize, wsize
      )
      reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
          batch_size, -1, 4
      )
      origin_preds.append(reg_output.clone())

  return self.get_losses(
      imgs,
      x_shifts,
      y_shifts,
      expanded_strides,
      labels,
      torch.cat(outputs, 1),
      origin_preds,
      dtype=xin[0].dtype,
  )
  ```
***

# YOLOR
## PyTorch YOLOR
  ```py
  from models import models as torch_yolor
  torch_yolor.ONNX_EXPORT = False
  mm = torch_yolor.Darknet('cfg/yolor_csp.cfg', [640, 640])
  _ = mm.eval()

  # download_and_load.try_save_pth_and_onnx(mm, [640, 640], True, True)
  import torch
  input_shape = [640, 640]
  output_name = mm.__class__.__name__ + ".onnx"
  torch.onnx.export(
      model=mm,
      args=torch.randn(1, 3, *input_shape),
      f=output_name,
      verbose=False,
      keep_initializers_as_inputs=True,
      training=torch.onnx.TrainingMode.PRESERVE,
      do_constant_folding=True,
      opset_version=13,
  )
  ```
  ```py
  sys.path.append('../yolor-paper/')
  from models import yolo as torch_yolor
  torch_model = torch_yolor.Model(cfg='../yolor-paper/models/yolor-d6.yaml')
  _ = torch_model.eval()

  import kecam
  kecam.download_and_load.try_save_pth_and_onnx(torch_model, [640, 640], False, True)
  ```
  ```py
  import torch
  from models import models as yolor
  img_size = [640, 640]
  weights_file = "yolor_csp_star.pt"
  model = yolor.Darknet('cfg/yolor_csp.cfg', img_size)
  weights = torch.load(weights_file, map_location=torch.device('cpu'))['model']
  model.load_state_dict(weights)
  _ = model.eval()

  img = torch.zeros((1, 3, *img_size))
  output = model(img)  # dry run

  import onnx
  print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
  target_onnx_file = weights_file.replace('.pt', '.onnx')  # filename
  # model.fuse()  # only for ONNX
  torch.onnx.export(model, img, target_onnx_file, verbose=False, opset_version=12, input_names=['images'],
                    output_names=['classes', 'boxes'] if output is None else ['output'])

  # Checks
  onnx_model = onnx.load(target_onnx_file)  # load onnx model
  onnx.checker.check_model(onnx_model)  # check onnx model
  print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
  print('ONNX export success, saved as %s' % target_onnx_file)
  ```
## Reload torch YOLOR weights
  ```py
  from keras_cv_attention_models import download_and_load
  from keras_cv_attention_models.yolor import yolor
  mm = yolor.YOLOR()

  tail_split_position = [1, 2] # 1 for backbone, 2 for pafpn
  tail_align_dict = [
      {"short_conv": 'deep_pre_conv', "short_bn": 'deep_pre_bn'},
      # {"pafpn": {"short_conv": "block1_1_conv", "short_bn": "block1_1_bn"}}
      {"pafpn": {"up_conv": -2, "up_bn": -2, "short_conv": "block1_1_conv", "short_bn": "block1_1_bn"}}
  ]

  full_name_align_dict = {
      "stack4_spp_short_conv": "stack4_spp_pre_1_conv", "stack4_spp_short_bn": "stack4_spp_pre_1_bn", "stack4_spp_output_bn": -1,
      "stack5_spp_short_conv": "stack5_spp_pre_1_conv", "stack5_spp_short_bn": "stack5_spp_pre_1_bn", "stack5_spp_output_bn": -1,
      # "pafpn_p4p5_up_conv": -2, "pafpn_p4p5_up_bn": -2, "pafpn_p4p5_output_bn": -1, "pafpn_p3p4p5_up_conv": -2, "pafpn_p3p4p5_up_bn": -2,
      # "pafpn_p4p5_output_bn": -1,  # For loading self trained model
      "pafpn_p5p6_output_bn": -1, "pafpn_p4p5p6_output_bn": -1
  }

  headers = [
      'head_1_shift_channel', 'head_2_shift_channel', 'head_3_shift_channel',
      'head_1_control_channel', 'head_2_control_channel', 'head_3_control_channel',
      'head_1_1_conv', 'head_1_1_bn', 'head_1_2_conv',
      'head_2_1_conv', 'head_2_1_bn', 'head_2_2_conv',
      'head_3_1_conv', 'head_3_1_bn', 'head_3_2_conv',
  ]
  headers_p6 = [
    'head_1_shift_channel', 'head_2_shift_channel', 'head_3_shift_channel', 'head_4_shift_channel',
    'head_1_control_channel', 'head_2_control_channel', 'head_3_control_channel', 'head_4_control_channel',
    'head_1_1_conv', 'head_1_1_bn', 'head_1_2_conv',
    'head_2_1_conv', 'head_2_1_bn', 'head_2_2_conv',
    'head_3_1_conv', 'head_3_1_bn', 'head_3_2_conv',
    'head_4_1_conv', 'head_4_1_bn', 'head_4_2_conv',
  ]
  specific_match_func = lambda tt: tt[:- len(headers)] + headers
  specific_match_func_p6 = lambda tt: tt[:- len(headers_p6)] + headers_p6

  additional_transfer = {yolor.ChannelAffine: lambda ww: [np.squeeze(ww[0])], yolor.BiasLayer: lambda ww: [np.squeeze(ww[0])]}

  download_and_load.keras_reload_from_torch_model(
      '../yolor/yolor_csp_star.pt',
      mm,
      tail_align_dict=tail_align_dict,
      tail_split_position=tail_split_position,
      full_name_align_dict=full_name_align_dict,
      specific_match_func=specific_match_func,
      additional_transfer=additional_transfer,
      save_name=mm.name + "_coco.h5",
      do_convert=True,
  )
  ```
  **Convert bboxes output `[left, top, right, bottom]` -> `top, left, bottom, right`**
  ```py
  from keras_cv_attention_models.yolor import yolor
  mm = yolor.YOLOR_CSP(pretrained="coco")
  for ii in range(1, 5):
      conv_layer = mm.get_layer('head_{}_2_conv'.format(ii))
      new_ww = []
      for ww in conv_layer.get_weights():
          ww = np.reshape(ww, [*ww.shape[:-1], 3, 85])[..., [1, 0, 3, 2, *np.arange(5, 85), 4]]
          ww = np.reshape(ww, [*ww.shape[:-2], -1])
          new_ww.append(ww)
      conv_layer.set_weights(new_ww)

      channel_layer = mm.get_layer('head_{}_control_channel'.format(ii))
      ww = channel_layer.get_weights()[0]
      ww = np.reshape(ww, [*ww.shape[:-1], 3, 85])[..., [1, 0, 3, 2, *np.arange(5, 85), 4]]
      ww = np.reshape(ww, [*ww.shape[:-2], -1])
      channel_layer.set_weights([ww])

  nn = yolor.YOLOR_CSP(pretrained="coco")
  aa = nn(tf.ones([1, *nn.input_shape[1:]]))
  bb = mm(tf.ones([1, *mm.input_shape[1:]]))
  print(np.allclose(aa, bb.numpy()[:, :, [1, 0, 3, 2, 84, *np.arange(4, 84)]]))
  # True
  mm.save(mm.name + "_coco.h5")


  from keras_cv_attention_models import test_images, coco
  imm = test_images.dog_cat()
  preds = mm(mm.preprocess_input(imm))
  bboxs, lables, confidences = mm.decode_predictions(preds)[0]
  coco.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
## Output verification
  ```py
  from keras_cv_attention_models import yolor, test_images, coco
  # imm = np.ones([1, 640, 640, 3], dtype="float32")
  imm = tf.image.resize(test_images.dog(), [640, 640]).numpy()[None] / 255

  import torch
  sys.path.append('../yolor/')
  from models import models as torch_yolor
  img_size = [640, 640]
  weights_file = "../yolor/yolor_csp_star.pt"
  model = torch_yolor.Darknet('../yolor/cfg/yolor_csp.cfg', img_size)
  weights = torch.load(weights_file, map_location=torch.device('cpu'))['model']
  model.load_state_dict(weights)
  _ = model.eval()
  aa, bb = model.forward_once(torch.from_numpy(imm).permute([0, 3, 1, 2]))

  mm = yolor.YOLOR_CSP(classifier_activation=None)
  cc = mm(imm)
  # dd = tf.reshape(cc[:, :80 * 80 * 3], bb[0].shape)
  dd = tf.reshape(tf.transpose(tf.reshape(cc[:, :np.prod(bb[0].shape[:-1])], [1, -1, 3, 85]), [0, 2, 1, 3]), bb[0].shape)
  dd = tf.gather(dd, [1, 0, 3, 2, 84, *np.arange(4, 84)], axis=-1)
  print(np.allclose(bb[0].detach().numpy(), dd, atol=1e-4))
  # True
  ```
  **Decode**
  ```py
  anchors = coco.anchors_func.get_yolor_anchors()
  dd = tf.sigmoid(cc)

  center_yx = (dd[:, :, :2] * 2 * anchors[:, 4:] + anchors[:, :2]) * 640
  hhww = ((dd[:, :, 2:4] * 2) ** 2 * anchors[:, 2:4]) * 640

  # center_yx = tf.reshape(tf.transpose(tf.reshape(center_yx, [1, -1, 3, 2]), [0, 2, 1, 3]), [1, -1, 2])
  # hhww = tf.reshape(tf.transpose(tf.reshape(hhww, [1, -1, 3, 2]), [0, 2, 1, 3]), [1, -1, 2])
  # print(np.allclose(aa.detach().numpy(), tf.concat([center_yx, hhww, dd[:, :, 4:]], axis=-1), atol=1e-3))
  center_yx = tf.split(center_yx, [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3], axis=1)
  center_yx = tf.concat([tf.reshape(tf.transpose(tf.reshape(ii, [1, -1, 3, 2]), [0, 2, 1, 3]), [1, -1, 2]) for ii in center_yx], axis=1)
  hhww = tf.split(hhww, [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3], axis=1)
  hhww = tf.concat([tf.reshape(tf.transpose(tf.reshape(ii, [1, -1, 3, 2]), [0, 2, 1, 3]), [1, -1, 2]) for ii in hhww], axis=1)
  bboxes = tf.gather(tf.concat([center_yx, hhww], axis=-1), [1, 0, 3, 2], axis=-1)
  print(np.allclose(aa.detach().numpy()[:, :, :4], bboxes, atol=1e-3))
  # True

  labels = tf.split(dd[:, :, 4:], [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3], axis=1)
  labels = tf.concat([tf.reshape(tf.transpose(tf.reshape(ii, [1, -1, 3, 81]), [0, 2, 1, 3]), [1, -1, 81]) for ii in labels], axis=1)
  dd = tf.gather(tf.concat([center_yx, hhww, labels], axis=-1), [1, 0, 3, 2, 84, *np.arange(4, 84)], axis=-1)
  print(np.allclose(aa.detach().numpy(), dd, atol=1e-3))
  # True
  ```
  **Prediction**
  ```py
  # detect.py
  from utils.general import non_max_suppression
  pred = non_max_suppression(aa, conf_thres=0.4, iou_thres=0.5)
  # Dog pred
  # [tensor([[348.77496, 279.56995, 520.57544, 535.72040,   0.80469,  16.00000]]

  nn = kecam.yolox.YOLOXS()
  nn.decode_predictions(nn(nn.preprocess_input(test_images.dog())))[0][0] * 640
  # [     276.95,       349.3,      550.25,      523.59]
  ```
  ```py
  # https://github.com/WongKinYiu/yolor/tree/master/utils/general.py#L311
  yolor output: (center x, center y, width, height)
  ```
  ```py
  def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
      nc = prediction[0].shape[1] - 5  # number of classes
      xc = prediction[..., 4] > conf_thres  # candidates

      # Settings
      min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
      max_det = 300  # maximum number of detections per image
      time_limit = 10.0  # seconds to quit after
      redundant = True  # require redundant detections
      multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

      t = time.time()
      output = [torch.zeros(0, 6)] * prediction.shape[0]
      for xi, x in enumerate(prediction):  # image index, image inference
          # Apply constraints
          # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
          x = x[xc[xi]]  # confidence

          # If none remain process next image
          if not x.shape[0]:
              continue

          # Compute conf
          x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

          # Box (center x, center y, width, height) to (x1, y1, x2, y2)
          box = xywh2xyxy(x[:, :4])

          # Detections matrix nx6 (xyxy, conf, cls)
          if multi_label:
              i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
              x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
          else:  # best class only
              conf, j = x[:, 5:].max(1, keepdim=True)
              x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

          # If none remain process next image
          n = x.shape[0]  # number of boxes
          if not n:
              continue

          # Sort by confidence
          # x = x[x[:, 4].argsort(descending=True)]

          # Batched NMS
          c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
          boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
          i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
          if i.shape[0] > max_det:  # limit detections
              i = i[:max_det]
          output[xi] = x[i]
          if (time.time() - t) > time_limit:
              break  # time limit exceeded

      return output
  ```
  ```py
  anchors = kecam.coco.anchors_func.get_yolor_anchors()
  dd = tf.sigmoid(cc)

  center_yx = (dd[:, :, :2] * 2 * anchors[:, 4:] + anchors[:, :2]) * 640
  hhww = ((dd[:, :, 2:4] * 2) ** 2 * anchors[:, 2:4]) * 640
  print(np.allclose(aa.detach().numpy(), tf.concat([center_yx, hhww, dd[:, :, 4:]], axis=-1), atol=1e-3))

  from keras_cv_attention_models import coco
  input_shape = mm.input_shape[1:-1]
  anchors = coco.get_yolor_anchors(input_shape=input_shape, pyramid_levels=[3, 5])
  ee = coco.decode_bboxes(dd[0], anchors).numpy()
  confs = (tf.reduce_max(dd[0, :, 5:], axis=-1) * dd[0, :, 4]).numpy()
  rr = tf.image.non_max_suppression(ee[:, :4], confs, score_threshold=0.3, max_output_size=15, iou_threshold=0.5)
  dd_nms = tf.gather(ee, rr).numpy()
  bboxes, labels, scores = dd_nms[:, :4], dd_nms[:, 4:].argmax(-1), dd_nms[:, 4:].max(-1)
  print(f"{bboxes = }, {labels = }, {scores = }")
  ```
## Reload YOLOR paper weights
  ```py
  sys.path.append('../yolor-paper/')
  from keras_cv_attention_models import download_and_load
  from keras_cv_attention_models.yolor import yolor
  mm = yolor.YOLOR_E6()

  # downsample: conv_down_1, conv_down_2, max_down
  # stack: deep_pre, short, output, concat_bn
  # ssp: pre_1, short, pre_2, pre_3, post_1, post_2, concat_bn, output
  # pafpn: up, down, pre, short, output, concat_bn
  tail_split_position = [1, 2] # 1 for backbone, 2 for pafpn
  tail_align_dict = [
      {"conv_down_2_bn": -1, "short_conv": 'block1_1_conv', "output_conv": 'block1_1_bn', "output_bn": 'block1_2_conv', "concat_bn": "block1_1_bn"},
      # {"pafpn": {"short_conv": "block1_1_conv", "short_bn": "block1_1_bn"}}
      {"pafpn": {
          "up_conv": -2, "up_bn": -2, "short_conv": "block1_1_conv", "output_conv": "block1_1_bn", "output_bn": "block1_2_conv", "concat_bn": "block1_1_bn",
          "conv_down_2_bn": -1
          }
      }
  ]

  full_name_align_dict = {
      "stack5_spp_short_conv": "stack5_spp_pre_2_conv", "stack5_spp_output_bn": -1,
      'head_1_1_conv': "pafpn_c3n3_conv_down_1_conv", "head_1_1_bn": "pafpn_c3n3_conv_down_1_bn",
      'head_2_1_conv': "pafpn_c3n4_conv_down_1_conv", "head_2_1_bn": "pafpn_c3n4_conv_down_2_conv",
      'head_3_1_conv': "pafpn_c3n5_conv_down_1_conv", "head_3_1_bn": "pafpn_c3n5_conv_down_2_bn",
  }

  headers = [
      'head_4_1_conv', 'head_4_1_bn',
      'head_1_2_conv', 'head_2_2_conv', 'head_3_2_conv', 'head_4_2_conv',
      'head_1_shift_channel', 'head_2_shift_channel', 'head_3_shift_channel', 'head_4_shift_channel',
      'head_1_control_channel', 'head_2_control_channel', 'head_3_control_channel', 'head_4_control_channel',
  ]
  specific_match_func = lambda tt: tt[:- len(headers)] + headers

  additional_transfer = {yolor.ChannelAffine: lambda ww: [np.squeeze(ww[0])], yolor.BiasLayer: lambda ww: [np.squeeze(ww[0])]}
  skip_weights = ["num_batches_tracked", "anchors", "anchor_grid"]
  download_and_load.keras_reload_from_torch_model(
      '../yolor-paper/yolor-e6-paper-564.pt',
      mm,
      tail_align_dict=tail_align_dict,
      tail_split_position=tail_split_position,
      full_name_align_dict=full_name_align_dict,
      specific_match_func=specific_match_func,
      additional_transfer=additional_transfer,
      skip_weights=skip_weights,
      save_name=mm.name + "_coco.h5",
      do_convert=True,
  )
  ```
  **Convert bboxes output `[left, top, right, bottom]` -> `top, left, bottom, right`**
  ```py
  from keras_cv_attention_models.yolor import yolor
  mm = yolor.YOLOR_CSP(pretrained="coco")
  for ii in range(1, 5):
      conv_layer = mm.get_layer('head_{}_2_conv'.format(ii))
      new_ww = []
      for ww in conv_layer.get_weights():
          ww = np.reshape(ww, [*ww.shape[:-1], 3, 85])[..., [1, 0, 3, 2, *np.arange(5, 85), 4]]
          ww = np.reshape(ww, [*ww.shape[:-2], -1])
          new_ww.append(ww)
      conv_layer.set_weights(new_ww)

      channel_layer = mm.get_layer('head_{}_control_channel'.format(ii))
      ww = channel_layer.get_weights()[0]
      ww = np.reshape(ww, [*ww.shape[:-1], 3, 85])[..., [1, 0, 3, 2, *np.arange(5, 85), 4]]
      ww = np.reshape(ww, [*ww.shape[:-2], -1])
      channel_layer.set_weights([ww])

  nn = yolor.YOLOR_CSP(pretrained="coco")
  aa = nn(tf.ones([1, *nn.input_shape[1:]]))
  bb = mm(tf.ones([1, *mm.input_shape[1:]]))
  print(np.allclose(aa, bb.numpy()[:, :, [1, 0, 3, 2, 84, *np.arange(4, 84)]]))
  # True
  mm.save(mm.name + "_coco.h5")

  from keras_cv_attention_models import test_images, coco
  imm = test_images.dog_cat()
  preds = mm(mm.preprocess_input(imm))
  bboxs, lables, confidences = mm.decode_predictions(preds)[0]
  coco.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
  **Fuse batchnorm**
  ```py
  def fuse_2_bn(bn_layer_1, bn_layer_2, inplace=False):
      # BatchNormalization returns: gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta
      batch_std_1 = tf.sqrt(bn_layer_1.moving_variance + bn_layer_1.epsilon)
      batch_std_2 = tf.sqrt(bn_layer_2.moving_variance + bn_layer_2.epsilon)
      new_gamma = bn_layer_1.gamma * bn_layer_2.gamma
      new_moving_mean = bn_layer_1.moving_mean + bn_layer_2.moving_mean * batch_std_1 / bn_layer_1.gamma
      new_moving_variance = (batch_std_1 * batch_std_2) ** 2 - bn_layer_1.epsilon
      new_beta = bn_layer_2.gamma * bn_layer_1.beta / batch_std_2 + bn_layer_2.beta

      if inplace:
          bn_layer_1.set_weights([new_gamma, new_beta, new_moving_mean, new_moving_variance])
          return bn_layer_1
      else:
          rr = keras.layers.BatchNormalization.from_config(bn_layer_1.get_config())
          rr.build(bn_layer_1.input_shape)
          rr.set_weights([new_gamma, new_beta, new_moving_mean, new_moving_variance])
          return rr
  ```
  ```py
  from keras_cv_attention_models.yolor import yolor_e6, yolor
  mm = yolor_e6.YOLOR_E6(pretrained='yolor_e6_coco.h5')
  nn = yolor.YOLOR_E6(pretrained='yolor_e6_coco.h5')

  aa = [ii.name for ii in mm.layers if ii.name.endswith('_concat_bn')]
  bb = [ii.name for ii in nn.layers if ii.name.endswith('_short_bn')]
  for ss, tt in zip(aa, bb):
      print("source:", ss, "target:", tt)
      ss, tt = mm.get_layer(ss), nn.get_layer(tt)
      tt.set_weights([np.split(ii, 2, axis=-1)[-1] for ii in ss.get_weights()])
  ```
## YOLOR_CSP eval
  ```sh
  # PyTorch yolor
  python test.py --data data/coco.yaml --img 640 --batch 4 --conf 0.001 --iou 0.65 --device 1 --cfg cfg/yolor_csp.cfg --weights yolor_csp_star.pt --name yolor_csp_val
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='1' ./coco_eval_script.py -m yolor.YOLOR_CSP --use_yolor_anchors_mode --nms_method hard --nms_iou_or_sigma 0.65 \
  --nms_max_output_size 300 --nms_topk -1 --letterbox_pad 64 --input_shape 704
  ```

  | nms_max_output_size | nms_topk | letterbox_pad | input_shape | Val AP 0.50:0.95, area=all |
  | ------------------- | -------- | ------------- | ----------- | -------------------------- |
  | 100                 | 5000     | -1            | 640         | 0.488                      |
  | 300                 | 5000     | -1            | 640         | 0.489                      |
  | 300                 | -1       | -1            | 640         | 0.494                      |
  | 300                 | -1       | 0             | 640         | 0.496                      |
  | 300                 | -1       | 0             | 704         | 0.495                      |
  | 300                 | -1       | 64            | 704         | 0.500                      |

  ```py
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.674
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.530
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.324
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.539
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.627
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.592
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.634
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.447
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.684
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779

  # --nms_max_output_size 300
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.489
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.676
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.532
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.540
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.627
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.596
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.647
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.476
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.696
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.783

  # --nms_topk 500000
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.683
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.536
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.335
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.542
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.635
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.376
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.621
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.678
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.505
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.726
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.823

  # --letterbox_pad 0
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.496
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.683
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.539
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.544
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.376
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.623
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.679
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.526
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.729
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.816

  # --letterbox_pad 64 -i 704
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.686
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.544
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.380
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.627
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.683
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.529
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.735
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817
  ```
## YOLOV4 Anchors
  ```py
  # yolor_p6
  19,27,  44,40,  38,94,  96,68,  86,152,  180,137,  140,301,  303,264,  238,542,  436,615,  739,380,  925,792

  # yolov4_p6
  13,17,  31,25,  24,51, 61,45,  48,102,  119,96,  97,189,  217,184,  171,384,  324,451, 545,357, 616,618

  # yolov4_p7
  13,17,  22,25,  27,66, 57,88,  112,69,  69,177,  136,138,  287,114,  134,275,  268,248,  232,504, 445,416,  812,393,  477,808,  1070,908
  ```
***

# YOLOR Training
## PyTorch yolor data augment
  - [load_image](https://github.com/WongKinYiu/yolor/blob/main/utils/datasets.py#L924): largest aspect_aware_resize_and_crop_image
  - [Data augment(https://github.com/WongKinYiu/yolor/blob/main/utils/datasets.py#L546): `load_mosaic` -> `random_perspective` -> `augment_hsv` -> `fliplr`
  ```py
  # vary img-size +/- 50%%
  if opt.multi_scale:
      sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
      sf = sz / max(imgs.shape[2:])  # scale factor
      if sf != 1:
          ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
          imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

  # Training output https://github.com/WongKinYiu/yolor/blob/main/models/models.py#L399
  # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
  p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction
  ```
## YOLOR random perspective
  ```py
  affine = [
    [scale_x, shear_x, translate_x]
    [shear_y, scale_y, translate_y]
    [perspective_x, perspective_y]
  ]
  ```
  ```py
  from keras_cv_attention_models.imagenet.augment import transform, wrap, unwrap
  from keras_cv_attention_models.visualizing import stack_and_plot_images
  from keras_cv_attention_models import test_images

  def show_transformed(image, transforms):
      stack_and_plot_images([transform(image=wrap(image), transforms=tt)[:, :, :3] for tt in transforms])

  image = test_images.cat()
  transforms = tf.convert_to_tensor([
      [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [1.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0],
  ])
  show_transformed(image, transforms)
  ```
  ```py
  def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
      height = img.shape[0] + border[0] * 2  # shape(h,w,c)
      width = img.shape[1] + border[1] * 2

      # Center
      C = np.eye(3)
      C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
      C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

      # Perspective
      P = np.eye(3)
      P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
      P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

      # Rotation and Scale
      R = np.eye(3)
      a = random.uniform(-degrees, degrees)
      # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
      # s = random.uniform(1 - scale, 1 + scale)
      s = scale
      # s = 2 ** random.uniform(-scale, scale)
      R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

      # Shear
      S = np.eye(3)
      S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
      S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

      # Translation
      T = np.eye(3)
      # T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
      # T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
      T[0, 2] = (0.5 - translate) * width  # x translation (pixels)
      T[1, 2] = (0.5 + translate) * height  # y translation (pixels)

      # Combined rotation matrix
      M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
      if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
          if perspective:
              img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
          else:  # affine
              img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
      return img, targets
  ```
## YOLOR random hsv
  ```py
  import cv2
  def augment_hsv(img, hgain=1.0, sgain=1.0, vgain=1.0):
      hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
      dtype = img.dtype  # uint8

      x = np.arange(0, 256, dtype=np.int16)
      lut_hue = ((x * hgain) % 180).astype(dtype)
      lut_sat = np.clip(x * sgain, 0, 255).astype(dtype)
      lut_val = np.clip(x * vgain, 0, 255).astype(dtype)

      img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
      return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

  from keras_cv_attention_models import test_images
  image = test_images.cat()
  hue_delta, saturation_delta, brightness_delta = 0.015, 0.7, 0.4
  aa = tf.concat([augment_hsv(image, hgain=1 - hue_delta), augment_hsv(image, hgain=1 + hue_delta)], axis=1)
  bb = tf.concat([augment_hsv(image, sgain=1 - saturation_delta), augment_hsv(image, sgain=1 + saturation_delta)], axis=1)
  cc = tf.concat([augment_hsv(image, vgain=1 - brightness_delta), augment_hsv(image, vgain=1 + brightness_delta)], axis=1)
  plt.imshow(tf.concat([aa, bb, cc], axis=0))
  plt.axis('off')
  plt.tight_layout()
  ```
  **TF**
  ```py

  ```
## YOLOR assign anchors
  ```py
  # targets in format [center_w, center_h, ww, hh], tt: targets * input_shape / stride
  def build_targets(pp, targets, model):
      num_targets = targets.shape[0]  # number of anchors, targets
      tcls, tbox, indices, anch = [], [], [], []
      gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
      off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

      g = 0.5  # offset
      multi_gpu = is_parallel(model)
      for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
          # get number of grid points and anchor vec for this yolo layer
          # anchor_vec = self.anchors / self.stride
          anchors = anchor_vec # [3， 2]
          gain[2:] = torch.tensor(pp[i].shape)[[3, 2, 3, 2]]  # xyxy gain, gain = input_shape / stride

          # Match targets to anchors, t: [num_targets, 6]
          a, tt, offsets = [], targets * gain, 0
          if num_targets:
              num_anchors = anchors.shape[0]  # number of anchors, 3
              at = torch.arange(num_anchors).view(num_anchors, 1).repeat(1, num_targets)  # [0, 1, 2] -> shape [3, num_targets]
              r = tt[None, :, 4:6] / anchors[:, None]  # wh ratio, [3, num_targets, 2]
              j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare, anchor_t = 4.0, j: [3, num_targets]
              # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
              a, tt = at[j], tt.repeat(num_anchors, 1, 1)[j]  # filter, tt: [num_targets, 6] -> t.repeat(3, 1, 1): [3, num_targets, 6]

              # overlaps
              gxy = tt[:, 2:4]  # grid xy
              z = torch.zeros_like(gxy)
              j, k = ((gxy % 1. < g) & (gxy > 1.)).T
              l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
              a, tt = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((tt, tt[j], tt[k], tt[l], tt[m]), 0)
              offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

          # Define
          b, c = tt[:, :2].long().T  # image, class
          gxy = tt[:, 2:4]  # grid xy
          gwh = tt[:, 4:6]  # grid wh
          gij = (gxy - offsets).long()
          gi, gj = gij.T  # grid xy indices

          # Append
          #indices.append((b, a, gj, gi))  # image, anchor, grid indices
          indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
          tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
          anch.append(anchors[a])  # anchors
          tcls.append(c)  # class

      return tcls, tbox, indices, anch
  ```
  ```py
  from keras_cv_attention_models.coco import data, anchors_func
  tt = data.init_dataset(batch_size=3, use_anchor_free_mode=True)[0]
  xx, yy = tt.as_numpy_iterator().next()
  flatten_targets = []
  for id, ii in enumerate(yy):
      valid = ii[np.any(ii > 0, axis=-1)]
      labels = tf.expand_dims(tf.cast(tf.argmax(valid[:, 4:-1], axis=-1), "float32"), 1)
      center_bboxes = tf.concat(anchors_func.corners_to_center_yxhw_nd(valid[:, :4]), axis=-1)
      batch_id = tf.zeros_like(valid[:, :1]) + id
      targets = tf.concat([batch_id, labels, center_bboxes], axis=-1).numpy()
      # print(targets.shape)
      flatten_targets.append(targets)
  flatten_targets = tf.concat(flatten_targets, axis=0).numpy()
  # anchors = anchors_func.get_yolor_anchors([256, 256])

  import torch
  gain = torch.ones(6)
  gain[2:] = torch.tensor([32, 32, 32, 32]) # 256 / 8

  # anchors = torch.tensor([[16.0, 12], [36, 19], [28, 40]]) / torch.tensor([256, 256]) # [3, 2]
  anchors = torch.tensor([[16.0, 12], [36, 19], [28, 40]]) / 8 # [3, 2]
  num_targets = flatten_targets.shape[0]
  num_anchors = anchors.shape[0]
  at = torch.arange(num_anchors).view(num_anchors, 1).repeat(1, num_targets) # [0, 1, 2] -> shape [3, num_targets]
  temp_targets = torch.from_numpy(flatten_targets) * gain # [num_targets, 6]
  ratio = temp_targets[None, :, 4:6] / anchors[:, None] # wh ratio, [3, num_targets, 2]
  pick = torch.max(ratio, 1. / ratio).max(2)[0] < 4.0  # compare, anchor_t = 4.0, j: [3, num_targets]
  anchors_pick, temp_targets = at[pick], temp_targets.repeat(num_anchors, 1, 1)[pick] # a: [j], temp_targets: [j, 6]

  # overlaps
  offset = 0.5
  off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]]).float()  # overlap offsets
  gxy = temp_targets[:, 2:4]  # grid xy, [j, 2], [center_w, center_h]
  z = torch.zeros_like(gxy)
  j, k = ((gxy % 1. < offset) & (gxy > 1.)).T
  l, m = ((gxy % 1. > (1 - offset)) & (gxy < (gain[[2, 3]] - 1.))).T
  anchors_pick = torch.cat((anchors_pick, anchors_pick[j], anchors_pick[k], anchors_pick[l], anchors_pick[m]), 0)
  temp_targets = torch.cat((temp_targets, temp_targets[j], temp_targets[k], temp_targets[l], temp_targets[m]), 0)
  offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * offset

  # Define
  b, c = temp_targets[:, :2].long().T  # image, class
  gxy = temp_targets[:, 2:4]  # grid xy
  gwh = temp_targets[:, 4:6]  # grid wh
  gij = (gxy - offsets).long()
  gi, gj = gij.T  # grid xy indices

  # Append
  #indices.append((b, a, gj, gi))  # image, anchor, grid indices
  indices = (b, anchors_pick, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))  # image, anchor, grid indices
  tbox = torch.cat((gxy - gij, gwh), 1)  # box
  anch = anchors[anchors_pick]  # anchors
  tcls = c  # class

  # model output
  from keras_cv_attention_models import yolor
  mm = yolor.YOLOR_CSP(input_shape=[256, 256, 3])
  pred = mm(xx)[:, :3072].numpy()
  pred = pred.reshape([-1, 32, 32, 3, 85]).transpose([0, 3, 1, 2, 4])

  b, a, gj, gi = indices # image, anchor, gridy, gridx
  ps = pred[b, a, gj, gi]
  ```
  ```py
  from keras_cv_attention_models.coco import data, anchors_func
  tt = data.init_dataset(batch_size=3, use_anchor_free_mode=True)[0]
  xx, yy = tt.as_numpy_iterator().next()

  from keras_cv_attention_models import yolor
  mm = yolor.YOLOR_CSP(input_shape=[256, 256, 3])
  pred = mm(xx).numpy()

  from keras_cv_attention_models.coco import data, anchors_func
  input_shape = [256, 256]
  anchor_ratios, feature_sizes = anchors_func.get_yolor_anchors([256, 256], pyramid_levels=[3, 5], is_for_training=True)
  print(sum([ii * jj * 3 for ii, jj in feature_sizes]))
  # 4032.0
  # feature_sizes: [(32, 32), (16, 16), (8, 8)]
  # anchor_ratios: [[[ 16.,  12.], [ 36.,  19.],[ 28.,  40.]],...]

  bbox_labels = tf.concat([yy[0][:, :4], tf.expand_dims(tf.cast(tf.argmax(yy[0][:, 4:-1], -1) + 1, 'float32'), -1)], axis=-1) * yy[0][:, -1:]

  bbox_labels = tf.gather_nd(yy[0], tf.where(yy[0][:, -1] > 0))
  bboxes, labels = bbox_labels[:, :4], bbox_labels[:, 4:]
  center_bboxes = tf.concat(anchors_func.corners_to_center_yxhw_nd(bboxes), axis=-1)
  input_shape = tf.convert_to_tensor(input_shape, tf.float32)
  anchor_aspect_thresh = 4.0
  num_anchors = anchor_ratios.shape[1]
  num_bboxes_true = bboxes.shape[0]
  overlap_offset = 0.5

  # pick by aspect ratio
  temp_center_bboxes = center_bboxes * tf.tile(feature_sizes[0], [2])
  aspect_ratio = tf.expand_dims(temp_center_bboxes[:, 2:], 0) / tf.expand_dims(anchor_ratios[0], 1)
  aspect_pick = tf.reduce_max(tf.maximum(aspect_ratio, 1 / aspect_ratio), axis=-1) < anchor_aspect_thresh
  anchors_pick = tf.repeat(tf.expand_dims(tf.range(num_anchors), -1), num_bboxes_true, axis=-1)[aspect_pick]
  aspect_picked_bboxes_labels = tf.concat([temp_center_bboxes, labels], axis=-1)
  aspect_picked_bboxes_labels = tf.repeat(tf.expand_dims(aspect_picked_bboxes_labels, 0), num_anchors, axis=0)[aspect_pick]

  # pick by centers
  centers = aspect_picked_bboxes_labels[:, :2]
  top, left = tf.unstack(tf.logical_and(centers % 1 < overlap_offset, centers > 1), axis=-1)
  bottom, right = tf.unstack(tf.logical_and(centers % 1 > (1 - overlap_offset), centers < (feature_sizes[0] - 1)), 2, axis=-1)
  anchors_pick_all = tf.concat([anchors_pick, anchors_pick[top], anchors_pick[left], anchors_pick[bottom], anchors_pick[right]], axis=0)
  matched_top, matched_left = aspect_picked_bboxes_labels[top], aspect_picked_bboxes_labels[left]
  matched_bottom, matched_right = aspect_picked_bboxes_labels[bottom], aspect_picked_bboxes_labels[right]
  matched_bboxes_all = tf.concat([aspect_picked_bboxes_labels, matched_top, matched_left, matched_bottom, matched_right], axis=0)

  matched_bboxes_idx = tf.cast(aspect_picked_bboxes_labels[:, :2], "int32")
  matched_top_idx = tf.cast(matched_top[:, :2] - [overlap_offset, 0], "int32")
  matched_left_idx = tf.cast(matched_left[:, :2] - [0, overlap_offset], "int32")
  matched_bottom_idx = tf.cast(matched_bottom[:, :2] + [overlap_offset, 0], "int32")
  matched_right_idx = tf.cast(matched_right[:, :2] + [0, overlap_offset], "int32")
  matched_bboxes_idx_all = tf.concat([matched_bboxes_idx, matched_top_idx, matched_left_idx, matched_bottom_idx, matched_right_idx], axis=0)

  bboxes_true = tf.concat([matched_bboxes_all[:, :2] - tf.cast(matched_bboxes_idx_all, matched_bboxes_all.dtype), matched_bboxes_all[:, 2:]], axis=-1)

  aa = tf.zeros([feature_sizes[0][0], feature_sizes[0][1], num_anchors, 5])
  bb = tf.tensor_scatter_nd_update(aa, tf.concat([matched_bboxes_idx_all, tf.expand_dims(anchors_pick_all, 1)], axis=-1), bboxes_true)

  print(tf.reduce_sum(tf.cast(tf.reduce_any(tf.reshape(bb, [-1, 5]) != 0, axis=-1), 'float32')))
  ```
  ```py
  # YOLOLayer
  io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
  io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
  io[..., :4] *= self.stride

  # compute_loss
  # anchors = YOLOLayer.anchor_vec = YOLOLayer.anchor_wh = YOLOLayer.anchors / strides
  pxy = ps[:, :2].sigmoid() * 2. - 0.5
  pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
  pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
  iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
  ```
## YOLOR losses
  ```py
  # https://github.com/WongKinYiu/yolor/blob/main/utils/general.py#L187
  def bbox_ciou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, ECIoU=False, eps=1e-9):
      # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
      box2 = box2.T

      # Get the coordinates of bounding boxes
      b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
      b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
      b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
      b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

      # Intersection area
      inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

      # Union Area
      w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
      w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
      union = w1 * h1 + w2 * h2 - inter + eps

      iou = inter / union

      cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
      ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

      c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
      rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
      v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
      with torch.no_grad():
          alpha = v / ((1 + eps) - iou + v)
      return iou - (rho2 / c2 + v * alpha)  # CIoU
  ```
  ```py
  def compute_loss(p, targets, model):  # predictions, targets, model
      device = targets.device
      #print(device)
      lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
      tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
      h = model.hyp  # hyperparameters

      # Define criteria
      BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
      BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

      # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
      cp, cn = smooth_BCE(eps=0.0)

      # Focal loss
      g = h['fl_gamma']  # focal loss gamma
      if g > 0:
          BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

      # Losses
      nt = 0  # number of targets
      no = len(p)  # number of outputs
      balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
      balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
      for i, pi in enumerate(p):  # layer index, layer predictions
          b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
          tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

          n = b.shape[0]  # number of targets
          if n:
              nt += n  # cumulative targets
              ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

              # Regression
              pxy = ps[:, :2].sigmoid() * 2. - 0.5
              pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
              pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
              iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
              lbox += (1.0 - iou).mean()  # iou loss

              # Objectness
              tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

              # Classification
              if model.nc > 1:  # cls loss (only if multiple classes)
                  t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                  t[range(n), tcls[i]] = cp
                  lcls += BCEcls(ps[:, 5:], t)  # BCE

              # Append targets to text file
              # with open('targets.txt', 'a') as file:
              #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

          lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

      s = 3 / no  # output count scaling
      lbox *= h['box'] * s
      lobj *= h['obj'] * s * (1.4 if no >= 4 else 1.)
      lcls *= h['cls'] * s
      bs = tobj.shape[0]  # batch size

      loss = lbox + lobj + lcls
      return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
  ```
## YOLOR LR
```py
lrf = 0.2
lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - lrf) + lrf
```
***

# YOLOXTiny training logs
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "YOLOXTiny, bs 64, basic": "checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_64_randaug_6_mosaic_0.5_RRC_1.0_lr512_0.008_wd_0.02_hist.json",
      "randaug_after_mosaic": "checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_64_randaug_after_mosaic_hist.json",
      "randaug_after_mosaic, randaug_scale_03": "checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_64_randaug_after_mosac_randaug_scale_03_hist.json",
      "randaug_after_mosaic, randaug_scale_03, random_hsv": "checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_64_randaug_after_mosac_randaug_scale_03_random_hsv_hist.json",
      # "randaug_after_mosaic, randaug_scale_03, effd_anchors": "checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_64_randaug_after_mosac_randaug_scale_03_effd_anchors_hist.json",
      "416, randaug_after_mosaic, randaug_scale_03, random_hsv": "checkpoints/YOLOXTiny_416_adamw_coco_2017_batchsize_64_randaug_after_mosaic_hist.json",
      "416, randaug_after_mosaic, random_hsv": "checkpoints/YOLOXTiny_416_adamw_coco_2017_batchsize_64_randaug_after_mosac_random_hsv_no_randaug_scale_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=3, base_size=8)
  ```
  - **Test**
  ```py
  from keras_cv_attention_models import efficientdet, efficientnet, yolox

  model = yolox.YOLOXTiny(pretrained='checkpoints/YOLOXTiny_256_adamw_coco_2017_batchsize_64_randaug_6_mosaic_0.5_RRC_1.0_lr512_0.008_wd_0.02.h5', input_shape=(256, 256, 3), rescale_mode='torch')

  # Run prediction
  from keras_cv_attention_models import test_images
  imm = test_images.dog_cat()
  bboxs, lables, confidences = model.decode_predictions(model(model.preprocess_input(imm)))[0]

  # Show result
  from keras_cv_attention_models.coco import data
  data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
  ```py
  import kecam
  mm = kecam.yolox.YOLOXTiny(anchors_mode="yolor", pretrained='checkpoints/test_latest.h5', rescale_mode='torch')

  imm = kecam.test_images.dog_cat()
  bboxs, lables, confidences = mm.decode_predictions(mm(mm.preprocess_input(imm)), score_threshold=0.05)[0]
  kecam.coco.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
  ```py
  import kecam
  mm = kecam.yolox.YOLOXTiny(input_shape=(256, 256, 3), anchors_mode="efficientdet", pretrained='checkpoints/test_latest.h5', rescale_mode='torch')

  imm = kecam.test_images.dog_cat()
  bboxs, lables, confidences = mm.decode_predictions(mm(mm.preprocess_input(imm)), score_threshold=0.001)[0]
  kecam.coco.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
  ```
  | optimizer | color                 | scale | positional   | rescale_mode | best           | latest |
  | --------- | --------------------- | ----- | ------------ | ------------ | -------------- | ------ |
  | adamw     | random_hsv            | 0     | rtsx         | torch        | Epoch 52 0.234 | 0.219  |
  | sgdw      | random_hsv            | 0     | rtsx         | torch        | Epoch 52 0.217 | 0.218  |
  | adamw     | random_hsv            | 0.3   | rtsx         | torch        | Epoch 48 0.242 | 0.230  |
  | lamb      | random_hsv            | 0.3   | rtsx         | torch        | Epoch 49 0.210 | 0.206  |
  | adamw     | randaug               | 0.3   | rtsx         | torch        | Epoch 50 0.236 | 0.231  |
  | adamw     | random_hsv            | 0.5   | rtsx         | torch        | Epoch 49 0.242 | 0.243  |
  | adamw     | randaug               | 0.5   | rtsx         | torch        | Epoch 53 0.234 | 0.234  |
  | adamw     | random_hsv            | 0.5   | tx           | torch        | Epoch 55 0.246 | 0.246  |
  | adamw     | random_hsv            | 0.5   | rtsx, mag 10 | torch        | Epoch 48 0.231 | 0.229  |
  | adamw     | random_hsv            | 0.8   | rtsx         | torch        | Epoch 55 0.241 | 0.241  |
  | adamw     | autoaug               | 0.5   | rtsx         | torch        | Epoch 53 0.232 | 0.228  |
  | adamw     | random_hsv            | 0.5   | rtsx         | tf           | Epoch 51 0.244 | 0.239  |
  | adamw     | random_hsv            | 0.5   | rts          | tf           | Epoch 46 0.235 | 0.235  |
  | adamw     | random_hsv            | 0.8   | rts          | tf           | Epoch 49 0.242 | 0.237  |
  | adamw     | random_hsv            | 0.8   | t            | tf           | Epoch 55 0.241 | 0.241  |
  | adamw     | random_hsv            | 0.8   | tx           | tf           | Epoch 54 0.243 | 0.245  |
  | adamw     | random_hsv            | 0.8   | tx           | torch        | Epoch 45 0.241 | 0.232  |
  | adamw     | random_hsv            | 0.5   | tx           | tf           | Epoch 54 0.249 | 0.244  |
  | adamw     | random_hsv            | 0.5   | txr          | tf           | Epoch 45 0.244 | 0.242  |
  | adamw     | random_hsv + contrast | 0.5   | tx           | tf           | Epoch 52 0.247 | 0.241  |
  | adamw     | random_hsv            | 0.5   | txs          | tf           | Epoch 54 0.246 | 0.244  |
  | adamw     | random_hsv            | 0.5   | tx           | raw01        | Epoch 52 0.247 | 0.243  |

  ```py
  # YOLOXTiny_416_adamw_coco_2017_batchsize_64_randaug_after_mosaic, random_hsv, scale 03, epoch 48
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.393
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.253
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.381
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.397
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.164
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.463
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649

  # YOLOXTiny_416_adamw_coco_2017_batchsize_64_randaug_after_mosaic_random_hsv_scale_05, epoch 49
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.394
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.255
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.256
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.374
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.392
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.466
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638

  # YOLOXTiny_416_adamw_coco_2017_batchsize_64_randaug_after_mosaic_random_hsv_scale_05_no_rotate_shear, epoch_55
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.246
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.393
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.260
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.260
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.385
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.246
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.402
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.160
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.655
  ```
***

# TOLOR_CSP training logs
## PyTorch yolor_csp
```py
{
  'lr0': 0.01, 'lrf': 0.2, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
  'box': 0.05, 'cls': 0.3, 'cls_pw': 1.0, 'obj': 0.7, 'obj_pw': 1.0, 'iou_t': 0.2, 'anchor_t': 4.0, 'fl_gamma': 0.0, 'hsv_h':0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
  'degrees': 0.0, 'translate': 0.1, 'scale': 0.9, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0
}
```
```sh
TF_XLA_FLAGS='--tf_xla_auto_jit=2' CUDA_VISIBLE_DEVICES='1' ./coco_train_script.py --det_header yolor.YOLOR_CSP -i 640 \
--optimizer sgdw --lr_base_512 0.1 --lr_min 0.002 --weight_decay 0.0005 --momentum 0.937 \
--positional_augment_methods t --magnitude 10 --mosaic_mix_prob 1.0 --rescale_mode raw01 --freeze_backbone_epochs 0 -b 48
```
