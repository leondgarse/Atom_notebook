# ___YOLOX___
***
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
## Pytorch post process
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
## TF post process
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
## bboxes center iou
  ```py
  def bboxes_iou(bboxes_a, bboxes_b):
      tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
      br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

      area_a = torch.prod(bboxes_a[:, 2:], 1)
      area_b = torch.prod(bboxes_b[:, 2:], 1)
      en = (tl < br).type(tl.type()).prod(dim=2)
      area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
      return area_i / (area_a[:, None] + area_b - area_i)
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
      args=torch.randn(10, 3, *input_shape),
      f=output_name,
      verbose=False,
      keep_initializers_as_inputs=True,
      training=torch.onnx.TrainingMode.PRESERVE,
      do_constant_folding=True,
      opset_version=13,
  )
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
## Reload YOLOR
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
  for ii in range(1, 4):
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
## Verification
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
  ```py
  # [yolo]
  # mask = 0,1,2
  # anchors = 12, 16, 19
  # classes=80
  # num=9
  # jitter=.3
  # ignore_thresh = .7
  # truth_thresh = 1
  # random=1
  # scale_x_y = 1.05
  # iou_thresh=0.213
  # cls_normalizer=1.0
  # iou_normalizer=0.07
  # iou_loss=ciou
  # nms_kind=greedynms
  # beta_nms=0.6
  ```
***
