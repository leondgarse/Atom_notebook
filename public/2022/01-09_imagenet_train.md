# ___2022 - 01 - 09 ImageNet Train___
***

# Flops
  ```py
  import kecam
  from inspect import getmembers
  exclude = ["aotnet", "attention_layers", "coco", "common_layers", "download_and_load", "imagenet", "mlp_family", "model_surgery", "resnet_family", "test_images", "version", "visualizing"]
  aa = [ii for ii in getmembers(kecam) if not ii[0].startswith("_") and not ii[0] in exclude]

  bb = [ii for ii in getmembers(aa[0][1]) if ii[0][0].isupper()]

  cc = [ii for ii in getmembers(kecam.efficientdet) if ii[0][0].isupper()]

  rrs = {}
  # for ii in getmembers(aa[0][1]):
  for ii in getmembers(kecam.efficientdet):
      if ii[0][0].isupper():
          print(">>>>", ii[0])
          try:
              rrs[ii[0]] = kecam.model_surgery.get_flops(ii[1](pretrained=None))
              print(rrs)
          except:
              pass
  {kk: '{:.2f}G'.format(vv / 1e9) for kk, vv in rrs.items()}
  ```
  ```py
  from tensorflow.python.profiler import model_analyzer, option_builder
  import kecam

  mm = kecam.uniformer.UniformerSmall64()
  input_signature = [tf.TensorSpec(shape=(1, *ii.shape[1:]), dtype=ii.dtype, name=ii.name) for ii in mm.inputs]
  forward_graph = tf.function(mm, input_signature).get_concrete_function().graph
  options = option_builder.ProfileOptionBuilder.float_operation()
  graph_info = model_analyzer.profile(forward_graph, options=options)
  flops = graph_info.total_float_ops // 2
  print('Flops: {:,}, GFlops: {:.4f}G'.format(flops, flops / 1e9))
  ```
***

# Training logs
## AotNet50
  ```py
  import json
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      # "timm, Resnet50, A3, 160, Epoch 100": eval_func.parse_timm_log("../pytorch-image-models/log_my_RRC.foo", pick_keys=['loss', 'val_acc']),
      # "A3, cutmix reverse, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_timm_cutmix_hist.json",
      # "A3, globclip1, cutmix reverse, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_globclip1_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_timm_cutmix_hist.json",
      # "A3, globclip1, cutmix random, get_box rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_cutmix_hist.json",
      # "A3, globclip1, ce 0.2, cutmix random, get_box ectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_ce_mixup_cutmix_hist.json",
      # "A3, globclip1, cutmix random, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_cutmix_timm_get_box_hist.json",
      # "A3, globclip1, ce 0.2, cutmix random, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_ce_02_mixup_cutmix_timm_get_box_hist.json",
      # "aotnet50_lr_restart_33": "checkpoints/aotnet50_lr_restart_33_hist.json",
      # "aotnet50_progressive_3_lr_steps_100": "checkpoints/aotnet50_progressive_3_lr_steps_100_hist.json",

      # "A3, aug_nearest": "checkpoints/aotnet50_A3_aug_nearest_hist.json",
      # "A3, aug_bilinear_resize_bilinear": "checkpoints/aotnet50_A3_aug_bilinear_resize_bilinear_hist.json",
      # "A3, bicubic_no_antialias": "checkpoints/aotnet50_A3_bicubic_no_antialias_hist.json",
      # "A3, rescale_mode_tf, cutmix random": "checkpoints/aotnet50_A3_rescale_mode_tf_hist.json",
      "A3, cutmix reverse, 2": "checkpoints/aotnet50_cutmix_reverse_hist.json",
      # "A3, cutmix reverse, mixup reverse": "checkpoints/aotnet50_cutmix_reverse_mixup_reverse_hist.json",
      # "A3, rescale_mode_tf, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_rescale_tf_hist.json",
      # "A3, rescale_mode_tf, clip255, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_rescale_tf_clip255_hist.json",
      # "A3, progressive_4, cutmix reverse": "checkpoints/aotnet50_progressive_4_lr_steps_100_cutmix_reverse_hist.json",
      "A3, bilinear_antialias, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_bilinear_antialias_hist.json",
      "A3, adamw_1e3_wd5e2, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_adamw_1e3_wd5e2_hist.json",
      "A3, adamw_4e3_wd5e2, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_adamw_4e3_wd5e2_hist.json",
      "A3, adamw_8e3_wd5e2, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_adamw_8e3_wd5e2_hist.json",
      "A3, adamw_4e3_wd2e2, cutmix random": "checkpoints/aotnet50_cutmix_random_adamw_4e3_wd2e2_hist.json",
      "A3, adamw_8e3_wd2e2, cutmix random": "checkpoints/aotnet50_cutmix_random_adamw_8e3_wd2e2_hist.json",
      "A3, progressive_3_lr_steps_33_lr_t_mul_1": "checkpoints/aotnet50_progressive_3_lr_steps_33_lr_t_mul_1_hist.json",
      "A3, progressive_3_lr_steps_33_lr_t_mul_1_on_batch": "checkpoints/aotnet50_progressive_3_lr_steps_33_lr_t_mul_1_on_batch_hist.json",
      # "A3, adamw_8e3_wd5e2, cutmix reverse, CE_mixup_0.8": "checkpoints/aotnet50_cutmix_reverse_adamw_8e3_wd5e2_CE_mixup_0.8_hist.json",
      "A3, evonorm": "checkpoints/aotnet50_evonorm_hist.json",
  }
  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=40, base_size=8)
  ```

|                                                            | Train acc | Eval loss | Eval 160     | Eval 224         | Epoch 106        |
| ---------------------------------------------------------- | --------- | --------- | ------------ | ---------------- | ---------------- |
| A3, cutmix reverse, get_box no-rectangle                   | 0.6322    | 0.001470  | E100, 0.7667 | 0.78214, 0.94126 | 0.78206, 0.94114 |
| A3, globclip1, cutmix reverse, get_box no-rectangle        | 0.6318    | 0.001452  | E103, 0.7674 | 0.78466, 0.94088 | 0.78476, 0.94098 |
| A3, globclip1, cutmix random, get_box rectangle            | 0.6426    | 0.001465  | E100, 0.7652 | 0.78208, 0.93824 | 0.78138, 0.93820 |
| A3, globclip1, ce 0.2, cutmix random, get_box rectangle    | 0.6629    | 0.94055   | E102, 0.7609 | 0.77254, 0.93778 |                  |
| A3, globclip1, cutmix random, get_box no-rectangle         | 0.6330    | 0.001478  | E100, 0.7682 | 0.78256, 0.94008 | 0.78276, 0.93966 |
| A3, globclip1, ce 0.2, cutmix random, get_box no-rectangle | 0.6880    | 0.93494   | E100, 0.7615 | 0.77174, 0.9373  |                  |
| A3, restart_33, cutmix random, get_box no-rectangle        | 0.6326    | 0.001439  | E100, 0.7644 | 0.78196, 0.93898 | 0.78218, 0.93922 |
| A3, progressive_3, cutmix random, get_box no-rectangle     | 0.6293    | 0.001438  | E101, 0.7672 | 0.78074, 0.93912 | 0.78090, 0.93912 |
| A3, progressive_4, cutmix reverse, get_box no-rectangle    | 0.6207    | 0.001485  | E102, 0.7645 | 0.77410, 0.93454 |                  |
| A3, cutmix reverse, mixup reverse                          | 0.6321    | 0.001463  | E99, 0.7672  | 0.78338, 0.93958 | 0.78336, 0.93952 |
| A3, evonorm                                                | 0.6346    | 0.001388  | E103, 0.7693 | 0.78652, 0.94190 | 0.78630, 0.94192 |

- **cutmix random, mixup random**

|                                                | Train acc | Eval loss | Eval 160     | Eval 224         | Epoch 106        |
| ---------------------------------------------- | --------- | --------- | ------------ | ---------------- | ---------------- |
| A3, aug_nearest                                | 0.6331    | 0.001458  | E101, 0.7653 | 0.78196, 0.9401  |                  |
| A3, resize_bilinear                            | 0.6310    | 0.001455  | E103, 0.7642 | 0.78024, 0.93974 | 0.78072, 0.93996 |
| A3, bicubic_no_antialias                       | 0.6313    | 0.001481  | E97, 0.7626  | 0.77994, 0.938   | 0.77956, 0.93808 |
| A3, rescale_mode_tf                            | 0.6328    | 0.001452  | E97, 0.7671  | 0.78316, 0.93898 | 0.78310, 0.93910 |
| A3, rescale_mode_tf, cutmix_reverse            | 0.6319    | 0.001458  | E100, 0.7688 | 0.78274, 0.94066 | 0.78292, 0.94046 |
| A3, resize_bilinear, antialias, cutmix_reverse | 0.6296    | 0.001491  | E104, 0.7676 | 0.78152, 0.93924 | 0.78128, 0.93944 |

- **Adamw**
```py
import json
from keras_cv_attention_models.imagenet import eval_func
hhs = {
    "A3, lamb, lr 8e-3, wd 2e-2": "checkpoints/aotnet50_cutmix_reverse_hist.json",
    "A3, adamw, lr 4e-3, wd 5e-2": "checkpoints/aotnet50_cutmix_reverse_adamw_4e3_wd5e2_hist.json",
    # "A3, adamw, lr 8e-3, wd 5e-2": "checkpoints/aotnet50_cutmix_reverse_adamw_8e3_wd5e2_hist.json",
    "A3, adamw, lr 4e-3, wd 2e-2": "checkpoints/aotnet50_cutmix_random_adamw_4e3_wd2e2_hist.json",
    "A3, adamw, lr 8e-3, wd 2e-2": "checkpoints/aotnet50_cutmix_random_adamw_8e3_wd2e2_hist.json",
    "A3, adamw, lr 8e-3, wd 2e-2, cutmix reverse": "checkpoints/aotnet50_cutmix_reverse_adamw_8e3_wd2e2_hist.json",
    "A3, adamw, lr 1e-2, wd 2e-2, cutmix reverse": "checkpoints/aotnet50_hist.json",
}
fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=3, base_size=8)
```

| lr base | Weight decay   | Train acc | Best Eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
| ------- | -------------- | --------- | --------------------------- | -------------------------- | ------------------ |
| 4e-3    | 0.05           | 0.6216    | Epoch 102, 0.001468, 0.7638 | 0.77862, 0.93876           | 0.77918, 0.93850   |
| 4e-3    | 0.02           | 0.6346    | Epoch 100, 0.001471, 0.7669 | 0.78060, 0.93842           | 0.78058, 0.93856   |
| 8e-3    | 0.02           | 0.6285    | Epoch 105, 0.001463, 0.7675 | 0.78268, 0.93828           | 0.78268, 0.93828   |
| 8e-3    | 0.02 (reverse) | 0.6274    | Epoch 103, 0.001511, 0.7646 | 0.78184, 0.93860           | 0.78142, 0.93862   |

- **A2 224**
  ```py
  import json
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "AotNet50, A3, 160, Epoch 100": "checkpoints/aotnet.AotNet50_LAMB_nobn_globclip1_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_timm_cutmix_hist.json",
      "AotNet50, A2, 224, Epoch 100": "checkpoints/aotnet.AotNet50_224_LAMB_imagenet2012_batchsize_128_randaug_7_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.005_wd_0.02_hist.json",
      "AotNet50, A2, 224, Epoch 300": {kk: vv[::3] for kk, vv in json.load(open("checkpoints/aotnet.AotNet50_A2_hist.json", "r")).items()},
      # "ConvNeXt-T": {kk: vv[::3] for kk, vv in log.bb.items()},
  }
  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=1, base_size=8)
  ```
## CoAtNet
  | Model       | stem                      | res_MBConv block      | res_mhsa block        | res_ffn block                 |
  | ----------- | ------------------------- | --------------------- | --------------------- | ----------------------------- |
  | CoAtNet0    | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm bn + gelu, V2 | bn,gelu,conv,bn,gelu,conv     |
  | CoAtNet0_2  | conv,bn,gelu,conv,bn,gelu | prenorm bn, V1        | prenorm ln, V1        | ln,conv,gelu,conv             |
  | CoAtNet0_3  | conv,bn,gelu,conv         | prenorm bn, V1        | prenorm ln, V1        | ln,conv,gelu,conv             |
  | CoAtNet0_4  | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm ln, V1        | ln,conv,gelu,conv             |
  | CoAtNet0_5  | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm bn + gelu, V1 | ln,conv,gelu,conv             |
  | CoAtNet0_6  | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm bn + gelu, V2 | ln,conv,gelu,conv             |
  | CoAtNet0_7  | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm bn + gelu, V2 | bn,gelu,conv,gelu,conv        |
  | CoAtNet0_8  | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm bn + gelu, V2 | bn,conv,gelu,conv             |
  | CoAtNet0_9  | conv,bn,gelu,conv         | prenorm bn + gelu, V2 | prenorm bn + gelu, V2 | bn,conv(bias),gelu,conv(bias) |
  | CoAtNet0_11 | conv,bn,gelu,conv         | prenorm bn, V2        | prenorm bn, V2        | bn,conv,gelu,conv             |
  | CoAtNet0_13 | conv,bn,gelu,conv         | prenorm bn + gelu, V1 | prenorm bn + gelu, V1 | bn,conv,gelu,conv             |
  | CoAtNet0_14 | conv,bn,gelu,conv         | prenorm bn, V1        | prenorm bn, V1        | bn,conv,gelu,conv             |
  | CoAtNet0_15 | conv,bn,gelu,conv         | prenorm bn, V2        | prenorm ln, V2        | ln,conv,gelu,conv             |
  | CoAtNet0_16 | conv,bn,gelu,conv         | prenorm bn, V1        | prenorm ln, V1        | ln,conv,gelu,conv             |
  | CoAtNet0_17 | conv,bn,gelu,conv         | prenorm bn, V1        | prenorm ln, V2        | ln,conv,gelu,conv             |

  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      # "AotNet50, A2, 224, Epoch 300": {kk: vv[::3] for kk, vv in json.load(open("checkpoints/aotnet.AotNet50_A2_hist.json", "r")).items()},
      # "CoAtNet0, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
      # "CoAtNet0_2, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_2_hist.json",
      # "CoAtNet0_3, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_3_hist.json",
      # "CoAtNet0_4, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_4_hist.json",
      # "CoAtNet0_5, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_5_hist.json",
      # "CoAtNet0_6, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_6_hist.json",
      # "CoAtNet0_7, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_7_hist.json",
      "CoAtNet0_8, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_8_hist.json",
      # "CoAtNet0_9, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_9_hist.json",
      # "CoAtNet0_11, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_11_hist.json",
      # "CoAtNet0_13, A3": "checkpoints/ccoatnet0/oatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_13_hist.json",
      # "CoAtNet0_14, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_14_hist.json",
      # "CoAtNet0_15, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_15_hist.json",
      "CoAtNet0_16, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_16_hist.json",
      # "CoAtNet0_160_dpc_02, bs 90": "checkpoints/coatnet.CoAtNet0_160_dpc_02_hist.json",
      "CoAtNet0_160_dpc_0.05, bs 128": "checkpoints/CoAtNet0_drc_005_hist.json",
      "CoAtNet0_160, wd_exclude_pos_emb, bs 128": "checkpoints/CoAtNet0_wd_exclude_pos_emb_hist.json",
      "CoAtNet0_160, wd_exclude_pos_emb, mag_10, bs 128": "checkpoints/CoAtNet0_wd_exclude_pos_emb_mag_10_hist.json",
      "CoAtNet0_160_act_first, bs 128": "checkpoints/CoAtNet0_160_act_first_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=25, base_size=8)
  ```
  - **fine-tune 224, lr_decay_steps 32, lr_base_512 0.004, batch_size 64**
  ```sh
  CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py --seed 0 \
  -m coatnet.CoAtNet0 --pretrained imagenet -i 224 --batch_size 64 \
  --lr_decay_steps 32 --lr_warmup_steps 0 --lr_base_512 0.004 \
  --additional_model_kwargs '{"drop_connect_rate": 0.05}' --magnitude 7 \
  -s coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc005_magnitude_7
  ```
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "ft 224, magnitude 6, drc 0, bs 64": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_hist.json",
      "ft 224, magnitude 7, drc 0, bs 64": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_magnitude_7_hist.json",
      "ft 224, magnitude 7, drc 0.05, bs 64": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc005_magnitude_7_hist.json",
      "ft 224, magnitude 7, drc 0.2, bs 64": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc02_magnitude_7_hist.json",
      "ft 224, magnitude 10, drc 0.05, bs 64, ": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc005_magnitude_10_1_hist.json",
      "ft 224, magnitude 10, drc 0.05, wd exc pos_emb, bs 64": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc005_magnitude_10_hist.json",
      "ft 224, magnitude 15, drc 0.05, wd exc pos_emb, bs 64": "checkpoints/coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc005_magnitude_15_hist.json",
      "ft 224, magnitude 10, drc 0.05, act_first": "checkpoints/coatnet.CoAtNet0_act_first_ft_224_lr_steps_32_lr4e3_drc005_magnitude_10_hist.json",
      "ft 224, magnitude 15, drc 0.05, act_first": "checkpoints/coatnet.CoAtNet0_act_first_ft_224_lr_steps_32_lr4e3_drc005_magnitude_15_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=0, base_size=8)
  ```

| magnitude | drop_connect_rate | Best val loss, acc                                                          |
| --------- | ----------------- | --------------------------------------------------------------------------- |
| 6         | 0                 | Epoch 35/37 loss: 0.0023 - acc: 0.7288 - val_loss: 0.0012 - val_acc: 0.8160 |
| 7         | 0                 | Epoch 34/37 loss: 0.0024 - acc: 0.7218 - val_loss: 0.0012 - val_acc: 0.8161 |
| 7         | 0.05              | Epoch 36/37 loss: 0.0026 - acc: 0.7026 - val_loss: 0.0011 - val_acc: 0.8193 |
| 7         | 0.2               | Epoch 34/37 loss: 0.0030 - acc: 0.6658 - val_loss: 0.0011 - val_acc: 0.8176 |
| 10        | 0.05              | Epoch 36/37 loss: 0.0028 - acc: 0.6783 - val_loss: 0.0011 - val_acc: 0.8199 |
***

# CMT
  ```sh
  CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py --seed 0 -m cmt.CMTTiny -b 256 --lr_decay_steps 50 -s cmt.CMTTiny_160_downsample_ln_use_bias_false
  ```
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "basic, bs 256": "checkpoints/cmt.CMTTiny_160_LAMB_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
      "downsample_ln, bs 256": "checkpoints/cmt.CMTTiny_160_downsample_ln_hist.json",
      "downsample_ln, use_bias_false, bs 256": "checkpoints/cmt.CMTTiny_160_downsample_ln_use_bias_false_hist.json",
      "downsample_ln, stem_use_bias, bs 256": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_hist.json",
      "wd_exclude_pos": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_bot_pos_wd_exclude_pos_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=1, base_size=8)
  ```
  | Model                           | Train acc | Best Eval loss, acc on 160   |
  | ------------------------------- | --------- | ---------------------------- |
  | Basic                           | 0.5949    | Epoch 53, 0.0017, 0.7445     |
  | - downsample_ln                 | 0.5988    | Epoch 54, 0.0016, 0.7486     |
  | - downsample_ln, use_bias_false | 0.5674    | Epoch 43, 0.0016, 0.7283     |
  | - downsample_ln, stem_use_bias  | 0.6013    | Epoch 55, 0.0016, **0.7523** |
  | + wd_exclude_pos                | 0.5927    | Epoch 47, 0.0016, 0.7472     |
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "downsample_ln, stem_use_bias, value, [dim, head, 2], bs 256": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_hist.json",
      "value, [split_2, head, dim]": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_fixed_pos_hist.json",
      "value, [split_2, dim, head]": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_kv_dim_heads_hist.json",
      "value, [dim, head, 2], 2": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_2_hist.json",
      "value, [dim, head, 2], wd_exc_bias": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_head_kv_hist.json",
      "value, [head, dim, 2]": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_kv_reshape_head_dim_hist.json",
      "value, [head, dim, 2], wd_exc_bias": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_kv_reshape_head_dim_wd_exc_bias_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=1, base_size=8)
  ```
  | Model                                                       | Train acc | Best Eval loss, acc on 160   |
  | ----------------------------------------------------------- | --------- | ---------------------------- |
  | downsample_ln, stem_use_bias, value, [dim, head, 2], bs 256 | 0.6013    | Epoch 55, 0.0016, **0.7523** |
  | value, [split_2, head, dim]                                 | 0.6007    | Epoch 52, 0.0016, 0.7499     |
  | value, [split_2, dim, head]                                 | 0.3178    | Epoch 12, 0.0032, 0.4766     |
  | value, [dim, head, 2], 2                                    | 0.5997    | Epoch 55, 0.0016, **0.7517** |
  | value, [dim, head, 2], wd_exc_bias                          | 0.5166    | Epoch 34, 0.0020, 0.6895     |
  | value, [head, dim, 2]                                       | 0.5995    | Epoch 55, 0.0015, 0.7483     |
  | value, [head, dim, 2], wd_exc_bias                          | 0.5980    | Epoch 55, 0.0016, 0.7500     |
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "downsample_ln, stem_use_bias, bs 256": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_hist.json",
      "beit_pos, value, [split_2, head, dim], num_heads 1": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_fixed_pos_beit_pos_hist.json",
      "beit_pos, value, [split_2, head, dim], MH": "checkpoints/cmt.CMTTiny_160_downsample_ln_stem_use_bias_fixed_pos_beit_pos_multi_head_hist.json",
      "beit_pos, value, [dim, head, 2], wd_exc_bias": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_head_kv_beit_pos_hist.json",
      "beit_pos, value, [head, dim, 2], wd_exc_bias": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_head_dim_kv_beit_pos_hist.json",
      "beit_pos, value, [split_2, head, dim], wd_exc_bias": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_hist.json",
      # "beit_pos, [split_2, head, dim], irffn_v2": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_irffn_v2_hist.json",
      "beit_pos, [split_2, head, dim], output_act_first": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_output_act_first_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=1, base_size=8)
  ```
  ```py
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "beit_pos, [split_2, head, dim], output_act_first": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_output_act_first_hist.json",
      "[split_2, head, dim], avg_pool": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_kv_avg_pool_hist.json",
      "[split_2, head, dim], max_pool": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_kv_max_pool_hist.json",
      "[dim, head, 2], avg_pool": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_head_kv_beit_pos_kv_max_pool_hist.json",
      "[split_2, head, dim], avg_pool, stack_conv_bias": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_kv_head_dim_beit_pos_avg_pool_stack_conv_bias_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=1, base_size=8)
  ```
  ```py
  import json
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "Base line, AtoNet50, A3": "checkpoints/aotnet50_cutmix_reverse_hist.json",
      "CMTTiny, lmhsa, dw+ln, KV [dim, head, 2]": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_stack_conv_bias_lmhsa_dw_ln_A3_hist.json",
      "CMTTiny, lmhsa, avg pool, KV [dim, head, 2]": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_stack_conv_bias_A3_hist.json",
      "CMTTiny, lmhsa, dw+ln, KV [split2, head, dim]": "checkpoints/cmt.CMTTiny_160_exc_bias_key_value_split2_head_dim_stack_conv_bias_lmhsa_dw_ln_A3_hist.json",
      "CMTTiny, epoch 305": {kk: vv[::3] for kk, vv in json.load(open('checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_head_2_beit_pos_dw_ln_300_hist.json', 'r')).items()},
      "CMTTiny, epoch 305, mag15": {kk: vv[::3] for kk, vv in json.load(open('checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_head_2_beit_pos_dw_ln_300_mag15_hist.json', 'r')).items()},
      "CMTTiny, epoch 305, mag7, drc0.05, bs 160": {kk: vv[::3] for kk, vv in json.load(open('checkpoints/cmt.CMTTiny_160_exc_bias_key_value_dim_head_2_beit_pos_dw_ln_300_mag7_drc_005_bs_240_hist.json', 'r')).items()},
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=20, base_size=8)
  ```
  | 305 epochs             | Train acc | Best eval loss, acc on 160 | Epoch 105 Eval acc on 224   |
  | ---------------------- | --------- | -------------------------- | --------------------------- |
  | mag6, drc 0, bs 256    | 0.6702    | Epoch 304, 0.0013, 0.7874  | top1: 0.79956 top5: 0.94850 |
  | mag7, drc 0.05, bs 160 | 0.6577    | Epoch 294, 0.0013,0.7880   | top1: 0.80126 top5: 0.94898 |
  | mag15, drc 0, bs 256   | 0.6390    | Epoch 304, 0.0014,0.7824   | top1: 0.79630 top5: 0.94794 |
***
# SwinTransformerV2Tiny_ns
```py
from keras_cv_attention_models.imagenet import eval_func
hhs = {
    "CoAtNet0_160, wd_exclude_pos_emb, bs 128": "checkpoints/CoAtNet0_wd_exclude_pos_emb_hist.json",
    "swin_transformer_v2, bs 128": "checkpoints/swin_transformer_v2.SwinTransformerV2Tiny_ns_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
    "swin_transformer_v2, bs 128, adamw": "checkpoints/swin_transformer_v2.SwinTransformerV2Tiny_ns_160_adamw_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_hist.json",
}

fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=15, base_size=8)
```
