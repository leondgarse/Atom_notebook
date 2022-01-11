# ___2022 - 01 - 09 ImageNet Train___
***

# Training logs
## AotNet50
  ```py
  import json
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      # "timm, Resnet50, A3, 160, Epoch 100": eval_func.parse_timm_log("../pytorch-image-models/log_my_RRC.foo", pick_keys=['loss', 'val_acc']),
      # "A3, cutmix reverse, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_timm_cutmix_hist.json",
      "A3, globclip1, cutmix reverse, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_globclip1_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_timm_cutmix_hist.json",
      "A3, globclip1, cutmix random, get_box rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_cutmix_hist.json",
      # "A3, globclip1, ce 0.2, cutmix random, get_box ectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_ce_mixup_cutmix_hist.json",
      "A3, globclip1, cutmix random, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_cutmix_timm_get_box_hist.json",
      # "A3, globclip1, ce 0.2, cutmix random, get_box no-rectangle": "checkpoints/aotnet.AotNet50_LAMB_nobn_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_ce_02_mixup_cutmix_timm_get_box_hist.json",
      # "aotnet50_lr_restart_33": "checkpoints/aotnet50_lr_restart_33_hist.json",
      # "aotnet50_progressive_3_lr_steps_100": "checkpoints/aotnet50_progressive_3_lr_steps_100_hist.json",

      "aotnet50_A3_aug_nearest": "checkpoints/aotnet50_A3_aug_nearest_hist.json",
      "aotnet50_A3_aug_bilinear_resize_bilinear": "checkpoints/aotnet50_A3_aug_bilinear_resize_bilinear_hist.json",
      "aotnet50_A3_bicubic_no_antialias": "checkpoints/aotnet50_A3_bicubic_no_antialias_hist.json",
  }
  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=40, base_size=8)
  ```
  |                                                            | Train acc | Eval 160 | Eval 224         |
  | ---------------------------------------------------------- | --------- | -------- | ---------------- |
  | A3, cutmix reverse, get_box no-rectangle                   | 0.6322    | 0.7667   | 0.78214, 0.94126 |
  | A3, globclip1, cutmix reverse, get_box no-rectangle        | 0.6318    | 0.7674   | 0.78466, 0.94088 |
  | A3, globclip1, cutmix random, get_box rectangle            | 0.6426    | 0.7652   | 0.78208, 0.93824 |
  | A3, globclip1, ce 0.2, cutmix random, get_box rectangle    | 0.6629    | 0.7609   | 0.77254, 0.93778 |
  | A3, globclip1, cutmix random, get_box no-rectangle         | 0.6330    | 0.7682   | 0.78256, 0.94008 |
  | A3, globclip1, ce 0.2, cutmix random, get_box no-rectangle | 0.6880    | 0.7615   | 0.77174, 0.9373  |
  | A3, restart_33, cutmix random, get_box no-rectangle        | 0.6326    | 0.7644   | 0.78196, 0.93898 |
  ```py
  import json
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "AotNet50, A3, 160, Epoch 100": "checkpoints/aotnet.AotNet50_LAMB_nobn_globclip1_imagenet2012_batchsize_256_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_bce_0.2_mixup_timm_cutmix_hist.json",
      "AotNet50, A2, 224, Epoch 100": "checkpoints/aotnet.AotNet50_224_LAMB_imagenet2012_batchsize_128_randaug_7_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.005_wd_0.02_hist.json",
      "AotNet50, A2, 224, Epoch 300": {kk: vv[::3] for kk, vv in json.load(open("checkpoints/aotnet.AotNet50_A2_hist.json", "r")).items()},
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
  from keras_cv_attention_models import imagenet
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
      "CoAtNet0_11, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_11_hist.json",
      # "CoAtNet0_13, A3": "checkpoints/ccoatnet0/oatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_13_hist.json",
      # "CoAtNet0_14, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_14_hist.json",
      "CoAtNet0_15, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_15_hist.json",
      "CoAtNet0_16, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_16_hist.json",
      "CoAtNet0_160_dpc_02, bs 90": "checkpoints/coatnet.CoAtNet0_160_dpc_02_hist.json",
  }

  fig = imagenet.plot_hists(hhs.values(), list(hhs.keys()), skip_first=30, base_size=8)
  ```
***
