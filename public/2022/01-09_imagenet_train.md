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
  }
  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=40, base_size=8)
  ```
  ```py
  import json
  from keras_cv_attention_models.imagenet import eval_func
  hhs = {
      "A3, lamb, lr 8e-3, wd 2e-2": "checkpoints/aotnet50_cutmix_reverse_hist.json",
      "A3, adamw, lr 4e-3, wd 5e-2": "checkpoints/aotnet50_cutmix_reverse_adamw_4e3_wd5e2_hist.json",
      # "A3, adamw, lr 8e-3, wd 5e-2": "checkpoints/aotnet50_cutmix_reverse_adamw_8e3_wd5e2_hist.json",
      "A3, adamw, lr 4e-3, wd 2e-2": "checkpoints/aotnet50_cutmix_random_adamw_4e3_wd2e2_hist.json",
      "A3, adamw, lr 8e-3, wd 2e-2": "checkpoints/aotnet50_cutmix_random_adamw_8e3_wd2e2_hist.json",
  }
  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=3, base_size=8)
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

| lr base | Weight decay | Train acc | Best Eval loss, acc on 160  | Eval acc top1, top5 on 224 | Epoch 105 eval acc |
| ------- | ------------ | --------- | --------------------------- | -------------------------- | ------------------ |
| 4e-3    | 0.05         | 0.6216    | Epoch 102, 0.001468, 0.7638 | 0.77862, 0.93876           |                    |
| 4e-3    | 0.02         | 0.6346    | Epoch 100, 0.001471, 0.7669 | 0.78060, 0.93842           | 0.78058, 0.93856   |
| 8e-3    | 0.02         | 0.6285    | Epoch 105, 0.001463, 0.7675 | 0.78268, 0.93828           | 0.78268, 0.93828   |

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
      "CoAtNet0_11, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_11_hist.json",
      # "CoAtNet0_13, A3": "checkpoints/ccoatnet0/oatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_13_hist.json",
      # "CoAtNet0_14, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_14_hist.json",
      "CoAtNet0_15, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_15_hist.json",
      "CoAtNet0_16, A3": "checkpoints/coatnet0/coatnet.CoAtNet0_160_LAMB_imagenet2012_batchsize_128_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_16_hist.json",
      "CoAtNet0_160_dpc_02, bs 90": "checkpoints/coatnet.CoAtNet0_160_dpc_02_hist.json",
      "CoAtNet0_160_dpc_0.05, bs 128": "checkpoints/CoAtNet0_drc_005_hist.json",
  }

  fig = eval_func.plot_hists(hhs.values(), list(hhs.keys()), skip_first=10, base_size=8)
  ```
***
```py
from sklearn.preprocessing import normalize
emb_unk = normalize(emb_unk)

# Use emb_unk[0] as known emb for test
cos_dist = np.dot(emb_unk[0], emb_unk.T)
euc_dist = np.sqrt(((emb_unk[0] - emb_unk) ** 2).sum(1))
print(np.allclose(np.sqrt(2 - 2 * cos_dist), euc_dist))

(2 - euc_dist ** 2) / 2
# True
```
