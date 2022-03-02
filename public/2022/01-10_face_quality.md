# ___2022 - 01 - 10 Face Quality___
***

# Similarity with face mask
```py
aa, bb = tf.random.uniform([1, 512]), tf.random.uniform([2, 512])

def half_split_weighted_cosine_similarity_11(aa, bb):
    half = aa.shape[-1] // 2
    bb = bb[:aa.shape[0]]

    top_weights = tf.norm(aa[:, :half], axis=1) * tf.norm(bb[:, :half], axis=1)
    bottom_weights = tf.norm(aa[:, half:], axis=1) * tf.norm(bb[:, half:], axis=1)

    top_sim = tf.reduce_sum(aa[:, :half] * bb[:, :half], axis=-1)
    bottom_sim = tf.reduce_sum(aa[:, half:] * bb[:, half:], axis=-1)
    return (top_sim + bottom_sim) / (top_weights + bottom_weights)

def half_split_weighted_cosine_similarity(aa, bb):
    half = aa.shape[-1] // 2
    bb = tf.transpose(bb)

    top_weights = tf.norm(aa[:, :half], axis=-1, keepdims=True) * tf.norm(bb[:half], axis=0, keepdims=True)
    bottom_weights = tf.norm(aa[:, half:], axis=-1, keepdims=True) * tf.norm(bb[half:], axis=0, keepdims=True)

    top_sim = aa[:, :half] @ bb[:half]
    bottom_sim = aa[:, half:] @ bb[half:]
    return (top_sim + bottom_sim) / (top_weights + bottom_weights)
```
# MagFace Face Quality test
  ```py
  from glob2 import glob
  imms = np.stack([(plt.imread(ii) - 127.5) * 0.0078125 for ii in glob('../face_flaw_aligned_112_112/*.jpg')])
  mm = keras.models.load_model('checkpoints/TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_53_0.985000.h5')
  ees = mm(imms)
  scores = tf.norm(ees, axis=1).numpy()
  idx = np.argsort(scores)
  imms, scores = imms[idx], scores[idx]

  sys.path.append('../keras_cv_attention_models/')
  from keras_cv_attention_models import visualizing

  _ = visualizing.stack_and_plot_images(imms / 2 + 0.5, texts=scores)
  ```
  ```py
  import evals
  bb = keras.models.load_model('checkpoints/TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest.h5')

  data_basic_path = '/datasets/ms1m-retinaface-t1'
  eval_paths = [os.path.join(data_basic_path, ii) for ii in ['lfw.bin', 'cfp_fp.bin', 'agedb_30.bin']]
  for ee in eval_paths:
      eea = evals.eval_callback(bb, ee, batch_size=16)
      eea.on_epoch_end()
      # eea.embs = np.load(eea.test_names + '.npy')

      # Plot face quality distribution using norm value of feature
      norm_embs = tf.norm(eea.embs, axis=1).numpy()
      _ = plt.hist(norm_embs, bins=512, alpha=1/len(eval_paths), label=eea.test_names + ' quality')
  plt.legend()
  plt.tight_layout()
  ```
***
# MagFace Training log
  - **Effv2S MagFace and Curricular**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 10
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_01_wd5e2lr_arc_emb512_adamw_exclude_bn_bs512_ms1m_float16_hist.json", fig_label="F, point_wise 512, wd 5e-4, dr 0.2, adamw cos16, ArcFace", names=names, **pp)
  pp["axes"] = axes

  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_curr_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_hist.json", fig_label="F, adamw cos16, curr", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_curr_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_2_hist.json", fig_label="F, adamw cos16, curr_2", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_hist.json", fig_label="mag, margin (0.45, 0.8), FN (10, 110)", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag05_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_hist.json", fig_label="mag, margin (0.5, 0.8), FN (10, 110)", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_05_1_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="mag, margin (0.5, 1.0), FN (10, 110)", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_1_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="mag, margin (0.4, 1.0), FN (10, 110)", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="mag, margin (0.4, 0.8), FN (10, 110)", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="mag, margin (0.4, 1.0), FN (1, 51)", **pp)
  ```
  | margin min max | Epoch 53 margin mean | lfw      | cfp_fp   | agedb_30 | IJBB         | IJBC         |
  | -------------- | -------------------- | -------- | -------- | -------- | ------------ | ------------ |
  | 0.45, 0.8      | missing, ~0.5        | 0.998500 | 0.991429 | 0.985000 | 0.957157     |              |
  | 0.5, 0.8       | missing, ~0.6        | 0.998500 | 0.991714 | 0.984500 | 0.957352     | 0.970803     |
  | 0.5, 1.0       | 0.554971933          | 0.998500 | 0.991429 | 0.984167 | 0.956865     | 0.969832     |
  | 0.4, 1.0       | 0.487977058          | 0.998167 | 0.991571 | 0.983333 | 0.957157     |              |
  | 0.4, 0.8       | 0.480038822          | 0.998500 | 0.991571 | 0.984667 | **0.958325** | **0.971212** |

  |                                                                                                                                                                            |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
  | TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag05_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_52_batch_8000_0.984833_IJBB_11 | 0.42814  | 0.912269 | 0.956962 | 0.970399 | 0.979357 | 0.987342 | 0.993758 |
  | TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag05_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_53_IJBB_11                     | 0.427167 | 0.911879 | 0.957352 | 0.970107 | 0.97926  | 0.987244 | 0.993767 |
  | TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_50_0.985500_IJBB_11              | 0.43223  | 0.918111 | 0.956767 | 0.969815 | 0.979065 | 0.988315 | 0.993454 |
  | TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_53_0.985000_IJBB_11              | 0.431646 | 0.924245 | 0.957157 | 0.97001  | 0.978384 | 0.988023 | 0.993332 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBB_11                                       | 0.427653 | 0.920643 | 0.958325 | 0.969718 | 0.980428 | 0.987634 | 0.993897 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBB_11                                        | 0.382765 | 0.920448 | 0.957157 | 0.969231 | 0.979649 | 0.987634 | 0.993294 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_51_batch_4000_0.984833_IJBB_11               | 0.435443 | 0.917819 | 0.956962 | 0.969231 | 0.979455 | 0.987731 | 0.993989 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBB_11                                        | 0.439825 | 0.921714 | 0.956865 | 0.969328 | 0.979357 | 0.987731 | 0.993974 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_52_batch_4000_0.984833_IJBB_11              | 0.430477 | 0.920156 | 0.958325 | 0.970107 | 0.980428 | 0.987537 | 0.993884 |

  |                                                                                                                                                                            |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_52_batch_8000_0.985000_IJBB_11                 | 0.449854 | 0.927167 | 0.957546 | 0.970789 | 0.979747 | 0.987926 | 0.993734 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBB_11                                          | 0.447322 | 0.92629  | 0.957449 | 0.970497 | 0.979942 | 0.988121 | 0.993725 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_045_08_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBB_11                                        | 0.416553 | 0.918695 | 0.956378 | 0.969328 | 0.979357 | 0.987731 | 0.993273 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_05_08_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_53_0.983667_IJBB_11                           | 0.404382 | 0.921811 | 0.957838 | 0.971568 | 0.979552 | 0.987731 | 0.993542 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_49_batch_8000_0.985167_IJBB_11                 | 0.432425 | 0.913729 | 0.954138 | 0.969912 | 0.979065 | 0.986855 | 0.993298 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBB_11                                          | 0.409348 | 0.914021 | 0.956378 | 0.970789 | 0.979552 | 0.987244 | 0.993408 |

  |                                                                                                                                                                            |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
  | TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag05_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_52_batch_8000_0.984833_IJBC_11 | 0.903615 | 0.953418 | 0.970752 | 0.980109 | 0.985939 | 0.991359 | 0.995349 |
  | TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag05_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_53_IJBC_11                     | 0.901672 | 0.953469 | 0.970803 | 0.980212 | 0.985836 | 0.991256 | 0.995312 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBC_11                                       | 0.919671 | 0.955617 | 0.971212 | 0.979956 | 0.986757 | 0.991972 | 0.995575 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_51_batch_4000_0.984833_IJBC_11               | 0.908677 | 0.954543 | 0.969934 | 0.979496 | 0.986092 | 0.991665 | 0.995347 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBC_11                                        | 0.90924  | 0.95439  | 0.969832 | 0.979394 | 0.986092 | 0.991717 | 0.99536  |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_52_batch_8000_0.985000_IJBC_11                 | 0.911132 | 0.955668 | 0.971161 | 0.980621 | 0.986041 | 0.992126 | 0.995711 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBC_11                                          | 0.908626 | 0.955566 | 0.971212 | 0.98057  | 0.986194 | 0.992177 | 0.995713 |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_05_08_35_emb512_adamw_bs512_ms1m_float16_basic_agedb_30_epoch_53_0.983667_IJBC_11                           | 0.916091 | 0.953827 | 0.970394 | 0.980416 | 0.986245 | 0.991717 | 0.99541  |
  | TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_05_1_35_emb512_adamw_bs512_ms1m_float16_basic_model_latest_IJBC_11                                          | 0.908524 | 0.952549 | 0.969423 | 0.980212 | 0.986092 | 0.991359 | 0.995494 |
  - **Effv2M MagFace**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 2
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="F, adamw cos16, mag 1_51_04_1_35", **pp)
  # pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="mag, margin (0.4, 0.8), FN (10, 110)", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_08_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="mag, margin (0.4, 0.8), FN (1, 51)", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_1_51_04_1_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="M, F, adamw cos16, mag 1_51_04_1_35", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="M, F, adamw cos16, mag 10_110_04_08", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_strides1_pw_1_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="M, F, pw -1, adamw cos16, mag 10_110_04_08", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_strides1_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs320_ms1m_float16_hist.json", fig_label="M, F, lr 0.01, adamw cos16, mag 10_110_04_08", **pp)
  ```
  - **Effv2B0 MagFace**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 10
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_warmup_3_float16_hist.json", fig_label="B0, wd 5e-4, SGD cos16", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_1e4_no_exc_bn_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 1e-4, adamw, no execlude bn cos16", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_1e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 1e-4, adamw, cos16", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_out_0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_hist.json", fig_label="B0 out 0, wd 5e-4, adamw, cos16", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag05_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.5", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag05_FN_1_51_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.5, FN_1_51", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag045_FN_1_51_RL_35_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.45, FN_1_51, RL 35", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag045_FN_1_51_RL_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.45, FN_1_51, RL 5", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag045_1_FN_1_51_RL_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.45_1, FN_1_51, RL 5", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag04_1_FN_1_51_RL_35_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.4_1, FN_1_51, RL 35", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag04_1_FN_1_51_RL_35_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_warm3_hist.json", fig_label="B0, l2 5e-4, sgd cos16, mag 0.4_1, FN_1_51, RL 35", **pp)
  ```
# Fine-tune
```py
hist_path = "checkpoints/"
pp = {}
pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
pp["epochs"] = [17]
pp["skip_epochs"] = 0
axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_arc_adamw_LA_lr0025_hist.json", fig_label="arc_adamw_LA_lr0025", **pp)
pp["axes"] = axes

axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_arc_adamw_lr00125_hist.json", fig_label="arc_adamw_lr00125", **pp)
axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_arc_adamw_lr0025_hist.json", fig_label="arc_adamw_lr0025", **pp)
# axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_arc_adamw_LA_hist.json", fig_label="arc_adamw_LA", **pp)

# axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_mag_1_51_045_1_35_adamw_LA_hist.json", fig_label="mag_1_51_045_1_35_adamw_LA", **pp)
# axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_mag_1_51_045_1_5_adamw_lr00125_hist.json", fig_label="mag_1_51_045_1_5_adamw_lr00125", **pp)
# axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_mag_1_51_045_1_5_adamw_LA_hist.json", fig_label="mag_1_51_045_1_5_adamw_LA", **pp)
axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_E53_mag_1_51_045_1_35_adamw_lr00125_hist.json", fig_label="mag_1_51_045_1_35_adamw_lr00125", **pp)
```
