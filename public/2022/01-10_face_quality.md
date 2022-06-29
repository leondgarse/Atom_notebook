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

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_hist.json", fig_label="mag_10_110_04_08_35, VPL", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_hist.json", fig_label="ms1mv3_cleaned_asian, mag_10_110_04_08_35", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr03_drc03_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_hist.json", fig_label="ms1mv3_cleaned_asian, mag, dr 0.3, adamw 1e-4", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr03_drc03_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_2_hist.json", fig_label="ms1mv3_cleaned_asian, mag, dr 0.3, adamw 5e-4", **pp)
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

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_cos_49_hist.json", fig_label="adamw cos49, mag 10_110_04_08", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd1e2lr_no_exc_bn_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_hist.json", fig_label="adamw wd1e2lr_no_exc_bn, mag 10_110_04_08", **pp)
  ```
  - **Effv2B0 MagFace**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 10
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, arcface, wd 5e-4, adamw cos16", names=names, **pp)
  pp["axes"] = axes

  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_warmup_3_float16_hist.json", fig_label="B0, wd 5e-4, SGD cos16", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_1e4_no_exc_bn_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 1e-4, adamw, no execlude bn cos16", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_1e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 1e-4, adamw, cos16", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_out_0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_hist.json", fig_label="B0 out 0, wd 5e-4, adamw, cos16", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag05_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.5", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag05_FN_1_51_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.5, FN_1_51", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag045_FN_1_51_RL_35_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.45, FN_1_51, RL 35", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag045_FN_1_51_RL_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.45, FN_1_51, RL 5", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag045_1_FN_1_51_RL_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.45_1, FN_1_51, RL 5", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag04_1_FN_1_51_RL_35_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, wd 5e-4, adamw cos16, mag 0.4_1, FN_1_51, RL 35", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag04_1_FN_1_51_RL_35_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_warm3_hist.json", fig_label="B0, l2 5e-4, sgd cos16, mag 0.4_1, FN_1_51, RL 35", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_1_51_04_08_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cos16_batch_float16_hist.json", fig_label="B0, 10_110_04_08_35", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_10_110_04_08_35_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_hist.json", fig_label="B0, ms1mv3_cleaned_asian", **pp)
  ```
  - **EffV2B0 VPL**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 10
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, arcface, wd 5e-4, adamw cos16", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_hist.json", fig_label="B0, arcface, wd 5e-4, adamw cos16, VPL start 8000, delta 200", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_128_hist.json", fig_label="B0, arcface, wd 5e-4, adamw cos16, VPL start 2000, delta 50", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_8000_100_hist.json", fig_label="B0, arcface, wd 5e-4, adamw cos16, VPL start 8000, delta 100", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_10_110_04_08_35_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_VPL_hist.json", fig_label="B0, ms1mv3_cleaned_asian, mag_10_110_04_08_35, VPL start 1, delta 100", **pp)


  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_1_51_04_08_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cos16_batch_float16_hist.json", fig_label="B0, mag_10_110_04_08_35", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_10_110_04_08_35_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_hist.json", fig_label="B0, ms1mv3_cleaned_asian, mag_10_110_04_08_35", **pp)
  ```
  - **ms1mv3_cleaned_asian**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 10
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_hist.json", fig_label="B0, arcface", names=names, **pp)
  pp["axes"] = axes
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_1_51_04_08_5_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cos16_batch_float16_hist.json", fig_label="B0, magface 10_110_04_08_35", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_hist.json", fig_label="B0, ms1mv3_cleaned_asian, arcface", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_10_110_04_08_35_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_hist.json", fig_label="B0, ms1mv3_cleaned_asian, magface 10_110_04_08_35", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_10_110_04_08_35_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_200_cos16_batch_float16_hist.json", fig_label="B0, ms1mv3_cleaned_asian, randaug_200", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_randaug_cos16_batch_float16_vpl_1_200_hist.json", fig_label="B0, ms1mv3_cleaned_asian, arcface, vpl_1_200", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_mag_10_110_04_08_35_emb512_dr0_adamw_5e4_bs512_ms1mv3_cleaned_asian_randaug_cos16_batch_float16_VPL_hist.json", fig_label="B0, ms1mv3_cleaned_asian, magface 10_110_04_08_35, vpl_1_200", **pp)
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
# XLA issue
```py
class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, append_norm=False, **kwargs):
        super().__init__(**kwargs)
        self.units, self.append_norm = units, append_norm

    def build(self, input_shape):
        self.w = self.add_weight(name="norm_dense_w", shape=(input_shape[-1], self.units), trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1)
        output = tf.matmul(norm_inputs, norm_w)
        if self.append_norm:
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "append_norm": self.append_norm})
        return config

class NormDenseLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits        

    def call(self, y_true, y_pred):
        if y_pred.shape[-1] == y_true.shape[-1]:
            norm_logits = y_pred
            margin = 0.3
            regularizer_loss = 0.0
        else:
            norm_logits, feature_norm = y_pred[:, :-1], y_pred[:, -1] * -1
            margin = 0.04 * (feature_norm - 10) + 10.0
            regularizer_loss = feature_norm / 1e4 + 1.0 / feature_norm

        pick_cond = tf.where(y_true > 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta_valid = y_pred_vals - margin

        # tf.print(">>>>", norm_logits.shape, pick_cond, tf.reduce_sum(tf.cast(y_true > 0, "float32")), theta_valid.shape)
        logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid)
        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # logits = tf.where(tf.cast(y_true, dtype=tf.bool), theta_one_hot, norm_logits)
        # tf.print(">>>>", norm_logits.shape, logits.shape, y_true.shape)
        cls_loss = tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=self.from_logits)

        # tf.print(">>>>", cls_loss.shape, regularizer_loss.shape)
        return cls_loss + regularizer_loss * 35.0

    def get_config(self):
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config

xx = tf.random.uniform([160, 32, 32, 3])
yy = tf.one_hot(tf.cast(tf.random.uniform([160], 0, 10), 'int32'), 10)
mm = keras.models.Sequential([keras.layers.Input([32, 32, 3]), keras.layers.Flatten(), keras.layers.Dense(32), NormDense(10, append_norm=True)])
mm.compile(loss=NormDenseLoss(), optimizer="adam")
mm.fit(xx, yy)
```
# DepthwiseConv2D float16 issue
```py
keras.mixed_precision.set_global_policy("mixed_float16")

xx = tf.random.uniform([160, 32, 32, 3])
yy = tf.one_hot(tf.cast(tf.random.uniform([160], 0, 10), 'int32'), 10)
bb = keras.models.Sequential([
    keras.layers.Input([32, 32, 3]),
    keras.layers.Conv2D(32, 1, use_bias=False),
    keras.layers.DepthwiseConv2D(1, use_bias=False),
    keras.layers.Activation("swish"),
    keras.layers.DepthwiseConv2D(32, use_bias=False),
    # keras.layers.Flatten(),
    # keras.layers.Dense(32, dtype="float32"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, 1, use_bias=False),
    keras.layers.Flatten(dtype="float32"),
])
mm = keras.models.Model(bb.inputs[0], keras.layers.Dense(10)(bb.output))

def ds_gen():
    embs = tf.stop_gradient(bb(xx))
    embs = tf.nn.l2_normalize(embs, axis=-1)
    dists = tf.matmul(embs, embs, transpose_b=True)
    for ii, jj, dist in zip(xx, yy, dists):
        yield ii, jj - tf.reduce_min(dist)

output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32), tf.TensorSpec(shape=(10,), dtype=tf.int64))
ds = tf.data.Dataset.from_generator(ds_gen, output_signature=output_signature).repeat().batch(16).prefetch(buffer_size=-1)

mm.compile(loss="categorical_crossentropy", optimizer="adam")
mm.fit(ds, steps_per_epoch=10)
```
***
# VPL
- **Results**
| VPL                   | lfw          | cfp_fp       | agedb_30     | IJBB 1e-4    | IJBC 1e-4    |
| --------------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| False                 | 0.997667     | 0.979429     | 0.978333     | **0.941188** | 0.955719     |
| start 8000, delta 200 | 0.997667     | **0.979571** | **0.978500** | 0.938559     | 0.955054     |
| start 2000, delta 50  | **0.998000** | 0.976429     | 0.977667     | 0.940117     | **0.956128** |

- **IJBB / IJBC detail**
| VPL                         |        1e-06 |        1e-05 |       0.0001 |        0.001 |         0.01 |          0.1 |          AUC |
|:--------------------------- | ------------:| ------------:| ------------:| ------------:| ------------:| ------------:| ------------:|
| False, IJBB                 |     0.338948 |     0.875365 | **0.941188** |     0.960467 |     0.974684 |     0.983642 |     0.991774 |
| start 8000, delta 200, IJBB | **0.376241** |   **0.8815** |     0.938559 | **0.962902** | **0.976339** | **0.985881** | **0.992184** |
| start 2000, delta 50, IJBB  |     0.353944 |     0.874002 |     0.940117 |     0.961538 |     0.974684 |     0.983934 |     0.991567 |
| start 8000, delta 100, IJBB |     0.349562 |     0.886563 |     0.939727 |     0.962707 |     0.975365 |     0.984713 |     0.991944 |
|                             |              |              |              |              |              |              |              |
| False, IJBC                 |     0.848954 |     0.927954 |     0.955719 |     0.972184 |     0.982462 |     0.989109 |     0.994352 |
| start 8000, delta 200, IJBC | **0.877895** | **0.928568** |     0.955054 | **0.973513** | **0.983689** | **0.990387** | **0.994527** |
| start 2000, delta 50, IJBC  |     0.867004 |     0.926778 | **0.956128** |     0.972797 |     0.982257 |     0.989211 |     0.994179 |
| start 8000, delta 100, IJBC |     0.876361 |     0.931431 |     0.956282 |     0.973411 |     0.983433 |      0.99008 |     0.994522 |

- Distribution
```py
import evals
aa = keras.models.load_model('checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_basic_agedb_30_epoch_53.h5')

data_basic_path = '/datasets/ms1m-retinaface-t1'
eval_paths = [os.path.join(data_basic_path, ii) for ii in ['lfw.bin', 'cfp_fp.bin', 'agedb_30.bin']]
ee = '/datasets/ms1m-retinaface-t1/lfw.bin'
eea = evals.eval_callback(aa, ee, batch_size=16)
eea.on_epoch_end()

def plot_pos_neg_distribution(positive_dist=None, negative_dist=None, label="", density=True, ax=None):
    # plot the histogram for positive and negative distribution
    if ax is None:
        fig, ax = plt.subplots()
    if negative_dist is not None:
        _ = ax.hist(negative_dist, bins=512, density=density, alpha=0.5, label="negative, " + label)
    if positive_dist is not None:
        _ = ax.hist(positive_dist, bins=512, density=density, alpha=0.5, label="positive, " + label)
    ax.legend()
    plt.tight_layout()
    return ax

_ = plot_pos_neg_distribution(eea.tt, eea.ff)
```
```py
ijbb_labels = pd.read_csv('/datasets/IJB_release/IJBB/meta/ijbb_template_pair_label.txt', sep=" ", header=None).values[:, 2]
aa = np.load('IJB_result/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_basic_agedb_30_epoch_53_IJBB_11.npz')
ijbb_basic = aa['scores'][0]

aa = np.load('IJB_result/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_basic_model_latest_IJBB_11.npz')
ijbb_vpl_8000_200 = aa['scores'][0]

import seaborn as sns
pps = {
  "IJBB positive, basic": ijbb_basic[ijbb_labels == 1],
  "IJBB positive, vpl_8000_200": ijbb_vpl_8000_200[ijbb_labels == 1],
}
sns.histplot(pps, stat='density', kde=True)

nns = {
  "IJBB negative, basic": ijbb_basic[ijbb_labels == 0],
  "IJBB negative, vpl_8000_200": ijbb_vpl_8000_200[ijbb_labels == 0],
}
sns.histplot(nns, stat='density', kde=True)
```
```py
ijbc_labels = pd.read_csv('/datasets/IJB_release/IJBC/meta/ijbc_template_pair_label.txt', sep=" ", header=None).values[:, 2]
aa = np.load('IJB_result/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_2_basic_agedb_30_epoch_53_IJBC_11.npz')
ijbc_basic = aa['scores'][0]

aa = np.load('IJB_result/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_basic_model_latest_IJBC_11.npz')
ijbc_vpl_8000_200 = aa['scores'][0]

aa = np.load('IJB_result/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_128_basic_model_latest_IJBC_11.npz')
ijbc_vpl_2000_50 = aa['scores'][0]

aa = np.load('IJB_result/TT_efv2_b0_swish_GDC_arc_emb512_dr0_adamw_5e4_bs512_ms1mv3_randaug_cos16_batch_float16_vpl_8000_100_basic_model_latest_IJBC_11.npz')
ijbc_vpl_8000_100 = aa['scores'][0]

import seaborn as sns
pps = {
  "IJBC positive, basic": ijbc_basic[ijbc_labels == 1],
  "IJBC positive, vpl_8000_200": ijbc_vpl_8000_200[ijbc_labels == 1],
  "IJBC positive, vpl_8000_100": ijbc_vpl_8000_100[ijbc_labels == 1],
  "IJBC positive, vpl_2000_50": ijbc_vpl_2000_50[ijbc_labels == 1],
}
sns.histplot(pps, stat='density', kde=True)

nns = {
  "IJBC negative, basic": ijbc_basic[ijbc_labels == 0],
  "IJBC negative, vpl_8000_200": ijbc_vpl_8000_200[ijbc_labels == 0],
}
sns.histplot(nns, stat='density', kde=True)
```
# PartialFC
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 18]
  pp["skip_epochs"] = 3
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_glint360k_mag_bs_256_test_random_0_E25_hist.json", fig_label="effv2_s, glint360k, no partialFC, bs256", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_glint360k_mag_bs_256_test_random_0_partial_4_E25_hist.json", fig_label="effv2_s, glint360k, partialFC 4, bs256", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_glint360k_mag_bs_480_test_random_100_partial_4_E25_hist.json", fig_label="effv2_s, glint360k, partialFC 4, bs480", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_glint360k_cos_bs_480_test_random_100_partial_4_E25_hist.json", fig_label="effv2_s, glint360k, partialFC 4, bs480, cos", **pp)
  ```

  | Method       | lfw      | cfp_fp       | agedb_30 | IJBB         | IJBC         |
  | ------------ | -------- | ------------ | -------- | ------------ | ------------ |
  | No PartialFC | 0.998500 | 0.992286     | 0.983667 | **0.958909** | **0.971212** |
  | PartialFC 4  | 0.998167 | **0.993000** | 0.983833 | 0.956378     | 0.969218     |

  | Method             | 1e-06    | 1e-05    | 0.0001       | 0.001    | 0.01     | 0.1      | AUC      |
  | ------------------ | -------- | -------- | ------------ | -------- | -------- | -------- | -------- |
  | IJBB, No PartialFC | 0.439435 | 0.923856 | **0.958909** | 0.969231 | 0.978286 | 0.985589 | 0.992529 |
  | IJBC, No PartialFC | 0.89528  | 0.956691 | **0.971212** | 0.978933 | 0.985172 | 0.99008  | 0.994988 |
  | IJBB, PartialFC 4  | 0.404284 | 0.92483  | 0.956378     | 0.970204 | 0.97887  | 0.987634 | 0.993442 |
  | IJBC, PartialFC 4  | 0.889042 | 0.955003 | 0.969218     | 0.979649 | 0.985274 | 0.990745 | 0.994939 |
# Triplet offline mining
```py
# !waitGPU 0
import losses, train, models
import tensorflow_addons as tfa
keras.mixed_precision.set_global_policy("mixed_float16")

data_basic_path = '/datasets/ms1m-retinaface-t1'
data_path = data_basic_path + '_112x112_folders'
# data_path = '/datasets/ms1mv3_cleaned_asian'
eval_paths = [os.path.join(data_basic_path, ii) for ii in ['lfw.bin', 'cfp_fp.bin', 'agedb_30.bin']]

from keras_cv_attention_models import efficientnet
basic_model = None
model = 'checkpoints/TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16.h5'

samples_per_mining = 0.1
lr_decay_steps = int(16 / samples_per_mining / 3)  # 3 is because each sample contains [anchor, positive, negative]
epoch = int(17 / samples_per_mining / 3)
lr_base, lr_min = 0.0025, 1e-6
lr_decay = lr_min / lr_base  # Do not restart lr

tt = train.Train(data_path, eval_paths=eval_paths,
    save_path='TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_001_wd5e2lr_mag_10_110_04_08_35_emb512_adamw_bs512_ms1m_float16_E53_mining_01.h5',
    basic_model=basic_model, model=model, lr_base=lr_base, lr_decay=lr_decay, lr_decay_steps=lr_decay_steps, lr_min=lr_min, lr_warmup_steps=0,
    batch_size=480, random_status=100, eval_freq=4000, output_weight_decay=1, samples_per_mining=samples_per_mining)

optimizer = tfa.optimizers.AdamW(learning_rate=1e-2, weight_decay=5e-4, exclude_from_weight_decay=["/gamma", "/beta"])
sch = [
    {"loss": losses.MagFaceLoss(scale=64, min_feature_norm=10, max_feature_norm=110, min_margin=0.4, max_margin=0.8, regularizer_loss_lambda=35), "epoch": epoch},
]
tt.train(sch, 0)
exit()
```
# AdaFace
```py
hist_path = "checkpoints/"
pp = {}
pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
pp["epochs"] = [3, 17, 33]
pp["skip_epochs"] = 1
names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([64, 64, 64], [0.1, 0.1, 0.05])]
axes, _ = plot.hist_plot_split(hist_path + "TT_r50_max_pool_E_prelu_dr04_lr_01_l2_5e4_adaface_emb512_sgd_m09_bs512_ms1m_64_only_margin_SG_scale_true_bias_false_random_100_hist.json", fig_label="SGD l2 5e-4", names=names, **pp)
pp["axes"] = axes

axes, _ = plot.hist_plot_split(hist_path + "TT_r50_max_pool_E_prelu_dr04_lr_01_wd5e4lr_adaface_emb512_sgdw_m09_bs512_ms1m_64_only_margin_SG_scale_true_bias_false_random_100_hist.json", fig_label="SGDW 5e-6", **pp)
# axes, _ = plot.hist_plot_split(hist_path + "TT_r50_max_pool_E_prelu_dr04_lr_01_wd5e2lr_adaface_emb512_sgdw_m09_bs512_ms1m_64_only_margin_SG_scale_true_bias_false_random_100_hist.json", fig_label="SGDW 5e-4", **pp)

axes, _ = plot.hist_plot_split(hist_path + "TT_r100_max_pool_E_prelu_dr04_lr_01_l2_5e4_adaface_emb512_sgd_m09_bs512_ms1m_64_only_margin_SG_scale_true_bias_false_random_100_hist.json", fig_label="r100, SGD l2 5e-4", names=names, **pp)
```
