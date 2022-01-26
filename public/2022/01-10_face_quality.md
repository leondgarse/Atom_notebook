# ___2022 - 01 - 10 Face Quality___
***

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
***
# MagFace Training log
  - **Effv2s MagFace and Curricular**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [3, 1, 3, 13, 33]
  pp["skip_epochs"] = 10
  names = ["Warmup"] + ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_dr02_drc02_lr_01_wd5e2lr_arc_emb512_adamw_exclude_bn_bs512_ms1m_float16_hist.json", fig_label="F, point_wise 512, wd 5e-4, dr 0.2, adamw cos16", names=names, **pp)
  pp["axes"] = axes

  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_curr_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_hist.json", fig_label="F, adamw cos16, curr", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_curr_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_2_hist.json", fig_label="F, adamw cos16, curr_2", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_hist.json", fig_label="F, adamw cos16, mag", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag05_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_hist.json", fig_label="F, adamw cos16, mag 0.5", **pp)
  ```
  - **Effv2b0 MagFace and Curricular**
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
