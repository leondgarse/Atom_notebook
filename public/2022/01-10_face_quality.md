# ___2022 - 01 - 10 Face Quality___
***

# MagFace Face Quality test
  ```py
  from glob2 import glob
  imms = np.stack([(plt.imread(ii) - 127.5) * 0.0078125 for ii in glob('../face_flaw_aligned_112_112/*.jpg')])
  mm = keras.models.load_model('checkpoints/TT_effv2_s_strides1_pw512_F_bias_false_dr02_drc02_lr_01_wd5e2lr_mag_emb512_adamw_exclude_bn_bs512_ms1m_cos16_float16_basic_agedb_30_epoch_53_0.985000.h5')
  ees = mm(imms)
  scores = tf.norm(ees, axis=1).numpy()

  sys.path.append('../keras_cv_attention_models/')
  from keras_cv_attention_models import visualizing
  _ = visualizing.stack_and_plot_images(imms / 2 + 0.5, texts=scores)
  ```
***
# MagFace Training log
  - **MagFace and Curricular**
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
