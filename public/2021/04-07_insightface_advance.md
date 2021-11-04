# ___2021 - 04 - 07 Keras Insightface Advance___
***

# 目录
***

# Q / A
  ```py
  """ Q:
  Called Trackable._track_trackable() with name='loss_scale', but a Trackable with this name is already declared as a dependency. Names must be unique (or overwrite=True).
  """
  """ A1:
  https://fixexception.com/tensorflow/called-trackable-track-trackable-with-name-s-but-a-trackable-with-this-name-is-already-declared-as-a-dependency-names-must-be-unique-or-overwrite-true/

  vi /opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/keras/mixed_precision/loss_scale_optimizer.py
  - 539       self._track_trackable(self._loss_scale, 'loss_scale')
  + 539       self._track_trackable(self._loss_scale, 'loss_scale', overwrite=True)
  - 550     self._track_trackable(FakeOptimizerForRestoration(self._optimizer), 'base_optimizer')
  + 550     self._track_trackable(FakeOptimizerForRestoration(self._optimizer), 'base_optimizer', overwrite=True)
  """
  ```
***

# Backbones
## BoTNet on MS1MV3
  ```py
  import json
  hist_path = "checkpoints/botnet/"
  pp = {}
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [1, 17, 17, 17, 20]
  pp["skip_epochs"] = 2
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([32, 64, 64, 64], [0.1, 0.1, 0.05, 0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json", fig_label="basic", names=names, **pp)

  pp["axes"] = axes
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_hist.json", fig_label="no bias", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_hist.json", fig_label="no bias, shortcut act none, tmul 2", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_randaug_hist.json", fig_label="no bias, shortcut act none, tmul 2, randaug", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_prelu_shortcut_act_none_GDC_arc_emb512_bs768_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json", fig_label="prelu, init 0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json", fig_label="swish, GDC", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json", fig_label="swish, E, use_bias True", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_true_conv_no_bias_tmul_2_random0_hist.json", fig_label="swish, E, use_bias False", **pp)

  TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json
  TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_hist.json
  TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_hist.json
  TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_randaug_hist.json
  TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json
  TT_botnet50_prelu_shortcut_act_none_GDC_arc_emb512_bs768_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json
  TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json
  TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_true_conv_no_bias_tmul_2_random0_hist.json


  from plot import choose_accuracy
  hist_path = "checkpoints/botnet/"
  aa = [
      hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json",
      hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_hist.json",
      hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_hist.json",
      hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_randaug_hist.json",
      hist_path + "TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json",
      hist_path + "TT_botnet50_prelu_shortcut_act_none_GDC_arc_emb512_bs768_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json",
      hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json",
      hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_true_conv_no_bias_tmul_2_random0_hist.json",
  ]
  _ = choose_accuracy(aa)
  ```
  |                                                                                                                            |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:---------------------------------------------------------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist              | 0.998    | 0.978    |   0.978167 |      33 |
  | TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_hist | 0.997833 | 0.981143 |   0.978833 |      53 |
  | TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_hist                | 0.9985   | 0.980286 |   0.979667 |      47 |
  | TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_randaug_hist        | 0.997667 | 0.981857 |   0.979333 |      46 |
  | TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist       | 0.9985   | 0.984571 |   0.979833 |      47 |
  | TT_botnet50_prelu_shortcut_act_none_GDC_arc_emb512_bs768_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist | 0.997833 | 0.978571 |   0.978    |      44 |
  | TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist   | 0.998    | 0.981571 |   0.979    |      16 |
  | TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_true_conv_no_bias_tmul_2_random0_hist    | 0.997667 | 0.983143 |   0.9785   |      15 |

  |                                                                                                                                                                           |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_basic_agedb_30_epoch_16_0.978167_IJBB_11                         | 0.360759 | 0.899318 | 0.941967 | 0.960273 | 0.972444 | 0.984129 |
  | TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_basic_agedb_30_epoch_50_batch_2000_0.979000_IJBB_11 | 0.313632 | 0.898832 | 0.941675 | 0.960759 | 0.971957 | 0.984031 |
  | TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_randaug_basic_agedb_30_epoch_47_0.979333_IJBB_11                   | 0.381694 | 0.894547 | 0.941967 | 0.963875 | 0.975657 | 0.984615 |
  | TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_basic_agedb_30_epoch_48_0.979667_IJBB_11                           | 0.384226 |  0.89591 | 0.940019 | 0.958325 | 0.973126 | 0.984323 |
  | TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_basic_agedb_30_epoch_45_batch_4000_0.980167_IJBB_11       | 0.349172 | 0.904284 | 0.944693 | 0.962707 | 0.974878 | 0.983739 |

  |                                                                                                                                                                     |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_basic_agedb_30_epoch_16_0.978167_IJBC_11                   | 0.880043 | 0.934499 | 0.955924 | 0.970241 | 0.980161 | 0.988597 |
  | TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_basic_agedb_30_epoch_48_0.979667_IJBC_11                     |   0.8894 | 0.933834 |  0.95577 | 0.970292 | 0.981439 | 0.988546 |
  | TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_basic_agedb_30_epoch_45_batch_4000_0.980167_IJBC_11 | 0.897735 | 0.936493 | 0.959145 | 0.973411 | 0.983075 | 0.989262 |
## GhostNet on MS1MV3
  ```py
  import json
  hist_path = "checkpoints/ghostnet/"
  pp = {}
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 10, 10, 20]
  pp["skip_epochs"] = 17
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.01, 0.001])]
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr04_wd5e4_bs512_ms1m_hist.json", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_pointwise_E_arc_emb512_dr04_wd5e4_bs512_ms1m_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json", **pp, fig_label="PReLU, cos16_batch")
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json", **pp)
  pp["axes"] = axes
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json", **pp, fig_label="PReLU, cos16_epoch")

  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json", **pp, limit_loss_max=80)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_hist.json", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_image_4_hist.json", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json", **pp, fig_label="PReLU, bias_false")

  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="swish, float16")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, float16")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_ipc10_float16_hist.json", **pp, fig_label="swish, float16, ipc10")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_randaug_float16_hist.json", **pp, fig_label="swish, float16, randaug")

  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, init 0.25, float16")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_kam0_float16_hist.json", **pp, fig_label="swish, float16, keep_as_min 0")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_LH_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_float16_hist.json", **pp, fig_label="swish, float16, SGD look ahead")

  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, float16, strides_1")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, float16, strides_1, se_relu")
  ```
  ```py
  from plot import choose_accuracy
  hist_path = "checkpoints/ghostnet/"
  aa = [
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr04_wd5e4_bs512_ms1m_hist.json",
      hist_path + "TT_ghostnet_pointwise_E_arc_emb512_dr04_wd5e4_bs512_ms1m_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_image_4_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_ipc10_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_randaug_float16_hist.json",
      hist_path + "TT_ghostnet_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_kam0_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_LH_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_float16_hist.json",
      hist_path + "TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json",
      hist_path + "TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_ghostnet_prelu_GDC_arc_emb512_dr0_"))
  ```
  |                                                                                                     |      lfw |   cfp_fp | agedb_30 | epoch |
  |:--------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| -----:|
  | _wd5e4_bs512_ms1m_hist                                                                              | 0.995333 | 0.957714 |    0.956 |    47 |
  | 04_wd5e4_bs512_ms1m_hist                                                                            | 0.995833 | 0.953571 |    0.959 |    45 |
  | sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist                                          | 0.997167 | 0.959429 | 0.969333 |    45 |
  | sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist                                            | 0.996167 | 0.961286 | 0.966833 |    46 |
  | sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist                                                 |   0.9965 |    0.965 |     0.97 |    48 |
  | sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist                                                 | 0.996833 | 0.962429 |    0.969 |    53 |
  | sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_hist                                                 | 0.997167 | 0.959857 | 0.968667 |    48 |
  | sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_image_4_hist                                         | 0.996333 | 0.959714 |    0.968 |    47 |
  | sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist                                                  |   0.9965 | 0.957857 |    0.966 |    45 |
  | sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist                                                | 0.996667 | 0.960429 | 0.968667 |    45 |
  | prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist         | 0.995333 | 0.957714 |    0.969 |    45 |
  | prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist      | 0.996667 | 0.960429 | 0.966833 |    49 |
  | swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist         | 0.996833 | 0.959857 |    0.969 |    44 |
  | swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_ipc10_float16_hist   |   0.9955 | 0.961857 | 0.971167 |    48 |
  | swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_randaug_float16_hist |   0.9965 |    0.958 | 0.960667 |    45 |
  | swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_kam0_float16_hist    |   0.9965 |    0.962 | 0.968333 |    47 |
  | sgd_LH_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_float16_hist                                     | 0.995833 | 0.948429 | 0.956167 |    15 |
  | c_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist                     | 0.997833 | 0.978286 | 0.98     |    46 |
  | u_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist             | 0.9975   | 0.978429 | 0.976333 |    47 |

  |                                                                                                                                                                            |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | GhostNet_x1.3_Arcface_Epoch_24_IJBB                                                                                                                                        | 0.352678 | 0.881694 | 0.928724 | 0.954041 | 0.972055 | 0.985784 |
  | TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_basic_agedb_30_epoch_15_0.970000_IJBB_11                         | 0.344888 | 0.864752 | 0.920156 | 0.951315 | 0.969912 | 0.984129 |
  | TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_E48_arc_trip_basic_agedb_30_epoch_15_batch_4000_0.971500_IJBB_11 |  0.39075 | 0.822785 | 0.911879 | 0.951607 | 0.974586 | 0.988413 |
  | TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_49_batch_2000_0.970000_IJBB_11                     | 0.337001 | 0.856573 | 0.922006 |  0.95297 |  0.97147 | 0.983544 |
  | TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_LH_sgd_lr25e3_float16_basic_agedb_30_epoch_14_batch_4000_0.971000_IJBB_11         | 0.342162 | 0.863778 | 0.923466 | 0.953749 | 0.970886 |  0.98296 |
  | TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_sgd_lr25e3_float16_basic_agedb_30_epoch_14_batch_2000_0.971000_IJBB_11            | 0.346641 | 0.857157 | 0.922687 | 0.954041 | 0.970691 | 0.983155 |
  | TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_47_0.980000_IJBB_11                   | 0.360467 | 0.879065 | 0.931159 | 0.957644 |  0.97186 | 0.985102 |
  | TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_50_0.976333_IJBB_11           | 0.327069 | 0.887244 | 0.932425 |  0.95482 | 0.972541 |  0.98481 |

  |                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | GhostNet_x1.3_Arcface_Epoch_24_IJBC_11                                                                                                                           | 0.876259 | 0.922023 | 0.945748 |  0.96477 | 0.978985 | 0.990336 |
  | TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_47_0.980000_IJBC_11         | 0.873038 | 0.921563 |  0.94943 | 0.967684 | 0.979189 | 0.989569 |
  | TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_50_0.976333_IJBC_11 | 0.872833 | 0.922892 | 0.949328 | 0.966457 | 0.980263 | 0.989262 |
## Finetune GhostNet on MS1MV3 with SAM or LH
  ```py
  import json
  hist_path = "checkpoints/ghostnet/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [20]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([64], [0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_sgd_lr25e3_float16_hist.json", **pp, names=names, fig_label="swish, sgd")
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_LH_sgd_lr25e3_float16_hist.json", **pp, fig_label="swish, LookAhead")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_SAM_sgd_lr25e3_float16_hist.json", **pp, fig_label="swish, SAM")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_SAM_LH_sgd_lr25e3_float16_hist.json", **pp, fig_label="swish, SAM, LookAhead")
  ```
  ```py
  hist_path = "checkpoints/ghostnet/"
  aa = [
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_sgd_lr25e3_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_LH_sgd_lr25e3_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_SAM_sgd_lr25e3_float16_hist.json",
      hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_SAM_LH_sgd_lr25e3_float16_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_"))
  ```
  |                                |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:-------------------------------|---------:|---------:|-----------:|--------:|
  | sgd_lr25e3_float16_hist        | 0.997667 | 0.962571 |   0.9705   |      13 |
  | LH_sgd_lr25e3_float16_hist     | 0.997833 | 0.964857 |   0.970833 |      15 |
  | SAM_sgd_lr25e3_float16_hist    | 0.997833 | 0.963571 |   0.97     |      14 |
  | SAM_LH_sgd_lr25e3_float16_hist | 0.997667 | 0.963857 |   0.970667 |      13 |
## Mobilenet ArcFace and CurricularFace on Emore
  ```py
  import json
  hist_path = "checkpoints/mobilenet_emore_tests/"
  pp = {}
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  # pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 10, 10, 80]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.01, 0.001])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json", **pp, names=names, fig_label="curricular, dr04, sgdw, scale_true_bias_true, Constant lr decay, E50")
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_batch_hist.json", **pp, fig_label="curricular, dr04, sgdw, scale_true_bias_true, cos30 batch, tmul 1")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_hist.json", **pp, fig_label="curricular, dr04, sgdw, scale_true_bias_true, cos30 epoch, tmul 1")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_hist.json", **pp, fig_label="curricular, dr04, sgd, scale_true_bias_true, cos7 epoch, tmul 2")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_hist.json", **pp, fig_label="curricular, dr0, sgd, scale_true_bias_true, cos7 epoch, tmul 2")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json", **pp, fig_label="curricular, dr0, sgd, scale_true_bias_true, cos7 batch, tmul 2")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json", **pp, fig_label="arc, dr04, sgdw, scale_true_bias_true, Constant lr decay, E50")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_hist.json", **pp, fig_label="arc, dr04, sgdw, scale_false_bias_true, Constant lr decay, E50")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json", **pp, fig_label="arc, dr0, sgd, scale_true_bias_true, cos7 batch, tmul 2")
  ```
  ```py
  hist_path = "checkpoints/mobilenet_emore_tests/"
  aa = [
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_mobilenet_GDC_emb512_"))
  ```
  |                                                                              |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:-----------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_batch_hist | 0.9955   | 0.904571 |   0.956    |      77 |
  | curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_hist       | 0.995    | 0.892857 |   0.950167 |      66 |
  | curricular_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_hist        | 0.993167 | 0.834286 |   0.935167 |      55 |
  | curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_hist         | 0.996    | 0.898143 |   0.952833 |      57 |
  | curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist   | 0.996333 | 0.895714 |   0.948667 |      47 |
  | curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist             | 0.993    | 0.871286 |   0.9415   |      49 |
  | arc_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist                    | 0.992833 | 0.914714 |   0.9375   |      49 |
  | arc_dr04_wd5e4_bs512_emore_sgdw_hist                                         | 0.9945   | 0.916714 |   0.935167 |      45 |
  | arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist          | 0.996167 | 0.927286 |   0.951333 |      45 |
## Mobilenet randaug
  ```py
  import json
  hist_path = "checkpoints/mobilenet_emore_tests/"
  pp = {}
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  pp["epochs"] = [5, 5, 7, 33, 80]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.05, 0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json", **pp, names=names, fig_label="scale_true_bias_false, random_0")
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_false_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json", **pp, fig_label="scale_false_bias_true")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json", **pp, fig_label="scale_true_bias_true, random_0")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_batch_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist.json", **pp, fig_label="scale_true_bias_true, random_0, l2 apply_to_batch_normal=True")

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_2_hist.json", **pp, fig_label="scale_true_bias_false, randaug_100")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_1_hist.json", **pp, fig_label="scale_true_bias_false, random_0, tmul_1")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_1_hist.json", **pp, fig_label="scale_true_bias_false, randaug_100, tmul_1")

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json", **pp, fig_label="scale_true_bias_false, random_0, ipc_10")

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_randaug_lrmin_1e4_tmul_2_ipc_10_hist.json", **pp, fig_label="scale_true_bias_true, randaug, ipc_10")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json", **pp, fig_label="scale_true_bias_true, random_0, ipc_10")
  ```
  ```py
  hist_path = "checkpoints/mobilenet_emore_tests/"
  pp = {}
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "lr"]
  pp["epochs"] = [30]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([64], [0.0125])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_E51_randaug_100_hist.json", **pp, names=names, fig_label="scale_true_bias_false, random_0, E51_randaug_100")
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_E51_random_0_hist.json", **pp, fig_label="scale_true_bias_false, randaug_100, E51_random_0")
  ```
  ```py
  hist_path = "checkpoints/mobilenet_emore_tests/"
  aa = [
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_false_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_batch_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_2_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_1_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_1_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_randaug_lrmin_1e4_tmul_2_ipc_10_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_E51_randaug_100_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_E51_random_0_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_"))
  ```
  |                                                                                           |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:------------------------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_hist          | 0.996667 | 0.911286 |   0.960667 |      46 |
  | bs512_emore_sgd_scale_false_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist          | 0.997    | 0.916429 |   0.957667 |      46 |
  | bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist           | 0.996167 | 0.916571 |   0.959167 |      49 |
  | batch_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist                               | 0.995667 | 0.925857 |   0.951333 |      46 |
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_2_hist       | 0.996333 | 0.940429 |   0.9515   |      47 |
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_1_hist          | 0.9965   | 0.913286 |   0.960333 |      50 |
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_1_hist       | 0.995833 | 0.938429 |   0.947833 |      49 |
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist   | 0.996167 | 0.913429 |   0.958167 |      46 |
  | bs512_emore_sgd_scale_true_bias_true_cos16_batch_randaug_lrmin_1e4_tmul_2_ipc_10_hist     | 0.995    | 0.941857 |   0.953833 |      47 |
  | bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist    | 0.995833 | 0.913143 |   0.957833 |      47 |
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_E51_randaug_100_hist | 0.996167 | 0.932714 |   0.954333 |      16 |
  | bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_E51_random_0_hist | 0.9955   | 0.936286 |   0.955833 |      12 |
## Finetune Mobilenet
  ```py
  hist_path = "checkpoints/mobilenet_emore_tests/"
  pp = {}
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "lr"]
  pp["epochs"] = [80]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([64], [0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_randaug_300_hist.json", names=names, fig_label="arc_trip_64_randaug_300_hist", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_randaug_300_hist.json", fig_label="arc_randaug_300_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_random2_E16_arc_hist.json", fig_label="arc_trip_64_random2_E16_arc_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_E16_arc_hist.json", fig_label="arc_trip_64_E16_arc_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_random3_hist.json", fig_label="arc_trip_64_random3_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_random2_hist.json", fig_label="arc_trip_64_random2_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_a25_hist.json", fig_label="arc_trip_64_a25_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_hist.json", fig_label="arc_trip_64_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_hist.json", fig_label="arc_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_hist.json", fig_label="arc_trip_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_curricular_trip_hist.json", fig_label="curricular_trip_hist", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_curricular_hist.json", fig_label="curricular_hist", **pp)
  ```
## MobileNet swish PReLU
  ```py
  import json
  hist_path = "checkpoints/mobilenet_emore_tests/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="basic relu scale_true_bias_false", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="relu scale_true_bias_true", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_prelu_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="PReLU, scale_true_bias_false", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="swish GDC", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_E_emb512_arc_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="swish E", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GAP_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="swish GAP", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist.json", fig_label="swish ms1m, GDC, dr0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GDC_arc_emb512_dr4_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist.json", fig_label="swish ms1m, GDC, dr4", **pp)
  ```
  ```py
  from plot import choose_accuracy
  hist_path = "checkpoints/mobilenet_emore_tests/"
  aa = [
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json",
      hist_path + "TT_mobilenet_prelu_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json",
      hist_path + "TT_mobilenet_swish_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json",
      hist_path + "TT_mobilenet_swish_E_emb512_arc_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json",
      hist_path + "TT_mobilenet_swish_GAP_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json",
      hist_path + "TT_mobilenet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist.json",
      hist_path + "TT_mobilenet_swish_GDC_arc_emb512_dr4_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_mobilenet_"))
  ```
  |                                                                                                                          |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:-------------------------------------------------------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist       | 0.996    | 0.916    |   0.962167 |      49 |
  | GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist        | 0.995833 | 0.916714 |   0.9575   |      48 |
  | prelu_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist | 0.997    | 0.919429 |   0.965167 |      49 |
  | swish_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist | 0.997    | 0.928286 |   0.964333 |      48 |
  | swish_E_emb512_arc_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist  | 0.996333 | 0.924286 |   0.961    |      49 |
  | swish_GAP_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist | 0.996167 | 0.918143 |   0.959833 |      48 |
  | swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist                               | 0.996833 | 0.960429 |   0.967167 |      46 |
  | swish_GDC_arc_emb512_dr4_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist                               | 0.995    | 0.953286 |   0.942667 |      15 |
## Resnet
  ```py
  import json
  hist_path = "checkpoints/resnetv2_50_101/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 3
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", fig_label="resnet101v2 1 warmup", names=names, **pp)
  pp["axes"] = axes

  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet101v2 conv_no_bias, relu", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet101v2 swish, E", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_hist.json", fig_label="resnet101v2, swish, first conv 3, strides 1", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet50v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet50v2 relu, basic", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet50v2_swish_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet50v2 swish", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet50v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet50v2 swish, E", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_hist.json", fig_label="resnet50v2, swish, first conv 3, strides 1", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_hist.json", fig_label="r50, swish", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json", fig_label="r50, swish, cleaned", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json", fig_label="r50, swish, cleaned, SD", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json", fig_label="se_r50, swish, cleaned", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json", fig_label="se_r50, swish, cleaned, SD", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_randaug_100_bnm09_bne1e4_cos16_hist.json", fig_label="se_r50, swish, cleaned, randaug 100", **pp)

  hist_path = "checkpoints/"
  axes, _ = plot.hist_plot_split(hist_path + "TT_resnest50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnest50, swish, randaug 100", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_resnest50_se_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnest50, se, swish, randaug 100", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="se_r50, swish, randaug 100, SD", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="se_r50, swish, randaug 100, SD, bs1024", **pp)
  ```
  ```py
  from plot import choose_accuracy
  hist_path = "checkpoints/resnetv2_50_101/"
  aa = [
      hist_path + "TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json",
      hist_path + "TT_resnet101v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json",
      hist_path + "TT_resnet101v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json",
      hist_path + "TT_resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_resnet50v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json",
      hist_path + "TT_resnet50v2_swish_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json",
      hist_path + "TT_resnet50v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json",
      hist_path + "TT_resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_randaug_100_bnm09_bne1e4_cos16_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_"))
  ```
  | Datasets          | backbone                                                                              | IJBC(1e-05) | IJBC(1e-04) | agedb30 | cfp_fp  | lfw   |
  | ----------------- | ------------------------------------------------------------------------------------- | ----------- | ----------- | ------- | ------- | ----- |
  | MS1MV3-Arcface    | r50-fp16                                                                              | 94.79       | 96.46       | 98.35   | 98.96   | 99.83 |
  | Glint360k-Cosface | r50-fp16-0.1                                                                          | 95.61       | 96.97       | 98.38   | 99.20   | 99.83 |
  | MS1MV3-Arcface    | se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist | 95.0759     | 96.6252     | 98.4    | 98.9714 | 99.83 |
  | MS1MV3-Arcface    | r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist    | 94.6106     | 96.4463     | 98.4333 | 98.9571 | 99.83 |

  |                                                                                                                                |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:-------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist                     | 0.998167 | 0.984    |   0.9785   |      46 |
  | resnet101v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist                    | 0.997833 | 0.983429 |   0.979167 |      46 |
  | resnet101v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist               | 0.998167 | 0.982714 |   0.980333 |      43 |
  | resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_hist | 0.9985   | 0.989143 |   0.9845   |      46 |
  | resnet50v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist                     | 0.998    | 0.980143 |   0.976667 |      46 |
  | resnet50v2_swish_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist               | 0.997333 | 0.977286 |   0.977    |      13 |
  | resnet50v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist                | 0.998    | 0.979    |   0.977667 |      43 |
  | resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_hist  | 0.9985   | 0.988571 |   0.9835   |      47 |
  | r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_hist                                                     | 0.998333 | 0.989714 |   0.984167 |      45 |
  | r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist                                             | 0.998333 | 0.989571 |   0.984333 |      47 |
  | r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist                                          | 0.9985   | 0.989714 |   0.983667 |      49 |
  | se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_hist                                          | 0.998333 | 0.989714 |   0.984    |      15 |

  |                                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_46_batch_2000_0.978833_IJBB_11                     | 0.395618 | 0.897955 | 0.943622 | 0.962707 | 0.973515 | 0.983544 |
  | TT_resnet101v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_basic_agedb_30_epoch_46_batch_2000_0.979667_IJBB_11                    | 0.346056 | 0.900487 | 0.944693 | 0.963291 | 0.973515 | 0.985005 |
  | TT_resnet50v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_basic_agedb_30_epoch_48_batch_2000_0.976667_IJBB_11                     | 0.284713 | 0.890068 | 0.941675 | 0.961149 | 0.973126 | 0.983057 |
  | TT_resnet50v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_IJBB_11                                                            | 0.278968 | 0.891431 | 0.940409 | 0.959786 | 0.972055 | 0.982084 |
  | TT_resnet101v2_swish_pad_same_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_basic_agedb_30_epoch_44_0.980333_IJBB_11                          | 0.315482 | 0.908471 | 0.945278 | 0.961733 | 0.973807 | 0.984226 |
  | TT_resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_6000_0.983667_IJBB_11  |  0.40224 | 0.916943 | 0.949951 |  0.96446 | 0.976728 |  0.98666 |
  | TT_resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_44_batch_2000_0.985000_IJBB_11 | 0.397371 | 0.914606 | 0.952483 | 0.967381 | 0.978773 | 0.987439 |
  | TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333_IJBB_11                                                        | 0.385589 | 0.915871 | 0.950828 | 0.965141 | 0.976923 | 0.985784 |
  | TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_46_batch_4000_0.984167_IJBB_11                                                     |  0.38705 | 0.908471 | 0.949951 | 0.965239 | 0.976728 | 0.985492 |
  | TT_se_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_16_batch_4000_0.984167_IJBB_11                                          | 0.388608 | 0.914411 | 0.950536 | 0.965336 | 0.977799 | 0.986855 |
  | TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_4000_0.984500_IJBB_11                                       | 0.390944 | 0.924537 | 0.954333 | 0.967868 | 0.979747 | 0.987244 |

  |                                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | TT_resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_6000_0.983667_IJBC_11  | 0.909853 | 0.946106 | 0.963696 | 0.974383 | 0.983842 | 0.990694 |
  | TT_resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_44_batch_2000_0.985000_IJBC_11 | 0.900138 | 0.948816 | 0.966406 | 0.977144 | 0.985172 |  0.99187 |
  | TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333_IJBC_11                                                        | 0.896712 | 0.946106 | 0.964463 |  0.97607 | 0.984251 | 0.990131 |
  | TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_46_batch_4000_0.984167_IJBC_11                                                     | 0.894718 | 0.945799 | 0.963798 | 0.975814 | 0.984456 | 0.990387 |
  | TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_4000_0.984500_IJBC_11                                       | 0.907757 | 0.950759 | 0.966252 | 0.976684 | 0.985172 | 0.990489 |
## AotNet
  TT_basic_aotnet50_14_expansion1_preact_true_strides2_se_SD_swish_E_arc_emb512_dr02_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50_14_expansion1_preact_true_strides2_se_DC_swish_E_arc_emb512_dr02_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50_14_expansion1_preact_true_strides2_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50v2_14_expansion1_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50_expansion1_preact_true_strides2_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50v2_expansion1_preact_false_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50v2_expansion1_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50v2_expansion1_se_SD_relu_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50v2_expansion1_SD_relu_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_
  TT_basic_aotnet50_SD_relu_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_
  ```py
  import json
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 8
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="se_r50, swish, randaug 100, SD, bs1024", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="se_r50, swish, randaug 100, SD, bs512", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_se_r50_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="se_r50, swish, randaug 100, SD, bs1024, dr 0", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50_SD_relu_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_float16_hist.json", fig_label="aotnet, bs512, relu", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50v2_expansion1_SD_relu_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, v2_expansion1_SD_relu", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50v2_expansion1_se_SD_relu_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, v2_expansion1_se_SD_relu", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50v2_expansion1_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, v2_expansion1_se_SD_swish, dr0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50v2_expansion1_preact_false_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, v2_expansion1_preact_false_se_SD_swish, dr0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50_expansion1_preact_true_strides2_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, expansion1_preact_true_strides2_se_SD_swish, dr0", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50v2_14_expansion1_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, v2_14_expansion1_se_SD_swish, dr0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50_14_expansion1_preact_true_strides2_se_SD_swish_E_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, 14_expansion1_preact_true_strides2_se_SD_swish, dr0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50_14_expansion1_preact_true_strides2_se_DC_swish_E_arc_emb512_dr02_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, 14_expansion1_preact_true_strides2_se_DC_swish, dr02", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_basic_aotnet50_14_expansion1_preact_true_strides2_se_SD_swish_E_arc_emb512_dr02_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="aotnet, bs512, 14_expansion1_preact_true_strides2_se_SD_swish, dr02", **pp)
  ```
## EfficientNetV2B0
  ```sh
  ./IJB_evals.py -P IJB_result/*efv2*IJBB* /datasets/IJB_release/IJBB/meta/ijbb_template_pair_label.txt
  ```
  ```py
  hist_path = "checkpoints/ghostnet/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 3
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", fig_label="ghostnet swish, float16", **pp)
  pp["axes"] = axes

  hist_path = "checkpoints/efficientnet_v2/"

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgdw_wd_5e4_bs512_ms1m_cos16_batch_float16_hist.json", fig_label="ebv2_b0, sgdw 5e-4", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_batch_float16_hist.json", fig_label="ebv2_b0, SGD, l2 5e-4, bnm 0.99, bne 1e-3", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="ebv2_b0, SGD l2 5e-4, bnm 0.9, bne 1e-4", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="ebv2_b0, SGD l2 5e-4, bnm 0.9, bne 1e-4, ms1m_cleaned", **pp)
  ```
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[1:] + ['lr', 'triplet_embedding_loss']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 13
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout", **pp)
  pp["axes"] = axes

  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random_2, cleaned", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_100_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="randaug_100, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="randaug_cutout, cleaned", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_M3_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random_2, arc_M3, cleaned", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_M4_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random_2, arc_M4, cleaned", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_r0_mixup_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random_0, mixup, cleaned", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nesterov_l2_5e4_bs512_ms1m_cleaned_random_0_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random_0, nesterov, cleaned", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_cleaned_random_0_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random_0, lookahead, cleaned", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc_M5, randaug_cutout, cleaned", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc_M5, randaug_cutout, lookahead, cleaned", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc_M5, randaug_cutout", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc_M5, randaug_cutout, lookahead", **pp)

  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, lookahead", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, lookahead, ipc 10", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_lrmin_1e5_float16_hist.json", fig_label="arc, randaug_cutout, lrmin 1e-5", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_epoch_17_float16_hist.json", fig_label="arc, randaug_cutout, decay on epoch", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, ipc 10", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nestrov_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, ipc 10, nestrov", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_mmul_1_float16_hist.json", fig_label="arc, randaug_cutout, mmul_1", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_cos_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, cosface", **pp)

  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_warmup_2_float16_hist.json", fig_label="arc, randaug_cutout, warmup_2", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, bs1024", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_lrbase_2_cos16_batch_float16_hist.json", fig_label="arc, randaug_cutout, bs1024, lrbase 0.2", **pp)

  axes, _ = plot.hist_plot_split("checkpoints/efficientnet_v2/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="random 0, cleaned", **pp)
  ```
  ```py
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr', 'triplet_embedding_loss']
  pp["epochs"] = [17]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_hist.json", fig_label="arc, randaug_cutout, lr_base 0.025", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_lrbase_1e3_hist.json", fig_label="arc, randaug_cutout, lr_base 1e-3", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_hist.json", fig_label="arc, randaug_cutout, lr_base 0.025, SGD lookahead", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_SAM_hist.json", fig_label="arc, randaug_cutout, lr_base 0.025, SGD lookahead, SAM", **pp)
  ```
  ```py
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr', 'triplet_embedding_loss']
  pp["epochs"] = [17]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_t_GDC_arc_emb512_dr0_sgd_l2_5e4_bs256_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="bs2562, lr_base 0.05", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_t_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json", fig_label="bs512, lr_base 0.05", **pp)
  ```
  ```py
  from plot import choose_accuracy
  hist_path = "checkpoints/efficientnet_v2/"  
  aa = [
      hist_path + "TT_early_efv2_s_add_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_hist.json",
      hist_path + "TT_early_efv2_s_sd_1_08_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_cos16_hist.json",
      hist_path + "TT_early_efv2_s_sd08_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_cos16_hist.json",
      hist_path + "TT_efv2_s_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgdw_wd_5e4_bs512_ms1m_cos16_batch_float16_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_batch_float16_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_100_bnm09_bne1e4_cos16_batch_float16_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_M4_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_M3_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_r0_mixup_bnm09_bne1e4_cos16_batch_float16_hist.json",
  ]
  _ = choose_accuracy(aa)
  ```
  ```py
  from glob2 import glob
  for src_name in glob('TT_ebv2_b0_swish_GDC_arc_*'):
      dst_name = src_name.replace('TT_ebv2_b0_', 'TT_efv2_b0_')
      print(f">>>> {src_name = }, {dst_name = }")
      aa = np.load(src_name)
      scores, names = aa.get("scores", []), aa.get("names", [])
      names = [ii.replace('TT_ebv2_b0_', 'TT_efv2_b0_') for ii in names]
      np.savez(src_name, scores=scores, names=names)
      os.rename(src_name, dst_name)
  ```
  |                                                                                                                                                      |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | TT_early_efv2_s_add_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_hist                                                                              | 0.997833 | 0.960429 |   0.9785   |      16 |
  | TT_early_efv2_s_sd_1_08_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_cos16_hist                                                                         | 0.998167 | 0.974857 |   0.982167 |      47 |
  | TT_early_efv2_s_sd08_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_cos16_hist                                                                            | 0.997667 | 0.981    |   0.981    |      48 |
  | TT_efv2_s_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_hist                                                                      | 0.997333 | 0.975571 |   0.977833 |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgdw_wd_5e4_bs512_ms1m_cos16_batch_float16_hist                                                                  | 0.995833 | 0.953429 |   0.964    |      48 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_batch_float16_hist                                                                   | 0.997167 | 0.974857 |   0.976    |      48 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_hist                                                      | 0.997167 | 0.975429 |   0.975    |      47 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_hist                                              | 0.9975   | 0.975714 |   0.976333 |      49 |
  | TT_efv2_b0_swish_GDC_arc_M4_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist                                  | 0.997333 | 0.972429 |   0.976333 |      48 |
  | TT_efv2_b0_swish_GDC_arc_M3_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_hist                                  | 0.996833 | 0.973571 |   0.974333 |      47 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_r0_mixup_bnm09_bne1e4_cos16_batch_float16_hist                                     | 0.997833 | 0.973571 |   0.974333 |      46 |

  ```sh
  ./IJB_evals.py -P IJB_result/TT_efv2_*epoch_5*IJBB* IJB_result/TT_efv2_*epoch_4*IJBB* /datasets/IJB_release/IJBB/meta/ijbb_template_pair_label.txt
  ./IJB_evals.py -P IJB_result/TT_efv2_*E50*IJBB* /datasets/IJB_release/IJBB/meta/ijbb_template_pair_label.txt
  ./IJB_evals.py -P IJB_result/TT_efv2_*IJBC* /datasets/IJB_release/IJBC/meta/ijbc_template_pair_label.txt
  ```

  |                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:| --------:|
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_batch_float16_basic_agedb_30_epoch_49_0.976000_IJBB_11                                           | 0.371373 | 0.880039 |  0.93593 |  0.96037 | 0.974294 | 0.984323 |          |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.975000_IJBB_11                              | 0.369133 | 0.871081 | 0.937098 | 0.959104 | 0.973905 | 0.984518 | 0.992391 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976333_IJBB_11                      | 0.351899 | 0.880234 |  0.93408 | 0.958715 | 0.973612 | 0.984907 | 0.991727 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nesterov_l2_5e4_bs512_ms1m_cleaned_random_0_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976833_IJBB_11    |  0.38296 | 0.870399 | 0.932911 |  0.95852 | 0.973612 | 0.983155 | 0.991932 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_cleaned_random_0_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.977167_IJBB_11          |  0.33369 | 0.885005 |  0.93369 | 0.956378 | 0.972639 | 0.983544 | 0.991878 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_r0_mixup_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.974333_IJBB_11             | 0.350828 | 0.757254 | 0.913729 | 0.957157 | 0.973807 | 0.985297 | 0.992829 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_100_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_48_0.976833_IJBB_11          |  0.37147 | 0.883934 | 0.935443 | 0.957254 | 0.972833 | 0.984323 | 0.991655 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976667_IJBB_11       | 0.352288 | 0.871276 | 0.937293 | 0.959981 | 0.975268 |  0.98481 | 0.991726 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.977333_IJBB_11             | 0.385005 | 0.883934 | 0.934372 | 0.958617 | 0.973515 | 0.983836 | 0.991745 |
  | TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.975500_IJBB_11    | 0.380428 | 0.868744 | 0.931159 | 0.958423 | 0.974391 | 0.986173 | 0.993756 |
  | TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976333_IJBB_11 | 0.382278 | 0.860954 | 0.932035 | 0.958715 | 0.975365 |  0.98666 | 0.994307 |
  | TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976167_IJBB_11         | 0.338559 | 0.864849 | 0.930282 |  0.95852 | 0.976241 | 0.986368 |  0.99383 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.977000_IJBB_11            | 0.38111  | 0.879649 | 0.936806 | 0.959981 | 0.974781 | 0.984031 | 0.992172 |
  | TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.977500_IJBB_11            | 0.374684 | 0.862999 | 0.933009 | 0.958812 | 0.97556  | 0.987829 | 0.99461  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976833_IJBB_11     | 0.368062 | 0.884615 | 0.938559 | 0.959688 | 0.973418 | 0.984907 | 0.991828 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nesterov_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976000_IJBB_11      | 0.385979 | 0.877702 | 0.938267 | 0.959883 | 0.975268 | 0.985005 | 0.99231  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976167_IJBB_11               | 0.389094 | 0.868939 | 0.938267 | 0.959786 | 0.974294 | 0.984031 | 0.992253 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nestrov_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.977167_IJBB_11| 0.353554 | 0.871081 | 0.935346 | 0.959396 | 0.973223 | 0.984518 | 0.992468 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nestrov_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976833_IJBB_11| 0.366018 | 0.872347 | 0.935248 | 0.959396 | 0.97332  | 0.984518 | 0.992438 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976167_IJBB_11        | 0.337975 | 0.881207 | 0.937098 | 0.959786 | 0.974781 | 0.984713 | 0.992272 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_mmul_1_tmul_1_float16_basic_agedb_30_epoch_33_0.975167_IJBB_11 | 0.403797 | 0.873418 | 0.932619 | 0.957644 | 0.974294 | 0.98481  | 0.992376 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.977667_IJBB_11              | 0.33369  | 0.886271 | 0.937488 | 0.959494 | 0.972833 | 0.984518 | 0.99226  |
  | TT_efv2_b0_swish_21k_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976667_IJBB_11           | 0.374878 | 0.872055 | 0.935638 | 0.959786 | 0.974391 | 0.983544 | 0.991683 |

  |                                                                                                                                                                                             |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_base_basic_agedb_30_epoch_15_0.975500_IJBB_11                                            | 0.335054 | 0.875365 | 0.934859 | 0.959883 | 0.973905 | 0.984518 | 0.992127 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_SD_basic_agedb_30_epoch_17_0.976000_IJBB_11                                              | 0.381305 | 0.871178 | 0.932717 | 0.960273 | 0.975073 | 0.984615 | 0.992523 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_basic_agedb_30_epoch_16_0.978500_IJBB_11                                          | 0.380526 | 0.83038  | 0.924927 | 0.95813  | 0.976923 | 0.98851  | 0.993972 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_curr_basic_agedb_30_epoch_17_0.976333_IJBB_11                                                | 0.374586 | 0.883447 | 0.936319 | 0.959104 | 0.974391 | 0.983252 | 0.99155  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_basic_agedb_30_epoch_17_0.977833_IJBB_11                                  | 0.365823 | 0.813145 | 0.924635 | 0.959202 | 0.978384 | 0.989192 | 0.994198 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_basic_agedb_30_epoch_17_0.978000_IJBB_11               | 0.337683 | 0.821519 | 0.921811 | 0.958033 | 0.979649 | 0.988997 | 0.9943   |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_cutout_basic_agedb_30_epoch_16_0.977500_IJBB_11        | 0.375463 | 0.823466 | 0.92483  | 0.959591 | 0.979065 | 0.988997 | 0.994654 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_random_2_basic_agedb_30_epoch_17_0.977833_IJBB_11                           | 0.360662 | 0.823466 | 0.922006 | 0.95813  | 0.978578 | 0.989094 | 0.994044 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_trip64_basic_agedb_30_epoch_17_0.977333_IJBB_11                                 | 0.344693 | 0.814314 | 0.924732 | 0.961052 | 0.980331 | 0.988997 | 0.994089 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_basic_agedb_30_epoch_16_0.976833_IJBB_11                          | 0.355209 | 0.874781 | 0.936806 | 0.958812 | 0.974489 | 0.984031 | 0.991466 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_2_basic_agedb_30_epoch_16_0.978000_IJBB_11                        | 0.383544 | 0.876339 | 0.936417 | 0.959299 | 0.974294 | 0.983252 | 0.991152 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_basic_agedb_30_epoch_15_0.978833_IJBB_11                       | 0.317235 | 0.87926  | 0.933982 | 0.957936 | 0.974976 | 0.985881 | 0.992409 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_05_64_basic_agedb_30_epoch_16_0.976833_IJBB_11            | 0.398053 | 0.828822 | 0.910321 | 0.955307 | 0.978092 | 0.988413 | 0.994661 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_035_64_basic_agedb_30_epoch_15_0.978500_IJBB_11           | 0.341675 | 0.833204 | 0.924148 | 0.957644 | 0.97887  | 0.988802 | 0.994248 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_035_64_SGD_LA_basic_agedb_30_epoch_16_0.979500_IJBB_11    | 0.370399 | 0.838364 |  0.92814 | 0.959104 | 0.978384 | 0.988315 | 0.993953 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_ada_margin_05_basic_agedb_30_epoch_14_0.972333_IJBB_11            | 0.355696 | 0.623759 | 0.772249 | 0.858909 | 0.92483  | 0.969328 | 0.987304 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_ada_margin_035_basic_agedb_30_epoch_17_0.974833_IJBB_11           | 0.375268 | 0.777799 | 0.856183 | 0.906329 | 0.943525 | 0.973807 | 0.98798  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_ranger_lrbase_1e3_basic_agedb_30_epoch_13_0.978667_IJBB_11     | 0.34333  | 0.875073 | 0.933301 | 0.959104 | 0.973807 | 0.984129 | 0.991442 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_radam_lrbase_1e3_basic_agedb_30_epoch_17_0.976667_IJBB_11      | 0.375755 | 0.866407 | 0.931451 | 0.957838 | 0.97371  | 0.983836 | 0.99186  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M1_basic_agedb_30_epoch_16_0.977667_IJBB_11                       | 0.368452 | 0.88335  | 0.935443 | 0.958325 | 0.974878 | 0.98481  | 0.991652 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_SGD_LA_randaug_cutout_basic_agedb_30_epoch_17_0.977333_IJBB_11 | 0.369718 | 0.877215 | 0.937877 | 0.960565 | 0.975657 | 0.985979 | 0.992154 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_basic_agedb_30_epoch_13_0.979000_IJBB_11        | 0.324927 |  0.8815  | 0.938462 | 0.959494 | 0.975463 | 0.984615 | 0.991511 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_basic_agedb_30_epoch_12_0.978333_IJBB_11                       | 0.339435 | 0.869815 | 0.931938 | 0.959007 | 0.976047 | 0.986465 | 0.992172 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_basic_agedb_30_epoch_17_0.977333_IJBB_11                           | 0.375755 | 0.872639 | 0.940701 | 0.959202 | 0.974781 | 0.984226 | 0.991931 |

  |                                                                                                                                                                                             |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.975000_IJBC_11                                                         | 0.869305 | 0.921511 | 0.951935 | 0.969781 | 0.98149  | 0.988751 | 0.99441  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_cleaned_random_0_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.977167_IJBC_11                                     | 0.877333 | 0.924068 | 0.950862 | 0.968298 | 0.980825 | 0.988649 | 0.993996 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nesterov_l2_5e4_bs512_ms1m_cleaned_random_0_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976833_IJBC_11                               | 0.858107 | 0.919108 | 0.951373 | 0.969065 | 0.980161 | 0.988597 | 0.994375 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_100_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_48_0.976833_IJBC_11                                     | 0.884287 | 0.926471 | 0.951015 | 0.968758 | 0.98057  | 0.988853 | 0.994079 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976667_IJBC_11                                  | 0.878714 | 0.924988 | 0.953214 | 0.969627 | 0.981643 | 0.988802 | 0.994152 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_random_2_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.977333_IJBC_11                                        | 0.876413 | 0.924733 | 0.950759 | 0.969423 | 0.980979 | 0.989058 | 0.994322 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.977000_IJBC_11                                       | 0.875646 | 0.924426 | 0.952856 | 0.970394 | 0.981899 | 0.989006 | 0.994436 |
  | TT_efv2_b0_swish_GDC_arc_M5_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.977500_IJBC_11                                       | 0.861431 | 0.918239 | 0.949686 | 0.968809 | 0.981746 | 0.990847 | 0.995892 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_LA_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976833_IJBC_11                                | 0.879992 | 0.925704 | 0.953214 | 0.970343 | 0.981797 | 0.989467 | 0.99448  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_basic_agedb_30_epoch_17_0.978000_IJBC_11               | 0.736565 | 0.894156 | 0.943294 | 0.968911 | 0.98466  | 0.991972 | 0.99578  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_cutout_basic_agedb_30_epoch_16_0.977500_IJBC_11        | 0.806207 | 0.894411 | 0.943141 | 0.97019  | 0.984558 | 0.992381 | 0.996228 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_basic_agedb_30_epoch_15_0.978833_IJBC_11                       | 0.882548 | 0.925398 | 0.951935 | 0.969218 | 0.981592 | 0.989569 | 0.994229 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_05_64_basic_agedb_30_epoch_16_0.976833_IJBC_11            | 0.803395 | 0.872475 | 0.932454 | 0.965946 | 0.983791 | 0.991665 | 0.995959 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_SGD_LA_randaug_cutout_basic_agedb_30_epoch_17_0.977333_IJBC_11 | 0.880247 | 0.927596 | 0.953214 | 0.970548 | 0.982359 | 0.989773 | 0.994434 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_basic_agedb_30_epoch_13_0.979000_IJBC_11        | 0.883571 | 0.924426 |  0.95352 | 0.969934 | 0.981541 | 0.989058 | 0.99409  |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nesterov_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976000_IJBC_11                                 | 0.869816 | 0.927034 | 0.953418 | 0.970292 | 0.982104 | 0.989262 | 0.994536 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976167_IJBC_11                                   | 0.870174 | 0.925807 | 0.952447 | 0.970394 | 0.98149  | 0.988904 | 0.994642 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_nestrov_l2_5e4_bs512_ms1m_randaug_cutout_ipc_10_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.977167_IJBC_11                           | 0.867976 | 0.924477 | 0.952191 | 0.969372 | 0.981285 | 0.989313 | 0.994807 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976167_IJBC_11                                          | 0.875339 | 0.925858 | 0.953623 | 0.970394 | 0.982001 | 0.989262 | 0.994469 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_48_0.976833_IJBC_11                                          | 0.877026 | 0.925755 | 0.953163 | 0.970292 | 0.981388 | 0.98916  | 0.994394 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_49_0.977667_IJBC_11                                         | 0.876873 | 0.927085 | 0.951884 | 0.970701 | 0.981541 | 0.989364 | 0.994539 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_basic_agedb_30_epoch_17_0.977333_IJBC_11                           | 0.870174 | 0.926471 | 0.955259 | 0.970394 | 0.982257 | 0.989058 | 0.994494 |
## Finetune EfficientNetV2B0
  ```py
  hist_path = "checkpoints/efficientnet_v2/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME + ['lr', 'triplet_embedding_loss']
  pp["epochs"] = [17]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([64], [0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_base_hist.json", fig_label="arc_base", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_curr_hist.json", fig_label="curr", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_SD_hist.json", fig_label="arc_SD", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_hist.json", fig_label="arc_trip64", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_hist.json", fig_label="arc_trip64, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_trip64_hist.json", fig_label="curr_trip64, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_hist.json", fig_label="curr, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_SD_hist.json", fig_label="curr_SD, cleaned", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_hist.json", fig_label="arc_trip_randaug_100, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_random_2_hist.json", fig_label="arc_trip_random_2, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_hist.json", fig_label="arc_trip_randaug_100_no_shear, cleaned", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_cutout_hist.json", fig_label="arc_trip_randaug_100_no_shear_cutout, cleaned", **pp)
  ```
  ```py


  hist_path = "checkpoints/efficientnet_v2/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[1:] + ['lr', 'triplet_embedding_loss']
  pp["epochs"] = [17]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([64], [0.025])]
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_hist.json", fig_label="randaug_cutout E50, arc_base, cleaned", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_2_hist.json", fig_label="random 0, lr_min 1e-5, arc scattter", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_hist.json", fig_label="random 0, arc_M5", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_mixup_hist.json", fig_label="random 0, mixup", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M6_mixup_hist.json", fig_label="random 0, mixup, M6", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_ranger_lrbase_1e3_hist.json", fig_label="arc_M5_ranger_lrbase_1e3", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_radam_lrbase_1e3_hist.json", fig_label="arc_M5_radam_lrbase_1e3", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M1_hist.json", fig_label="arc_M1", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_05_64_hist.json", fig_label="random 0, arc_Arctrip_05_64", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_05_64_SGD_LA_hist.json", fig_label="random 0, arc_Arctrip_05_64_SGD_LA", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_035_64_hist.json", fig_label="random 0, arc_Arctrip_035_64", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_035_64_SGD_LA_hist.json", fig_label="random 0, arc_Arctrip_035_64_SGD_LA", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_ada_margin_05_hist.json", fig_label="random 0, arc_ada_margin_05", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_ada_margin_035_hist.json", fig_label="random 0, arc_ada_margin_035", **pp)

  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_hist.json", fig_label="sphere, lr_base 1e-3", **pp)
  # axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_hist.json", fig_label="arc_M5_randaug_cutout, lr_base 1e-3", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_2_hist.json", fig_label="sphere", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_2_hist.json", fig_label="arc_M5_randaug_cutout", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_SGD_LA_randaug_cutout_hist.json", fig_label="arc_M5_SGD_LA_randaug_cutout", **pp)

  axes, _ = plot.hist_plot_split("checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_softmax_LS_Arctrip_035_1_SGD_LA_hist.json", fig_label="softmax_LS_Arctrip_035_1_SGD_LA", **pp)
  ```
  ```py
  from plot import choose_accuracy
  hist_path = "checkpoints/efficientnet_v2/"  
  aa = [
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_base_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_curr_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_SD_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_trip64_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_SD_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_random_2_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_hist.json",
      hist_path + "TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_cutout_hist.json",

      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_ranger_lrbase_1e3_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_radam_lrbase_1e3_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M1_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_05_64_SGD_LA_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_035_64_SGD_LA_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_ada_margin_035_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_SGD_LA_randaug_cutout_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_2_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_2_hist.json",
      "checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_softmax_LS_Arctrip_035_1_SGD_LA_hist.json",
  ]
  _ = choose_accuracy(aa)

  TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='0' ./IJB_evals.py -m checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_SGD_LA_randaug_cutout_basic_agedb_30_epoch_17_0.977333.h5 -d /datasets/IJB_release
  TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='0' ./IJB_evals.py -m checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_basic_agedb_30_epoch_13_0.979000.h5 -d /datasets/IJB_release
  TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='1' ./IJB_evals.py -m checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_basic_agedb_30_epoch_12_0.978333.h5 -d /datasets/IJB_release
  ```
  |                                                                                                                                                            |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_base_hist                                               | 0.997333 | 0.972    |   0.9755   |      14 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_curr_hist                                                   | 0.997    | 0.973429 |   0.976333 |      16 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_SD_hist                                                 | 0.997167 | 0.974    |   0.976    |      16 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_hist                                             | 0.997667 | 0.981143 |   0.9785   |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip64_hist                                     | 0.998    | 0.982714 |   0.977833 |      14 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_trip64_hist                                    | 0.997833 | 0.983286 |   0.977333 |      16 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_hist                                           | 0.998167 | 0.975857 |   0.976167 |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_curr_SD_hist                                        | 0.9975   | 0.973857 |   0.975167 |      16 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_hist                           | 0.997833 | 0.981714 |   0.974167 |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_random_2_hist                              | 0.997833 | 0.983429 |   0.977833 |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_hist                  | 0.998333 | 0.981    |   0.978    |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_E50_arc_trip_randaug_100_no_shear_cutout_hist           | 0.9975   | 0.983    |   0.9775   |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_hist                             | 0.9975   | 0.977143 |   0.976833 |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_ranger_lrbase_1e3_hist        | 0.997667 | 0.977143 |   0.978667 |      12 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_radam_lrbase_1e3_hist         | 0.997167 | 0.973286 |   0.976667 |      13 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_05_64_SGD_LA_hist        | 0.997667 | 0.978857 |   0.978333 |      14 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_Arctrip_035_64_SGD_LA_hist       | 0.997667 | 0.982286 |   0.9795   |      14 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_ada_margin_035_hist              | 0.997833 | 0.975714 |   0.974833 |      16 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_hist                          | 0.997667 | 0.98     |   0.978333 |      11 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_hist           | 0.9975   | 0.980714 |   0.979    |      12 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_SGD_LA_randaug_cutout_hist    | 0.9975   | 0.980000 |   0.977333 |      16 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_sphere_2_hist                        | 0.998167 | 0.979571 |   0.970667 |      15 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_M5_randaug_cutout_2_hist         | 0.9975   | 0.978714 |   0.977167 |      14 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_softmax_LS_Arctrip_035_1_SGD_LA_hist | 0.997833 | 0.981714 |   0.9775   |      16 |
## EfficientNetV2S
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 3
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_GDC_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="ebv2_s, 21K, GDC, wd5e5, dropout 0.2, drop_connect 0.2, bs512", **pp)
  pp["axes"] = axes

  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="ebv2_s, 21K, GDC, adamw wd1e4, bs512", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_F_wd5e5_arc_emb512_dr03_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="ebv2_m, 21K, GDC, wd5e5, dropout 0.3, drop_connect 0.2, bs512", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_F_wd5e5_arc_emb512_dr03_dr_conn03_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="ebv2_m, 21K, GDC, wd5e5, dropout 0.3, drop_connect 0.3, bs512", **pp)
  ```
  - **EfficientNetV2S**
  ```py
  hist_path = "checkpoints/resnetv2_50_101/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 3
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  # axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet101v2 conv_no_bias, relu", **pp)
  axes, _ = plot.hist_plot_split("checkpoints/TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw", **pp)
  pp["axes"] = axes

  hist_path = "checkpoints/efficientnet_v2/"
  # axes, _ = plot.hist_plot_split(hist_path + "TT_early_efv2_s_add_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cos16_hist.json", fig_label="TT_early_efv2_s_add_GDC", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_early_efv2_s_sd08_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_cos16_hist.json", fig_label="TT_early_efv2_s_sd08_GDC", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_early_efv2_s_sd_1_08_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_cos16_hist.json", fig_label="early, sd_1_08_GDC, l2_5e4_bs1024, random0", **pp)

  hist_path = "checkpoints/"
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_GDC_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="GDC_wd5e5_arc_emb512_dr02", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_F_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="F_wd5e5_arc_emb512_dr02", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_F_wd5e4_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="F_wd5e4_arc_emb512_dr02", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="deep_stem_GDC_wd5e5_arc_emb512_dr02_sgd", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_out_512_F_wd5e5_lr05_arc_emb512_dr04_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="deep_stem_out_512_F_wd5e5_lr05_arc_emb512_dr04", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc0_wd5e4_arc_emb512_sgd_bs1024_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="early_GDC_dr0_drc0_wd5e4_arc_emb512_sgd_bs1024, randaug_cutout", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_random0_bnm09_bne1e5_cos16_float16_hist.json", fig_label="early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512, random0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_random0_bnm09_bne1e5_cos16_float16_2_hist.json", fig_label="early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512, random0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_random0_bnm09_bne1e5_cos49_float16_hist.json", fig_label="early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512, random0, cos49", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512, randaug", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_timm_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="early_GDC_dr0_drc0_lr_01_wd5e4_arc_emb512_sgd_bs512, timm_randaug", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_timm_randaug_cutout_bnm09_bne1e5_cos16_float16_2_hist.json", fig_label="early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512, timm_randaug", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_timm_randaug_no_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512, timm_randaug_no_cutout", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_timm_randaug_no_cutout_bnm09_bne1e5_cos16_float16_reuse_opt_hist.json", fig_label="early_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512, timm_randaug_no_cutout, reuse_opt", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_timm_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="s_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_timm_randaug", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_random0_bnm09_bne1e5_cos16_float16_hist.json", fig_label="s_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_random0", **pp)
  ```
  - **EfficientNetV2S adamw fine-tune**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [17]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_E50_adamw_1e2_hist.json", fig_label="E50, adamw_1e2", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_E50_adamw_1e2_lr_125_hist.json", fig_label="E50, adamw_1e2_lr_125", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_E50_adamw_5e2_lr_125_hist.json", fig_label="E50, adamw_5e2_lr_125", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_deep_stem_GDC_wd5e5_arc_emb512_dr02_adamw_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_E50_adamw_1e1_lr_125_hist.json", fig_label="E50, adamw_1e1_lr_125", **pp)
  ```
  - **EfficientNetV2S strides1**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 3
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  # axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_GDC_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="GDC_wd5e5_arc_emb512_dr02", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_GDC_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_GDC_wd5e5_arc_emb512_dr02", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_F_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_F_wd5e5_arc_emb512_dr02", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_F_wd5e5_arc_emb512_dr04_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_F_wd5e5_arc_emb512_dr04", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_F_wd5e5_lr05_arc_emb512_dr04_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_F_wd5e5_lr05_arc_emb512_dr04", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_out_512_F_wd5e5_arc_emb512_dr04_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_out_512_F_wd5e5_arc_emb512_dr04", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_pw512_F_wd5e5_arc_emb512_dr04_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_pw512_F_wd5e5_arc_emb512_dr04", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_randaug_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_GDC_dr0_drc02_lr_01_wd5e4", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_strides1_F_dr04_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_randaug_bnm09_bne1e5_cos16_float16_hist.json", fig_label="strides1_F_dr04_drc02_lr_01_wd5e4", **pp)

  TT_effv2_s_strides1_GDC_dr0_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_randaug_bnm09_bne1e5_cos16_float16_hist.json
  TT_effv2_s_strides1_F_dr04_drc02_lr_01_wd5e4_arc_emb512_sgd_bs512_ms1m_randaug_bnm09_bne1e5_cos16_float16_hist.json
  ```
  - **EfficientNetV2M**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = plot.EVALS_NAME[:3] + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 3
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_s_F_wd5e5_arc_emb512_dr02_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="s_F_wd5e5_arc_emb512_dr02", **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_GDC_wd5e5_arc_emb512_dr02_sgd_bs1024_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="m_GDC_wd5e5_arc_emb512_dr02", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_F_wd5e5_arc_emb512_dr03_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="m_F_wd5e5_arc_emb512_dr03", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_effv2_m_F_wd5e5_arc_emb512_dr03_dr_conn03_sgd_bs512_ms1m_randaug_cutout_bnm09_bne1e5_cos16_float16_hist.json", fig_label="m_F_wd5e5_arc_emb512_dr03_dr_conn03", **pp)
  ```
  ```py
  from plot import choose_accuracy
  from glob2 import glob

  aa = glob('checkpoints/TT_effv2_*.json')
  _ = choose_accuracy(aa)
  ```
***

# AutoAugment and RandAugment
## Tensorflow image random
  ```py
  import data
  ds, steps_per_epoch = data.prepare_dataset('/datasets/faces_casia_112x112_folders/', random_status=0)
  imms, labels = ds.as_numpy_iterator().next()

  cc = (imms + 1) / 2
  plt.imshow(np.vstack([np.hstack(cc[ii * 16:(ii+1)*16]) for ii in range(int(np.ceil(cc.shape[0] / 16)))]))
  plt.axis('off')
  plt.tight_layout()

  img = cc[4] * 255
  random_status = 3
  total = 10
  aa = np.vstack([
      np.hstack([tf.image.adjust_brightness(img, ii) for ii in arange(-12.75 * random_status, 12.75 * random_status, 12.75 * random_status * 2 / total)]),
      np.hstack([tf.image.adjust_contrast(img, ii) for ii in arange(1 - 0.1 * random_status, 1 + 0.1 * random_status, 0.1 * random_status * 2/ total)]),
      np.hstack([tf.image.adjust_saturation(img, ii) for ii in arange(1 - 0.1 * random_status, 1 + 0.1 * random_status, 0.1 * random_status * 2/ total)]),
      np.hstack([tf.image.adjust_hue(img, ii) for ii in arange(1 - 0.02 * random_status, 1 + 0.02 * random_status, 0.02 * random_status * 2 / total)[:total]]),
      np.hstack([tf.image.adjust_jpeg_quality(img / 255, ii) * 255 for ii in arange(80 - random_status * 5, 80 + random_status * 5, random_status * 5 * 2 / total)]),
  ])
  plt.imshow(aa / 255)
  plt.axis('off')
  plt.tight_layout()
  ```
## RandAugment
  - [Github tensorflow/models augment.py](https://github.com/tensorflow/models/blob/HEAD/official/vision/image_classification/augment.py)
  ```py
  sys.path.append("/home/leondgarse/workspace/tensorflow_models/official/vision/image_classification")
  import augment

  aa = augment.RandAugment(magnitude=5)
  # ['AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd']
  aa.available_ops = ['AutoContrast', 'Equalize', 'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY']
  dd = np.stack([aa.distort(tf.image.random_flip_left_right(tf.convert_to_tensor(ii * 255))) / 255 for ii in cc])
  fig = plt.figure()
  plt.imshow(np.vstack([np.hstack(dd[ii * 16:(ii+1)*16]) for ii in range(int(np.ceil(dd.shape[0] / 16)))]))
  plt.axis('off')
  plt.tight_layout()

  enlarge = 2
  aa = augment.RandAugment(magnitude=5 * enlarge)
  aa.available_ops = ['AutoContrast', 'Equalize', 'Posterize', 'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY']
  dd = np.stack([aa.distort(tf.convert_to_tensor(ii * 255)) / 255 for ii in cc])
  fig = plt.figure()
  plt.imshow(np.vstack([np.hstack(dd[ii * 16:(ii+1)*16]) for ii in range(int(np.ceil(dd.shape[0] / 16)))]))
  plt.axis('off')
  plt.tight_layout()
  ```
## AutoAugment
  ```py
  policy = [
      [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
      [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
      [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
      [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
      [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
      [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
      [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
      [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
      [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
      [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
  ]
  aa = augment.AutoAugment()
  aa.policies = policy
  dd = np.stack([aa.distort(tf.convert_to_tensor(ii * 255)) / 255 for ii in cc])
  fig = plt.figure()
  plt.imshow(np.vstack([np.hstack(dd[ii * 16:(ii+1)*16]) for ii in range(int(np.ceil(dd.shape[0] / 16)))]))
  plt.axis('off')
  plt.tight_layout()
  ```
  ```py
  import autoaugment
  policy = autoaugment.ImageNetPolicy()
  policy_func = lambda img: np.array(policy(tf.keras.preprocessing.image.array_to_img(img)), dtype=np.float32)

  dd = np.stack([policy_func(tf.convert_to_tensor(ii)) / 255 for ii in cc])
  fig = plt.figure()
  plt.imshow(np.vstack([np.hstack(dd[ii * 16:(ii+1)*16]) for ii in range(int(np.ceil(dd.shape[0] / 16)))]))
  plt.axis('off')
  plt.tight_layout()
  ```
***

# EuclideanDense
  ```py
  class EuclideanDense(NormDense):
      def call(self, inputs, **kwargs):
          # Euclidean Distance
          # ==> (xx - yy) ** 2 = xx ** 2 + yy ** 2 - 2 * (xx * yy)
          # xx = np.arange(8).reshape(2, 4).astype('float')
          # yy = np.arange(1, 17).reshape(4, 4).astype('float')
          # aa = np.stack([((yy - ii) ** 2).sum(1) for ii in xx])
          # bb = (xx ** 2).sum(1).reshape(-1, 1) + (yy ** 2).sum(1) - np.dot(xx, yy.T) * 2
          # print(np.allclose(aa, bb))  # True
          a2 = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
          b2 = tf.reduce_sum(tf.square(self.w), axis=-1)
          ab = tf.matmul(inputs, tf.transpose(self.w))
          # output = tf.sqrt(a2 + b2 - 2 * ab) * -1
          # output = (a2 + b2 - 2 * ab) / 2 * -1
          output = ab - (a2 + b2) / 2
          return output
  ```
  ```py
  hist_path = "checkpoints/mobilenet_casia_tests/"
  pp = {}
  pp["epochs"] = [5, 5, 10, 10, 40]
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "lr"]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.01, 0.001])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_base_bs400_hist.json", fig_label='Mobilnet, CASIA, emb256, dr0, bs400, base', names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_MSEDense_margin_softmax_sgdw_5e4_emb256_dr0_bs400_hist.json", **pp)
  ```
***

# MXNet Pytorch Results
  ```sh
  ./IJB_evals.py -P IJB_result/GhostNet_x1.3_Arcface_Epoch_24_IJBB.npz IJB_result/iresnet50_IJBB.npz IJB_result/r100-arcface-msfdrop75_IJBB.npz IJB_result/glint360k_r100FC_0.1_fp16_cosface8GPU_model_average_IJBB.npz IJB_result/glint360k_r100FC_1.0_fp16_cosface8GPU_model_average_IJBB.npz IJB_result/glint360k_r100FC_1.0_fp16_cosface8GPU_model_IJBB.npz IJB_result/glint360k_r100FC_0.1_fp16_cosface8GPU_model_IJBB.npz IJB_result/MS1MV2-ResNet100-Arcface_IJBB.npz /datasets/IJB_release/IJBB/meta/ijbb_template_pair_label.txt

  ./IJB_evals.py -P "IJB_result/glint360k_r100FC_1.0_fp16_cosface8GPU_IJBC.npz", "IJB_result/MS1MV2-ResNet100-Arcface_IJBC.npz", "IJB_result/GhostNet_x1.3_Arcface_Epoch_24_IJBC_11.npz", "/datasets/IJB_release/IJBC/meta/ijbc_template_pair_label.txt"
  ```

  - **MXNet MS1MV3**
  | Datasets       | backbone   | IJBC(1e-05) | IJBC(1e-04) | agedb30 | cfp_fp | lfw   |
  | -------------- | ---------- | ----------- | ----------- | ------- | ------ | ----- |
  | MS1MV3-Arcface | r18-fp16   | 92.07       | 94.66       | 97.77   | 97.73  | 99.77 |
  | MS1MV3-Arcface | r34-fp16   | 94.10       | 95.90       | 98.10   | 98.67  | 99.80 |
  | MS1MV3-Arcface | r50-fp16   | 94.79       | 96.46       | 98.35   | 98.96  | 99.83 |
  | MS1MV3-Arcface | r100-fp16  | 95.31       | 96.81       | 98.48   | 99.06  | 99.85 |
  | MS1MV3-Arcface | r2060-fp16 | 95.34       | 97.11       | 98.67   | 99.24  | 99.87 |

  - **MXNet Glint360k**
  | Datasets          | backbone      | IJBC(1e-05) | IJBC(1e-04) | agedb30 | cfp_fp | lfw   |
  | ----------------- | ------------- | ----------- | ----------- | ------- | ------ | ----- |
  | Glint360k-Cosface | r18-fp16-0.1  | 93.16       | 95.33       | 97.72   | 97.73  | 99.77 |
  | Glint360k-Cosface | r34-fp16-0.1  | 95.16       | 96.56       | 98.33   | 98.78  | 99.82 |
  | Glint360k-Cosface | r50-fp16-0.1  | 95.61       | 96.97       | 98.38   | 99.20  | 99.83 |
  | Glint360k-Cosface | r100-fp16-0.1 | 95.88       | 97.32       | 98.48   | 99.29  | 99.82 |

  - **cavaface.pytorch**
  | Backbone             | Dataset   | Head           | Loss     | Flops/Params  | Megaface(Id/ver@1e-6) | IJBC(tar@far=1e-4) |
  | -------------------- | --------- | -------------- | -------- | ------------- | --------------------- | ------------------ |
  | AttentionNet-IRSE-92 | MS1MV3    | MV-AM          | Softmax  | 17.63G/55.42M | 99.1356/99.3999       | 96.56              |
  | IR-SE-100            | MS1MV3    | Arcface        | Softmax  | 24.18G/65.5M  | 99.0881/99.4259       | 96.69              |
  | IR-SE-100            | MS1MV3    | ArcNegface     | Softmax  | 24.18G/65.5M  | 99.1304/98.7099       | 96.81              |
  | IR-SE-100            | MS1MV3    | Curricularface | Softmax  | 24.18G/65.5M  | 99.0497/98.6162       | 97.00              |
  | IR-SE-100            | MS1MV3    | Combined       | Softmax  | 24.18G/65.5M  | 99.0718/99.4493       | 96.83              |
  | IR-SE-100            | MS1MV3    | CircleLoss     | Softplus | 24.18G/65.5M  | 98.5732/98.4834       | 96.52              |
  | ResNeSt-101          | MS1MV3    | Arcface        | Softmax  | 18.45G/97.61M | 98.8746/98.5615       | 96.63              |
  | DenseNet-201         | MS1MV3    | Arcface        | Softmax  | 8.52G/66.37M  | 98.3649/98.4294       | 96.03              |
  | IR-100               | Glint360k | Arcface        | Softmax  | 24.18G/65.5M  | 99.2964/98.8792       | 97.19              |
  | IR-100               | Glint360k | CosFace        | Softmax  | 24.18G/65.5M  | 99.2625/99.1812       | 97.40              |

  - **IJBB**
  |                                                         |        1e-06 |        1e-05 |       0.0001 |        0.001 |         0.01 |          0.1 |          AUC |
  |:------------------------------------------------------- | ------------:| ------------:| ------------:| ------------:| ------------:| ------------:| ------------:|
  | GhostNet_x1.3_Arcface_Epoch_24_IJBB                     |     0.352678 |     0.881694 |     0.928724 |     0.954041 |     0.972055 |     0.985784 |     0.993646 |
  | MS1MV2-ResNet100-Arcface_IJBB                           |      0.42814 |     0.908179 |     0.948978 |     0.964654 |     0.976728 |     0.986563 |     0.993771 |
  | r100-arcface-msfdrop75_IJBB                             |     0.441772 |     0.905063 |     0.949464 |     0.965823 |     0.978578 |     0.988802 |     0.995144 |
  | glint360k_iresnet50_IJBB                                |     0.436611 |     0.926972 |     0.957157 |     0.970691 |     0.979065 |     0.986855 |     0.993991 |
  | glint360k_r100FC_1.0_fp16_cosface8GPU_model_IJBB        |     0.460857 | **0.938364** | **0.962317** |     0.970789 |      0.98111 |     0.988023 |     0.993649 |
  | glint360k_r100FC_1.0_fp16_cosface8GPU_mode_average_IJBB | **0.464849** |     0.937001 |      0.96222 |     0.970789 |     0.981597 |     0.988023 |     0.993612 |
  | glint360k_r100FC_0.1_fp16_cosface8GPU_model_IJBB        |     0.450536 |     0.931938 |     0.961928 |     0.972639 |     0.981986 |     0.989679 | **0.995287** |
  | glint360k_r100FC_0.1_fp16_cosface8GPU_mode_average_IJBB |      0.44742 |     0.932619 |     0.961831 | **0.972833** | **0.982278** | **0.989971** |     0.995285 |

  - **Keras IJBB**
  |                                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
  | TT_mobilenet_swish_pointwise_distill_128_arc_emb512_dr04_wd5e4_bs512_r100_ms1m_fp16_cosine_basic_agedb_30_epoch_50_0.975333_IJBB_11                                              | 0.380721 |  0.85258 |  0.91889 | 0.951412 | 0.972833 | 0.986465 |
  | TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_50_0.976333_IJBB_11                 | 0.327069 | 0.887244 | 0.932425 |  0.95482 | 0.972541 |  0.98481 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.975000_IJBB_11                                              | 0.369133 | 0.871081 | 0.937098 | 0.959104 | 0.973905 | 0.984518 |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976667_IJBB_11                       | 0.352288 | 0.871276 | 0.937293 | 0.959981 | 0.975268 |  0.98481 |
  | TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_basic_agedb_30_epoch_45_batch_4000_0.980167_IJBB_11              | 0.349172 | 0.904284 | 0.944693 | 0.962707 | 0.974878 | 0.983739 |
  | TT_resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_6000_0.983667_IJBB_11  |  0.40224 | 0.916943 | 0.949951 |  0.96446 | 0.976728 |  0.98666 |
  | TT_resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_44_batch_2000_0.985000_IJBB_11 | 0.397371 | 0.914606 | 0.952483 | 0.967381 | 0.978773 | 0.987439 |
  | TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333_IJBB_11                                                        | 0.385589 | 0.915871 | 0.950828 | 0.965141 | 0.976923 | 0.985784 |
  | TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_4000_0.984500_IJBB_11                                       | 0.390944 | 0.924537 | 0.954333 | 0.967868 | 0.979747 | 0.987244 |

  - **IJBC**
  |                                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:| --------:|
  | glint360k_r100FC_1.0_fp16_cosface8GPU_IJBC                                                                                                                                       | 0.872066 | 0.961497 | 0.973871 | 0.980672 | 0.987421 | 0.991819 | 0.995874 |
  | GhostNet_x1.3_Arcface_Epoch_24_IJBC_11                                                                                                                                           | 0.876259 | 0.922023 | 0.945748 |  0.96477 | 0.978985 | 0.990336 | 0.995519 |
  | TT_mobilenet_swish_pointwise_distill_128_arc_emb512_dr04_wd5e4_bs512_r100_ms1m_fp16_cosine_basic_agedb_30_epoch_50_0.975333_IJBC_11                                              | 0.851255 | 0.907808 | 0.940328 | 0.963133 | 0.979751 | 0.991154 |          |
  | TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_50_0.976333_IJBC_11                 | 0.872833 | 0.922892 | 0.949328 | 0.966457 | 0.980263 | 0.989262 |          |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.975000_IJBC_11                                              | 0.869305 | 0.921511 | 0.951935 | 0.969781 |  0.98149 | 0.988751 |          |
  | TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976667_IJBC_11                       | 0.878714 | 0.924988 | 0.953214 | 0.969627 | 0.981643 | 0.988802 |          |
  | TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_basic_agedb_30_epoch_45_batch_4000_0.980167_IJBC_11              | 0.897735 | 0.936493 | 0.959145 | 0.973411 | 0.983075 | 0.989262 |          |
  | TT_resnet50v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_6000_0.983667_IJBC_11  | 0.909853 | 0.946106 | 0.963696 | 0.974383 | 0.983842 | 0.990694 |          |
  | TT_resnet101v2_swish_pad_same_first_conv_k3_stride_1_conv_no_bias_E_arc_emb512_dr04_sgd_l2_5e4_bs384_ms1m_bnm09_bne1e4_cos16_basic_agedb_30_epoch_44_batch_2000_0.985000_IJBC_11 | 0.900138 | 0.948816 | 0.966406 | 0.977144 | 0.985172 |  0.99187 |          |
  | TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333_IJBC_11                                                        | 0.896712 | 0.946106 | 0.964463 |  0.97607 | 0.984251 | 0.990131 |          |
  | TT_se_r50_SD_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_49_batch_4000_0.984500_IJBC_11                                       | 0.907757 | 0.950759 | 0.966252 | 0.976684 | 0.985172 | 0.990489 |          |
***

# cavaface.pytorch
  ```py
  from optimizer.lr_scheduler import CosineWarmupLR
  aa = CosineWarmupLR(None, batches=100, epochs=16, base_lr=0.1, target_lr=1e-5, warmup_epochs=0, warmup_lr=0)
  aa.last_iter = -1
  xx, yy = range(100 * 32), []
  for ii in xx:
      aa.last_iter += 1
      aa.get_lr()
      yy.append(aa.learning_rate)
  plt.plot(xx, yy)

  targetlr = 1e-5
  baselr = 0.1
  niters = 100
  warmup_iters = 0
  learning_rate = lambda nn: targetlr + (baselr - targetlr) * (1 + np.cos(np.pi * (nn - warmup_iters) / (niters - warmup_iters))) / 2
  xx = np.arange(1, 120)
  plt.plot(xx, learning_rate(xx))
  ```
  ```py
  import evals
  from data_distiller import teacher_model_interf_wrapper
  mm = teacher_model_interf_wrapper('../models/GhostNet_x1.3_Arcface_Epoch_24.pth')
  for aa in [os.path.join("/datasets/ms1m-retinaface-t1", ii) for ii in ["lfw.bin", "cfp_fp.bin", "agedb_30.bin"]]:
      evals.eval_callback(lambda imm: mm(imm * 128 + 127.5), aa).on_epoch_end()
  # lfw evaluation max accuracy: 0.997333, thresh: 0.290534
  # cfp_fp evaluation max accuracy: 0.975571, thresh: 0.176912
  # agedb_30 evaluation max accuracy: 0.974667, thresh: 0.196453
  ```
  ```py
  from torchsummary import summary
  from backbone import ghostnet
  tt = ghostnet.GhostNet([112, 112])
  summary(tt, (3, 112, 112))
  ```
  ```py
  import mxnet as mx
  from tqdm import tqdm

  def load_mx_rec(rec_path, PICK_FIRST=-1):
      save_path = os.path.join(rec_path, "imgs")
      if not os.path.exists(save_path):
          os.makedirs(save_path)

      idx_path = os.path.join(rec_path, "train.idx")
      bin_path = os.path.join(rec_path, "train.rec")
      imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
      img_info = imgrec.read_idx(0)
      header,_ = mx.recordio.unpack(img_info)
      max_idx = int(header.label[0])
      pre_label, image_idx = -1, 0
      for idx in tqdm(range(1, max_idx if PICK_FIRST == -1 else PICK_FIRST)):
          img_info = imgrec.read_idx(idx)
          header, img = mx.recordio.unpack(img_info)
          # label = int(header.label)
          label = int(header.label if isinstance(header.label, float) else header.label[0]) + max_idx

          if label != pre_label:
              image_idx = 0
              pre_label = label

          label_path = os.path.join(save_path, "0_" + str(label))
          if not os.path.exists(label_path):
              os.makedirs(label_path)
          with open(os.path.join(label_path, '{}.jpg'.format(image_idx)), "wb") as ff:
              ff.write(img)
          image_idx += 1
  ```
  ```py
  with open('retina_clean.txt', 'r') as ff:
      aa = ff.readlines()
  dd = {}
  for ii in aa:
      ii = ii.strip()
      kk, vv = os.path.dirname(ii), int(os.path.splitext(os.path.basename(ii))[0])
      dd.setdefault(kk, []).append(vv)

  PATH = '/datasets/ms1m-retinaface-t1/imgs/'
  print(sum([ii in dd for ii in os.listdir(PATH)]))
  # 92317

  rr = []
  for kk, vv in dd.items():
      cc = [int(os.path.splitext(ii)[0]) for ii in os.listdir(os.path.join(PATH, kk))]
      rr.append([ii in cc for ii in vv])
  print(sum([sum(ii) for ii in rr]))
  # 5096068

  rr = []
  for kk, vv in dd.items():
      cc = [int(os.path.splitext(ii)[0]) for ii in os.listdir(os.path.join(PATH, kk))]
      rr.append([ii in cc for ii in vv])
  print(sum([sum(ii) for ii in rr]))
  # 5096068

  tt = []
  for ii in os.listdir(PATH):
      vv = dd.get(ii, [])
      tt.append([int(os.path.splitext(jj)[0]) in vv for jj in os.listdir(os.path.join(PATH, ii))])
  cc = [(id, sum(ii), len(ii)) for id, ii in enumerate(tt) if False in ii]
  print(len(cc))
  # 37730

  from tqdm import tqdm
  target = "/datasets/ms1m-retinaface-t1-cleaned_112x112_folders"
  if not os.path.exists(target):
      os.makedirs(target)
  for id, (kk, vv) in tqdm(enumerate(dd.items())):
      image_source = os.path.join(PATH, kk)
      image_target = os.path.join(target, str(id))
      if not os.path.exists(image_target):
          os.makedirs(image_target)
      for ii in vv:
          image_name = str(ii) + ".jpg"
          os.rename(os.path.join(image_source, image_name), os.path.join(image_target, image_name))
          # print(os.rename(os.path.join(image_source, image_name), os.path.join(image_target, image_name)))

  for ii in os.listdir(PATH):
      if len(os.listdir(os.path.join(PATH, ii))) == 0:
          os.removedirs(os.path.join(PATH, ii))
  ```
# MLP
  ```py
  sys.path.append('../Keras_mlp/')
  import mlp_mixer
  bb = mlp_mixer.MlpMixerModel_S16(input_shape=(112, 112, 3), num_classes=512, dropout=0.4, classifier_activation=None)

  embedding = keras.layers.BatchNormalization()(bb.outputs[0])
  embedding_fp32 = keras.layers.Activation("linear", dtype="float32", name="embedding")(embedding)
  basic_model = keras.models.Model(bb.inputs[0], embedding_fp32)

  bb = mlp_mixer.MlpMixerModel_S16(input_shape=(112, 112, 3), num_classes=0)
  out = keras.layers.Reshape((7, 7, 512))(bb.outputs[0])
  basic_model = keras.models.Model(bb.inputs[0], out)
  basic_model = models.buildin_models(basic_model, output_layer='GDC')
  ```
  ```py
  sys.path.append('../Keras_mlp/')
  import res_mlp
  bb = res_mlp.ResMLP(input_shape=(112, 112, 3), num_blocks=8, patch_size=16, hidden_dim=512, mlp_dim=512 * 4, num_classes=512, dropout=0.4, classifier_activation=None)
  ```
# mixup
  - [mixup: BEYOND EMPIRICAL RISK MINIMIZATION](https://arxiv.org/pdf/1710.09412.pdf)
    - For mixup, we find that α ∈ [0.1, 0.4] leads to improved performance over ERM, whereas for large α, mixup leads to underfitting.
    - We also find that models with higher capacities and/or longer training runs are the ones to benefit the most from mixup.
    - For example, when trained for 90 epochs, the mixup variants of ResNet-101 and ResNeXt-101 obtain a greater improvement (0.5% to 0.6%) over their ERM analogues than the gain of smaller models such as ResNet-50 (0.2%).
    - When trained for 200 epochs, the top-1 error of the mixup variant of ResNet-50 is further reduced by 1.2% compared to the 90 epoch run, whereas its ERM analogue stays the same.
  ```py
  DEFAULT_ALPHA = 0.4
  def sample_beta_distribution(size, concentration_0=DEFAULT_ALPHA, concentration_1=DEFAULT_ALPHA):
      gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
      gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
      return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

  def mixup(image, label, alpha=DEFAULT_ALPHA):
      """Applies Mixup regularization to a batch of images and labels.

      [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
        Mixup: Beyond Empirical Risk Minimization.
        ICLR'18, https://arxiv.org/abs/1710.09412

      Arguments:
        batch_size: The input batch size for images and labels.
        alpha: Float that controls the strength of Mixup regularization.
        image: a Tensor of batched images.
        label: a Tensor of batch labels.

      Returns:
        A new dict of features with updated images and labels with the same
        dimensions as the input with Mixup regularization applied.
      """
      # mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
      batch_size = tf.shape(image)[0]
      mix_weight = sample_beta_distribution(batch_size, alpha, alpha)
      mix_weight = tf.maximum(mix_weight, 1. - mix_weight)

      # Regard values with `> 0.9` as no mixup, this probability is near `1 - alpha`
      # alpha: no_mixup --> {0.2: 0.6714, 0.4: 0.47885, 0.6: 0.35132, 0.8: 0.26354, 1.0: 0.19931}
      mix_weight = tf.where(mix_weight > 0.9, tf.ones_like(mix_weight), mix_weight)

      label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
      img_mix_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), image.dtype)

      shuffle_index = tf.random.shuffle(tf.range(batch_size))
      image = image * img_mix_weight + tf.gather(image, shuffle_index) * (1. - img_mix_weight)
      label = tf.cast(label, "float32")
      label = label * label_mix_weight + tf.gather(label, shuffle_index) * (1 - label_mix_weight)
      return image, label

  import tensorflow_datasets as tfds
  preprocessing = lambda data: (tf.cast(data["image"], tf.float32) / 255.0, tf.one_hot(data["label"], depth=10))
  dataset = tfds.load("cifar10", split="train").map(preprocessing).batch(64)
  image, label = dataset.as_numpy_iterator().next()
  aa, bb = mixup(image, label)
  plt.imshow(np.vstack([np.hstack(aa[ii * 8: (ii + 1) * 8]) for ii in range(8)]))

  cc = []
  for alpha in np.arange(0, 1.2, 0.2):
      aa = sample_beta_distribution(10000, alpha, alpha)
      bb = tf.maximum(aa, 1 - aa).numpy()
      name = "alpha {:.1f}".format(alpha)
      print(name, dict(zip(*np.histogram(bb, bins=5)[::-1])))
      cc.append((bb > 0.9).sum() / bb.shape[0])
      # plt.plot(np.histogram(bb, bins=50)[0], label=name)
      plt.hist(bb, bins=50, alpha=0.5, label=name)
  plt.legend()
  plt.xlim(0.5, 1)
  plt.tight_layout()
  # alpha 0.0 {0.0: 0, 0.2: 0, 0.4: 10000, 0.6: 0, 0.8: 0}
  # alpha 0.2 {0.5001278: 663, 0.60010225: 710, 0.7000767: 814, 0.8000511: 1130, 0.90002555: 6683}
  # alpha 0.4 {0.50006217: 1051, 0.60004973: 1166, 0.7000373: 1317, 0.80002487: 1647, 0.90001243: 4819}
  # alpha 0.6 {0.5001339: 1521, 0.60010684: 1522, 0.70007986: 1613, 0.8000528: 1905, 0.90002584: 3439}
  # alpha 0.8 {0.5001678: 1736, 0.6001339: 1779, 0.7001: 1848, 0.8000661: 2043, 0.9000322: 2594}
  # alpha 1.0 {0.500016: 1959, 0.5999865: 2043, 0.699957: 2079, 0.7999276: 1973, 0.8998981: 1946}

  print(dict(zip(np.arange(0, 1.2, 0.2), cc)))
  # {0.0: 0.0, 0.2: 0.6714, 0.4: 0.47885, 0.6000000000000001: 0.35132, 0.8: 0.26354, 1.0: 0.19931}
  ```
# Cutout
  ```py
  def cutout(image: tf.Tensor, pad_size: int, replace: int = 0) -> tf.Tensor:
      """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

      This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
      a random location within `img`. The pixel values filled in will be of the
      value `replace`. The located where the mask will be applied is randomly
      chosen uniformly over the whole image.

      Args:
        image: An image Tensor of type uint8.
        pad_size: Specifies how big the zero mask that will be generated is that is
          applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
        replace: What pixel value to fill in the image in the area that has the
          cutout mask applied to it.

      Returns:
        An image Tensor that is of type uint8.
      """
      image_height = tf.shape(image)[0]
      image_width = tf.shape(image)[1]

      # Sample the center location in the image where the zero mask will be applied.
      cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height, dtype=tf.int32)

      cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)

      lower_pad = tf.maximum(0, cutout_center_height - pad_size)
      upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
      left_pad = tf.maximum(0, cutout_center_width - pad_size)
      right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

      cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
      padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
      mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
      mask = tf.expand_dims(mask, -1)
      mask = tf.tile(mask, [1, 1, 3])
      image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image)
      return image

  import skimage.data
  imm = tf.convert_to_tensor(skimage.data.chelsea())
  cutout_const = 80
  cutout_cond = lambda img: cutout(img, cutout_const, 128) if np.random.uniform() > 0.5 else img
  plt.imshow(np.hstack([cutout_cond(imm) / 255 for _ in range(5)]))
  plt.axis("off")
  plt.tight_layout()
  ```
  ![](images/cut_out.png)
# insightface recognition
  - **Save to jit and onnx model**
    ```py
    !cd insightface/recognition/arcface_torch

    import torch
    from torchsummary import summary
    from backbones import iresnet50, iresnet100

    model_path = "../../../models/partial_fc_pytorch/glint360k_cosface_r50_fp16_0.1/backbone.pth"
    model_dir = os.path.dirname(os.path.dirname(model_path))
    pth_model_name = os.path.basename(os.path.dirname(model_path)) + ".pth"
    onnx_model_name = os.path.basename(os.path.dirname(model_path)) + ".onnx"
    pth_save_path = os.path.join(model_dir, pth_model_name)
    onnx_save_path = os.path.join(model_dir, onnx_model_name)
    print(f"{model_path = } {model_dir = }")

    model = iresnet50() if "r50" in model_path else iresnet100()
    weight = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weight)
    model.eval()
    summary(model, (3, 112, 112))

    dummy_input = torch.randn(10, 3, 112, 112)
    # Default is training=torch.onnx.TrainingMode.EVAL, will fuse conv and batchnorm layers
    # torch.onnx.export(model, dummy_input, onnx_save_path, verbose=False, keep_initializers_as_inputs=True)
    # torch.onnx.export(model, dummy_input, onnx_save_path, verbose=False, keep_initializers_as_inputs=True, training=torch.onnx.TrainingMode.TRAINING)
    torch.onnx.export(model, dummy_input, onnx_save_path, verbose=False, keep_initializers_as_inputs=True, training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=True)
    print(f"Saved onnx: {onnx_save_path = }")

    traced_cell = torch.jit.trace(model, (dummy_input))
    torch.jit.save(traced_cell, pth_save_path)
    print(f"Saved jit pth: {pth_save_path = }")

    aa = torch.jit.load(pth_save_path)
    ```
  - **onnx to keras**
    ```sh
    git clone https://github.com/gmalivenko/onnx2keras.git
    cd onnx2keras
    vi /opt/anaconda3/lib/python3.8/site-packages/onnx2keras/converter.py
    - 203             1: 3,
    + 203             1: -1,

    pip install .
    cd -
    ```
    ```py
    import onnx
    from onnx2keras import onnx_to_keras

    # Load ONNX model
    onnx_model_file = 'glint360k_cosface_r50_fp16_0.1.onnx'
    onnx_model = onnx.load(onnx_model_file)
    k_model = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name], name_policy="renumerate", change_ordering=True)
    k_model.save(os.path.splitext(onnx_model_file)[0] + "_channel_last.h5")
    ```
  - **keras model load weights**
    ```py
    import models
    import torch

    torch_model_path = '../models/partial_fc_pytorch/glint360k_cosface_r50_fp16_0.1.pth'
    torch_model = torch.jit.load(torch_model_path)
    model_type = "r50" if "r50" in torch_model_path else "r100"
    mm = models.buildin_models(model_type, dropout=0, emb_shape=512, output_layer='E', bn_momentum=0.9, bn_epsilon=1e-5, use_bias=True, scale=True, activation='PReLU')

    torch_params = {kk: np.cumproduct(vv.shape)[-1] for kk, vv in torch_model.state_dict().items() if ".num_batches_tracked" not in kk}
    print("torch_model total_parameters :", np.sum(list(torch_params.values())))

    keras_params = {ii.name: int(sum([np.cumproduct(jj.shape)[-1] for jj in ii.weights])) for ii in mm.layers}
    print("torch_model total_parameters :", np.sum(list(keras_params.values())))

    input_output_rr = {
        "conv1" : "0_conv",
        'bn1': '0_bn',
        'prelu': '0_PReLU',
        "bn2": "E_batchnorm",
        "fc": "E_dense",
        "features": "pre_embedding",
    }
    stack_rr = {"layer{}".format(ii): "stack{}_".format(ii) for ii in range(1, 5)}
    block_rr = {"{}".format(ii): "block{}_".format(ii + 1) for ii in range(30)}
    layer_rr = {
        "bn1": "1_bn",
        "conv1": "1_conv",
        "bn2": "2_bn",
        "prelu": "2_PReLU",
        "conv2": "2_conv",
        "bn3": "3_bn",
        "downsample.0": "shortcut_conv",
        "downsample.1": "shortcut_bn",
    }

    def match_layer_name(torch_layer_name):
        splitted_name = torch_layer_name.split('.')
        if splitted_name[0] in input_output_rr:
             return input_output_rr[splitted_name[0]]
        else:
            stack_nn, block_nn = splitted_name[0], splitted_name[1]
            if len(splitted_name) == 5: # 'layer1.0.downsample.0.weight'
                layer_nn = ".".join(splitted_name[2:4])
            else:
                layer_nn = splitted_name[2]
            return "".join([stack_rr[stack_nn], block_rr[block_nn], layer_rr[layer_nn]])

    aa = torch_model.state_dict()
    bb = {ii: match_layer_name(ii) for ii in aa.keys()}
    cc = set(bb.values())
    print("TF layers not contained in torch:", [ii.name for ii in mm.layers if ii.name not in cc])

    # dd = {kk: (aa[kk].shape, mm.get_layer(vv).weights[0 if "weight" in kk else 1].shape) for kk, vv in bb.items()}

    tf_weights_dict = {"weight": 0, "bias": 1, "running_mean": 2, "running_var": 3}
    for kk, vv in bb.items():
        torch_weight = aa[kk].detach().numpy()
        torch_weight_type = kk.split(".")[-1]
        if torch_weight_type == "num_batches_tracked":
            continue

        tf_layer = mm.get_layer(vv)
        tf_weights = tf_layer.get_weights()
        tf_weight_pos = tf_weights_dict[torch_weight_type]

        print("[{}] torch: {}, tf: {}".format(kk, torch_weight.shape, tf_weights[tf_weight_pos].shape))

        if tf_weight_pos == 0:
            if isinstance(tf_layer, keras.layers.Conv2D):
                torch_weight = np.transpose(torch_weight, (2, 3, 1, 0))
            elif isinstance(tf_layer, keras.layers.BatchNormalization):
                torch_weight = torch_weight
            elif isinstance(tf_layer, keras.layers.PReLU):
                torch_weight = np.expand_dims(np.expand_dims(torch_weight, 0), 0)
            elif isinstance(tf_layer, keras.layers.Dense):
                # fc layer after flatten, weights need to reshape according to NCHW --> NHWC
                torch_weight = torch_weight.reshape(512, 512, 7, 7).transpose([2, 3, 1, 0]).reshape(-1, 512)

        tf_weights[tf_weight_pos] = torch_weight
        tf_layer.set_weights(tf_weights)

    save_path = os.path.splitext(torch_model_path)[0] + ".h5"
    mm.save(save_path)
    print("Saved model:", save_path)

    torch_out = torch_model(torch.from_numpy(np.ones([1, 3, 112, 112], dtype='float32'))).detach().numpy()
    keras_out = mm(np.ones([1, 112, 112, 3], dtype='float32'))
    print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
    ```
  - **Evaluating**
    ```sh
    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='0' ./IJB_evals.py -m ../models/partial_fc_pytorch/glint360k_cosface_r100_fp16_0.1.h5 -d /datasets/IJB_release/ -s IJBC
    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='0' ./IJB_evals.py -m ../models/partial_fc_pytorch/partial_fc_glint360k_r50.h5 -d /datasets/IJB_release/ -s IJBC
    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='0' ./IJB_evals.py -m ../models/partial_fc_pytorch/ms1mv3_arcface_r50_fp16.h5 -d /datasets/IJB_release/ -s IJBC

    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='1' ./IJB_evals.py -m ../models/partial_fc_pytorch/glint360k_cosface_r50_fp16_0.1.h5 -d /datasets/IJB_release/ -s IJBC
    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='1' ./IJB_evals.py -m ../models/partial_fc_pytorch/ms1mv3_arcface_r100_fp16.h5 -d /datasets/IJB_release/ -s IJBC
    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='1' ./IJB_evals.py -m ../models/partial_fc_pytorch/partial_fc_glint360k_r100.h5 -d /datasets/IJB_release/ -s IJBC

    TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES='1' ./evals.py -m ../models/partial_fc_pytorch/partial_fc_glint360k_r100.h5 -t /datasets/celeb_deepglint/lfw.bin /datasets/celeb_deepglint/cfp_fp.bin /datasets/celeb_deepglint/agedb_30.bin
    ```
    | Datasets          | backbone     | IJBC(1e-05) | IJBC(1e-04) | agedb30 | cfp_fp | lfw   |
    | ----------------- | ------------ | ----------- | ----------- | ------- | ------ | ----- |
    | MS1MV3-Arcface    | r34-fp16     | 94.10       | 95.90       | 98.10   | 98.67  | 99.80 |
    | Glint360k-Cosface | r34-fp16-0.1 | 95.16       | 96.56       | 98.33   | 98.78  | 99.82 |
    | MS1MV3-Arcface    | r18-fp16     | 92.07       | 94.66       | 97.77   | 97.73  | 99.77 |
    | Glint360k-Cosface | r18-fp16-0.1 | 93.16       | 95.33       | 97.72   | 97.73  | 99.77 |

    | model                           | lfw      | cfp_fp   | agedb_30 | IJBB     | IJBC     |
    | ------------------------------- | -------- | -------- | -------- | -------- | -------- |
    | glint360k_cosface_r18_fp16_0.1  | 0.997500 | 0.977143 | 0.976500 | 0.936806 | 0.9533   |
    | glint360k_cosface_r34_fp16_0.1  | 0.998167 | 0.987000 | 0.982833 | 0.951801 | 0.9656   |
    | ms1mv3_arcface_r50_fp16         | 0.998667 | 0.989143 | 0.983500 | 0.950049 | 0.964463 |
    | glint360k_cosface_r50_fp16_0.1  | 0.998333 | 0.991429 | 0.983000 | 0.958228 | 0.969832 |
    | partial_fc_glint360k_r50        | 0.998333 | 0.991000 | 0.983500 | 0.957157 | 0.970292 |
    | ms1mv3_arcface_r100_fp16        | 0.998500 | 0.990143 | 0.984333 | 0.95482  | 0.968042 |
    | glint360k_cosface_r100_fp16_0.1 | 0.998333 | 0.992143 | 0.984333 | 0.961928 | 0.973155 |
    | partial_fc_glint360k_r100       | 0.998500 | 0.992286 | 0.985167 | 0.962512 | 0.974689 |

    |                                         |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
    |:--------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:| --------:|
    | glint360k_cosface_r18_fp16_0.1_IJBB_11  |  0.40224 | 0.893574 | 0.936806 | 0.959202 | 0.976534 | 0.986952 |  0.99489 |
    | glint360k_cosface_r34_fp16_0.1_IJBB_11  | 0.416748 | 0.917819 | 0.951801 | 0.968257 | 0.979552 | 0.989484 | 0.994851 |
    | ms1mv3_arcface_r50_fp16_IJBB_11         | 0.383642 | 0.913535 | 0.950049 | 0.966018 | 0.978676 | 0.986563 | 0.993787 |
    | glint360k_cosface_r50_fp16_0.1_IJBB_11  | 0.452872 | 0.926095 | 0.958228 | 0.969912 | 0.981402 | 0.989484 | 0.994999 |
    | partial_fc_glint360k_r50_IJBB_11        | 0.436611 | 0.926972 | 0.957157 | 0.970691 | 0.979065 | 0.986855 | 0.993991 |
    | ms1mv3_arcface_r100_fp16_IJBB_11        | 0.402045 | 0.921032 |  0.95482 | 0.969231 | 0.978384 |  0.98705 | 0.993661 |
    | glint360k_cosface_r100_fp16_0.1_IJBB_11 |  0.46962 | 0.926193 | 0.961928 | 0.972444 | 0.981792 | 0.989289 | 0.994904 |
    | partial_fc_glint360k_r100_IJBB_11       | 0.438169 | 0.935443 | 0.962512 | 0.972249 | 0.979649 | 0.987439 | 0.993628 |

    |                                         |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |      AUC |
    |:--------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:| --------:|
    | ms1mv3_arcface_r50_fp16_IJBC_11         | 0.900343 | 0.947947 | 0.964463 | 0.976223 | 0.985427 | 0.991359 | 0.995856 |
    | glint360k_cosface_r50_fp16_0.1_IJBC_11  | 0.914762 | 0.956128 | 0.969832 | 0.979751 | 0.987063 | 0.992841 | 0.996581 |
    | partial_fc_glint360k_r50_IJBC_11        | 0.912001 | 0.956026 | 0.970292 | 0.980212 | 0.986143 | 0.991717 | 0.995828 |
    | ms1mv3_arcface_r100_fp16_IJBC_11        | 0.909802 |  0.95306 | 0.968042 | 0.978831 | 0.985376 | 0.991665 | 0.995768 |
    | glint360k_cosface_r100_fp16_0.1_IJBC_11 | 0.905763 | 0.958787 | 0.973155 | 0.981899 | 0.987319 | 0.992381 | 0.996522 |
    | partial_fc_glint360k_r100_IJBC_11       | 0.877947 | 0.962622 | 0.974689 | 0.981643 |  0.98691 | 0.991768 | 0.995767 |
# insightface scrfd
  ```py
  !pip install -U insightface onnxruntime onnx-simplifier mmcv-full terminaltables pycocotools

  import cv2
  import insightface

  model_file = os.path.expanduser('~/.insightface/models/antelope/scrfd_10g_bnkps.onnx')
  if not os.path.exists(model_file):
      import zipfile
      model_url = 'http://storage.insightface.ai/files/models/antelope.zip'
      zip_file = os.path.expanduser('~/.insightface/models/antelope.zip')
      zip_extract_path = os.path.splitext(zip_file)[0]
      if not os.path.exists(os.path.dirname(zip_file)):
          os.makedirs(os.path.dirname(zip_file))
      insightface.utils.download(model_url, path=zip_file, overwrite=True)
      with zipfile.ZipFile(zip_file) as zf:
          zf.extractall(zip_extract_path)
      os.remove(zip_file)
  dd = insightface.model_zoo.SCRFD(model_file=model_file)
  dd.prepare(-1)

  def show_result(image, bbs, ccs=[], pps=[]):
      plt.figure()
      plt.imshow(image)
      for id, bb in enumerate(bbs):
          plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]], [bb[1], bb[1], bb[3], bb[3], bb[1]])
          if len(ccs) != 0:
              plt.text(bb[0], bb[1], '{:.4f}'.format(ccs[id]))
          if len(pps) != 0:
              pp = pps[id]
              if len(pp.shape) == 2:
                  plt.scatter(pp[:, 0], pp[:, 1], s=8)
              else:
                  plt.scatter(pp[::2], pp[1::2], s=8)
      plt.axis('off')
      plt.tight_layout()

  imm = cv2.imread('../../test_images/Fotos_anuales_del_deporte_de_2012.jpg')
  bcs, pps = dd.detect(imm, input_size=(640, 640))
  bbs, ccs = bcs[:, :4], bcs[:, -1]
  show_result(imm[:, :, ::-1], bbs, ccs, pps)
  ```
  ```sh
  cd detection/scrfd
  python tools/scrfd2onnx.py configs/scrfd/scrfd_500m.py ../../../models/scrfd/SCRFD_500M_KPS.pth --input-img ../../sample-images/t1.jpg
  python tools/scrfd2onnx.py configs/scrfd/scrfd_500m.py ../../../models/scrfd/SCRFD_500M_KPS.pth --input-img ../../sample-images/t1.jpg --shape 640
  ```
  ```py
  import onnx
  import torch
  from mmdet.core import build_model_from_cfg, generate_inputs_and_wrap_model, preprocess_example_input

  checkpoint_path = "../../../models/scrfd/SCRFD_500M_KPS.pth"
  config_path = "configs/scrfd/scrfd_500m.py"
  normalize_cfg = {'mean': [127.5, 127.5, 127.5], 'std': [128.0, 128.0, 128.0]}
  input_config = {"input_shape": (1, 3, 640, 640), "input_path": "../../sample-images/t1.jpg", "normalize_cfg": normalize_cfg}
  model, tensor_data = generate_inputs_and_wrap_model(config_path, checkpoint_path, input_config)

  xx = torch.randn(1, 3, 640, 640)
  print(model([xx])[0][0].shape)  # (2, 5)

  torch.onnx.export(model, tensor_data, "test.onnx", keep_initializers_as_inputs=False, verbose=False)
        opset_version=opset_version,
        output_names=,
    )
  ```
***

# IJB to bins
## Convert tests
  ```py
  import IJB_evals
  import cv2
  templates, medias, p1, p2, label, img_names, landmarks, face_scores = IJB_evals.extract_IJB_data_11('/datasets/IJB_release/', "IJBB")
  print(f"{p1.shape = }, {np.unique(p1).shape = }")
  # p1.shape = (8010270,), np.unique(p1).shape = (1845,)
  print(f"{p2.shape = }, {np.unique(p2).shape = }")
  # p2.shape = (8010270,), np.unique(p2).shape = (10270,)
  print(f"{label.shape = }, {dict(zip(*np.unique(label, return_counts=True))) = }")
  # label.shape = (8010270,), dict(zip(*np.unique(label, return_counts=True))) = {0: 8000000, 1: 10270}
  print(f"{img_names.shape = }, {landmarks.shape = }, {face_scores.shape = }")
  # img_names.shape = (227630,), landmarks.shape = (227630, 5, 2), face_scores.shape = ((227630, 5, 2), (227630,))
  print(f"{templates.shape = }, {medias.shape = }, {np.unique(templates).shape = }")
  # templates.shape = (227630,), medias.shape = (227630,), np.unique(templates).shape = (12115,)
  print(f"{img_names[templates == p1[234]] = }")
  # img_names[templates == p1[234]] = array(['/datasets/IJB_release/IJBB/loose_crop/291.jpg',
  #        '/datasets/IJB_release/IJBB/loose_crop/292.jpg',
  #        '/datasets/IJB_release/IJBB/loose_crop/293.jpg',
  #        '/datasets/IJB_release/IJBB/loose_crop/294.jpg',
  #        '/datasets/IJB_release/IJBB/loose_crop/295.jpg',
  #        '/datasets/IJB_release/IJBB/loose_crop/296.jpg',
  #        '/datasets/IJB_release/IJBB/loose_crop/297.jpg'], dtype='<U48')

  aa = np.load("./IJB_result/TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333_IJBB_11.npz")
  score = aa.get("scores", [])[0]
  print(f"{score.shape = }")
  # score.shape = (8010270,)

  pos = score[label == 1]
  neg = score[label == 0]
  pos_arg_sort = np.argsort(pos)
  neg_arg_sort = np.argsort(neg)
  print(f"{pos[pos_arg_sort][1000: 1010] = }")
  # pos[pos_arg_sort][1000: 1010] = array([0.49577296, 0.49584524, 0.49629315, 0.49635248, 0.49639132, 0.49678287, 0.49680407, 0.49717923, 0.49744828, 0.49750962])
  print(f"{neg[neg_arg_sort][-1000:-990] = }")
  # neg[neg_arg_sort][-1000:-990] = array([0.33684705, 0.33684838, 0.33687524, 0.33694452, 0.33697327, 0.33700055, 0.33704885, 0.33705383, 0.33719019, 0.33723291])

  bins, issame_list = np.load('/datasets/ms1m-retinaface-t1/agedb_30.bin', encoding="bytes", allow_pickle=True)
  print(f"{len(bins) = }, {len(issame_list) = }, {sum(issame_list) = }")
  # len(bins) = 12000, len(issame_list) = 6000, sum(issame_list) = 3000

  pos_num, neg_num = 3000, 3000
  p1_pos, p2_pos = p1[label == 1], p2[label == 1]
  p1_neg, p2_neg = p1[label == 0], p2[label == 0]
  p1_pos_sorted, p2_pos_sorted = p1_pos[pos_arg_sort][: pos_num], p2_pos[pos_arg_sort][: pos_num]
  p1_neg_sorted, p2_neg_sorted = p1_neg[neg_arg_sort][-neg_num :], p2_neg[neg_arg_sort][-neg_num :]
  print(f"{len(set(tuple(zip(p1_pos_sorted, p2_pos_sorted)))) = }, {len(set(tuple(zip(p1_neg_sorted, p2_neg_sorted)))) = }")
  # len(set(tuple(zip(p1_pos_sorted, p2_pos_sorted)))) = 3000, len(set(tuple(zip(p1_neg_sorted, p2_neg_sorted)))) = 3000

  get_ndimgs = lambda id: np.array([IJB_evals.face_align_landmark(cv2.imread(img), landmark) for img, landmark in zip(img_names[templates == id], landmarks[templates == id])])
  plt.imshow(np.hstack(np.vstack([get_ndimgs(p1_pos_sorted[4]), get_ndimgs(p2_pos_sorted[4])])))

  from sklearn.preprocessing import normalize
  mm = keras.models.load_model("checkpoints/TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333.h5")
  p1_emb = normalize(mm((get_ndimgs(p1_pos_sorted[4]) - 127.5) * 0.0078125).numpy())
  p2_emb = normalize(mm((get_ndimgs(p2_pos_sorted[4]) - 127.5) * 0.0078125).numpy())
  print(np.dot(p1_emb, p2_emb.T))
  # [[-0.06832784  0.022935   -0.04413189 -0.03303798 -0.04058807]
  #  [-0.21254557 -0.13249388 -0.16092591 -0.17386593 -0.19386724]]

  aa = np.dot(p1_emb, p2_emb.T)
  p2_len = aa.shape[1]
  p1_idx = aa.argmin() // p2_len
  p2_idx = aa.argmin() % p2_len
  picked_dist = np.dot(p1_emb[p1_idx], p2_emb[p2_idx])
  true_min_dist = aa.min()
  print(f"{picked_dist = }, {true_min_dist = }, {np.allclose(picked_dist, true_min_dist) = }")
  # picked_dist = -0.21254553, true_min_dist = -0.21254557, np.allclose(picked_dist, true_min_dist) = True

  from tqdm import tqdm
  def get_emb_by_template_id(tid):
      p_images = img_names[templates == tid]
      p_landmarks = landmarks[templates == tid]
      p_ndimgs = np.array([IJB_evals.face_align_landmark(cv2.imread(img), landmark) for img, landmark in zip(p_images, p_landmarks)])
      p_emb = normalize(mm((p_ndimgs - 127.5) * 0.0078125).numpy())
      return p_images, p_landmarks, p_emb

  def pick_pos_images(idx):
      p1_images, p1_landmarks, p1_emb = get_emb_by_template_id(p1_pos_sorted[idx])
      p2_images, p2_landmarks, p2_emb = get_emb_by_template_id(p2_pos_sorted[idx])

      aa = np.dot(p1_emb, p2_emb.T)
      p1_idx = aa.argmin() // aa.shape[1]
      p2_idx = aa.argmin() % aa.shape[1]
      picked_dist = np.dot(p1_emb[p1_idx], p2_emb[p2_idx])
      true_min_dist = aa.min()
      return p1_images[p1_idx], p2_images[p2_idx], picked_dist, true_min_dist

  def pick_neg_images(idx):
      n1_images, n1_landmarks, n1_emb = get_emb_by_template_id(p1_neg_sorted[idx])
      n2_images, n2_landmarks, n2_emb = get_emb_by_template_id(p2_neg_sorted[idx])

      aa = np.dot(n1_emb, n2_emb.T)
      n1_idx = aa.argmax() // aa.shape[1]
      n2_idx = aa.argmax() % aa.shape[1]
      picked_dist = np.dot(n1_emb[n1_idx], n2_emb[n2_idx])
      true_min_dist = aa.max()
      return n1_images[n1_idx], n2_images[n2_idx], picked_dist, true_min_dist

  pos_picked = [pick_pos_images(ii) for ii in tqdm(range(pos_num))]
  neg_picked = [pick_neg_images(ii) for ii in tqdm(range(neg_num))]

  print(f"{np.sum([np.allclose(*ii[2:], atol=1e-5) for ii in pos_picked]) = }")
  print(f"{np.sum([np.allclose(*ii[2:], atol=1e-5) for ii in neg_picked]) = }")

  pos_picked_images = np.array(pos_picked)[:, :2]
  neg_picked_images = np.array(neg_picked)[:, :2]
  ```
## Convert function
  ```sh
  MXNET_CUDNN_AUTOTUNE_DEFAULT=0 ipy 1
  ```
  ```py
  import IJB_evals
  import cv2
  import pickle
  from sklearn.preprocessing import normalize
  from tqdm import tqdm

  class Convert_IJB_to_bin:
      def __init__(self, data_path, model_interf, score, subset="IJBB", pos_num=3000, neg_num=3000, save_dest=None, batch_size=64, min_pos=0.2, max_neg=0.5, nfold=10, pick_median=True):
          self.model_interf, self.pos_num, self.neg_num, self.batch_size = model_interf, pos_num, neg_num, batch_size
          self.min_pos, self.max_neg, self.nfold, self.pick_median = min_pos, max_neg, nfold, pick_median
          self.save_dest = save_dest if save_dest is not None else subset + ".bin"

          templates, medias, p1, p2, label, img_names, landmarks, face_scores = IJB_evals.extract_IJB_data_11(data_path, subset)
          self.templates, self.img_names, self.landmarks = templates, img_names, landmarks

          score_filter = np.logical_or(np.logical_and(label == 1, score > min_pos), np.logical_and(label == 0, score < max_neg))
          print(">>>> Filtered size by min_pos and max_neg:", score_filter.sum(), "/", score.shape[0])
          score, label, p1, p2 = score[score_filter], label[score_filter], p1[score_filter], p2[score_filter]

          pos, neg = score[label == 1], score[label == 0]
          pos_arg_sort, neg_arg_sort = np.argsort(pos), np.argsort(neg)
          p1_pos, p2_pos = p1[label == 1], p2[label == 1]
          p1_neg, p2_neg = p1[label == 0], p2[label == 0]
          self.p1_pos_sorted, self.p2_pos_sorted = p1_pos[pos_arg_sort][: pos_num], p2_pos[pos_arg_sort][: pos_num]
          self.p1_neg_sorted, self.p2_neg_sorted = p1_neg[neg_arg_sort][-neg_num :], p2_neg[neg_arg_sort][-neg_num :]

          self.tid_embs = {}

      def get_emb_by_template_id(self, tid):
          p_images = self.img_names[self.templates == tid]
          p_landmarks = self.landmarks[self.templates == tid]
          if tid in self.tid_embs:
              p_embs_norm = self.tid_embs[tid]
          else:
              p_ndimgs = np.array([IJB_evals.face_align_landmark(cv2.imread(img), landmark) for img, landmark in zip(p_images, p_landmarks)])
              p_embs = [self.model_interf(p_ndimgs[batch_id: batch_id + self.batch_size]) for batch_id in range(0, p_ndimgs.shape[0], self.batch_size)]
              p_embs_norm = normalize(np.vstack(p_embs))
              self.tid_embs[tid] = p_embs_norm
          return p_images, p_landmarks, p_embs_norm

      def pick_pos_image_single(self, idx):
          p1_images, p1_landmarks, p1_emb = self.get_emb_by_template_id(self.p1_pos_sorted[idx])
          p2_images, p2_landmarks, p2_emb = self.get_emb_by_template_id(self.p2_pos_sorted[idx])

          dist = np.dot(p1_emb, p2_emb.T).ravel()

          if self.pick_median:
              pick_idx = np.argsort(dist)[- (dist > self.min_pos).sum() // 2] # median index
          else:
              pick_idx = np.argmin(np.where(dist > self.min_pos, dist, np.ones_like(aa)))  # argmin
          p1_idx = pick_idx // p2_emb.shape[0]
          p2_idx = pick_idx % p2_emb.shape[0]
          picked_dist = np.dot(p1_emb[p1_idx], p2_emb[p2_idx])
          true_min_dist = dist[pick_idx]
          return p1_images[p1_idx], p2_images[p2_idx], p1_landmarks[p1_idx], p2_landmarks[p2_idx], picked_dist, true_min_dist

      def pick_neg_image_single(self, idx):
          n1_images, n1_landmarks, n1_emb = self.get_emb_by_template_id(self.p1_neg_sorted[idx])
          n2_images, n2_landmarks, n2_emb = self.get_emb_by_template_id(self.p2_neg_sorted[idx])

          dist = np.dot(n1_emb, n2_emb.T).ravel()

          if self.pick_median:
              pick_idx = np.argsort(dist)[(dist < self.max_neg).sum() // 2] # median index
          else:
              pick_idx = np.argmax(np.where(dist < self.max_neg, dist, np.zeros_like(aa) - 1)) # argmax
          n1_idx = pick_idx // n2_emb.shape[0]
          n2_idx = pick_idx % n2_emb.shape[0]
          picked_dist = np.dot(n1_emb[n1_idx], n2_emb[n2_idx])
          true_max_dist = dist[pick_idx]
          return n1_images[n1_idx], n2_images[n2_idx], n1_landmarks[n1_idx], n2_landmarks[n2_idx], picked_dist, true_max_dist

      def pick_images(self, pos_or_neg="pos"):
          pick_func = self.pick_pos_image_single if pos_or_neg == "pos" else self.pick_neg_image_single
          total = self.pos_num if pos_or_neg == "pos" else self.neg_num

          picked_images, picked_landmarks, picked_values = [], [], []
          for ii in tqdm(range(total), "Picking " + pos_or_neg):
              images_1, images_2, landmarks_1, landmarks_2, picked_dist, true_dist = pick_func(ii)
              picked_images.append([images_1, images_2])
              picked_landmarks.append([landmarks_1, landmarks_2])
              picked_values.append([picked_dist, true_dist])

          values_close_count = np.sum([np.allclose(*ii, atol=1e-5) for ii in picked_values])
          print("Check values close: {} / {}".format(values_close_count, total))
          picked_images, picked_landmarks, picked_values = np.array(picked_images), np.array(picked_landmarks), np.array(picked_values)
          return picked_images, picked_landmarks, picked_values

      def convert(self):
          pos_picked_images, pos_picked_landmarks, pos_picked_values = self.pick_images(pos_or_neg="pos")
          neg_picked_images, neg_picked_landmarks, neg_picked_values = self.pick_images(pos_or_neg="neg")
          picked_images = np.vstack([pos_picked_images, neg_picked_images])
          picked_landmarks = np.vstack([pos_picked_landmarks, neg_picked_landmarks])
          picked_values = np.vstack([pos_picked_values, neg_picked_values])

          bins = []
          for image_2, landmark_2 in tqdm(zip(picked_images, picked_landmarks), "Creating bins", total=picked_images.shape[0]):
              bins.append(tf.image.encode_png(IJB_evals.face_align_landmark(cv2.imread(image_2[0]), landmark_2[0])).numpy())
              bins.append(tf.image.encode_png(IJB_evals.face_align_landmark(cv2.imread(image_2[1]), landmark_2[1])).numpy())

          if self.nfold > 1:
              """ nfold """
              pos_fold, neg_fold = self.pos_num // self.nfold, self.neg_num // self.nfold
              issame_list = ([True] * pos_fold + [False] * neg_fold) * self.nfold
              pos_bin_fold = lambda ii: bins[ii * pos_fold * 2: (ii + 1) * pos_fold * 2]
              neg_bin_fold = lambda ii: bins[self.pos_num * 2 :][ii * neg_fold * 2: (ii + 1) * neg_fold * 2]
              bins = [pos_bin_fold(ii) + neg_bin_fold(ii) for ii in range(self.nfold)]
              bins = np.ravel(bins).tolist()
          else:
              issame_list = [True] * self.pos_num + [False] * self.neg_num

          print("Saving to %s" % self.save_dest)
          with open(self.save_dest, "wb") as ff:
              pickle.dump([bins, issame_list], ff)

          tt, ff = np.sort(picked_values[:self.pos_num, 0]), np.sort(picked_values[self.pos_num:, 0])
          t_steps = int(0.2 * ff.shape[0])
          acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]]).max()
          print("Accuracy:", acc_count / picked_values.shape[0])
          return picked_images, picked_landmarks, picked_values

  # model_interf = IJB_evals.keras_model_interf("checkpoints/TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333.h5")
  # scores = np.load("IJB_result/TT_r50_swish_E_arc_emb512_dr04_sgd_l2_5e4_bs1024_ms1m_cleaned_bnm09_bne1e4_cos16_basic_agedb_30_epoch_48_0.984333_IJBB_11.npz")
  model_interf = IJB_evals.Mxnet_model_interf("../models/partial_fc/mxnet/glint360k_r100FC_1.0_fp16_cosface8GPU/model,0")
  scores = np.load("IJB_result/glint360k_r100FC_1.0_fp16_cosface8GPU_model_IJBB.npz")
  score = scores.get("scores", [])[0]

  aa = Convert_IJB_to_bin("/datasets/IJB_release/", model_interf, score, pos_num=3000, neg_num=3000, save_dest="IJBB.bin")
  picked_images, picked_landmarks, picked_values = aa.convert()

  # model_interf = IJB_evals.Mxnet_model_interf("../models/partial_fc/mxnet/glint360k_r100FC_1.0_fp16_cosface8GPU/model,0")
  scores = np.load("IJB_result/glint360k_r100FC_1.0_fp16_cosface8GPU_IJBC.npz")
  score = scores.get("scores", [])[0]
  bb = Convert_IJB_to_bin("/datasets/IJB_release/", model_interf, score, subset="IJBC", pos_num=3000, neg_num=3000, save_dest="IJBC.bin")
  picked_images, picked_landmarks, picked_landmarks, picked_values = bb.convert()

  import evals
  evals.eval_callback(lambda imm: model_interf(imm.numpy() * 128 + 127.5), "IJBB.bin", flip=False).on_epoch_end()

  bins, issame_list = np.load("IJBB.bin", encoding="bytes", allow_pickle=True)
  plt.imshow(np.vstack([np.hstack([tf.image.decode_jpeg(jj, channels=3).numpy() for jj in bins[ii * 16: (ii + 1) * 16]]) for ii in range(8)]))
  plt.imshow(np.vstack([np.hstack([tf.image.decode_jpeg(jj, channels=3).numpy() for jj in bins[ii * 16 - 129: (ii + 1) * 16 - 129]]) for ii in range(8)]))
  ```
  ```py
  cc = Convert_IJB_to_bin("/datasets/IJB_release/", model_interf, score, subset="IJBC", pos_num=3000, neg_num=3000, min_pos=0.2, max_neg=0.5, nfold=10, pick_median=True, save_dest="IJBC.bin")
  cc.tid_embs = bb.tid_embs
  picked_images, picked_landmarks, picked_values = cc.convert()
  ```
***

# Fuse Conv2D and BatchNorm
## Basic fuse layer test
  ```py
  def fuse_conv_bn(conv_layer, bn_layer):
      # BatchNormalization returns: gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta
      # --> conv_w_new = gamma * conv_w / np.sqrt(var + epsilon)
      # --> conv_b_new = gamma * (conv_b - mean) / sqrt(var + epsilon) + beta
      batch_std = tf.sqrt(bn_layer.moving_variance + bn_layer.epsilon)
      if isinstance(conv_layer, keras.layers.DepthwiseConv2D):
          ww = tf.transpose(conv_layer.depthwise_kernel, [0, 1, 3, 2]) * bn_layer.gamma / batch_std
          ww = tf.transpose(ww, [0, 1, 3, 2])
      else:
          ww = conv_layer.kernel * bn_layer.gamma / batch_std

      if conv_layer.use_bias:
          bias = bn_layer.gamma * (conv_layer.bias - bn_layer.moving_mean) / batch_std + bn_layer.beta
      else:
          bias = bn_layer.gamma * (-1 * bn_layer.moving_mean) / batch_std + bn_layer.beta

      cc = conv_layer.get_config()
      cc['use_bias'] = True
      fused_conv_bn = conv_layer.__class__.from_config(cc)
      fused_conv_bn.build(conv_layer.input_shape)
      fused_conv_bn.set_weights([ww, bias])
      return fused_conv_bn

  input_shape = (224, 224, 3)
  mm = keras.models.Sequential([
      keras.layers.InputLayer(input_shape),
      keras.layers.Conv2D(64, 7, use_bias=False),
      keras.layers.BatchNormalization(axis=-1),
  ])
  # Random set BatchNormalization weights
  mm.layers[1].set_weights([tf.random.uniform(ii.shape) for ii in mm.layers[1].get_weights()])

  inputs = tf.ones([1, * input_shape])
  orign_out = mm(inputs)

  conv_layer, bn_layer = mm.layers[0], mm.layers[1]
  fused_conv_bn = fuse_conv_bn(conv_layer, bn_layer)
  fused_out = fused_conv_bn(inputs)
  print("allclose:", np.allclose(orign_out.numpy(), fused_out.numpy(), atol=1e-7))
  # allclose: True
  ```
  ```py
  input_shape = (56, 56, 64)
  mm = keras.models.Sequential([
      keras.layers.InputLayer(input_shape),
      keras.layers.DepthwiseConv2D((7, 7), use_bias=False),
      keras.layers.BatchNormalization(axis=-1),
  ])
  # Random set BatchNormalization weights
  mm.layers[1].set_weights([tf.random.uniform(ii.shape) for ii in mm.layers[1].get_weights()])

  inputs = tf.ones([1, * input_shape])
  orign_out = mm(inputs)

  conv_layer, bn_layer = mm.layers[0], mm.layers[1]
  fused_conv_bn = fuse_conv_bn(conv_layer, bn_layer)
  fused_out = fused_conv_bn(inputs)
  print("allclose:", np.allclose(orign_out.numpy(), fused_out.numpy(), atol=1e-7))
  # allclose: True
  ```
## Fuse layers in model
  ```py
  import json

  def convert_to_fused_conv_bn_model(model):
      """ Check bn layers with conv layer input """
      model_config = json.loads(model.to_json())
      ee = {layer['name']: layer for layer in model_config['config']['layers']}
      fuse_convs, fuse_bns = [], []
      conv_names = ["Conv2D", "DepthwiseConv2D"]
      for layer in model_config['config']['layers']:
          if layer['class_name'] == "BatchNormalization" and len(layer["inbound_nodes"]) == 1:
              input_node = layer["inbound_nodes"][0][0]
              if isinstance(input_node, list) and ee.get(input_node[0], {"class_name": None})['class_name'] in conv_names:
                  fuse_convs.append(input_node[0])
                  fuse_bns.append(layer['name'])
      print(f">>>> {len(fuse_convs) = }, {len(fuse_bns) = }")
      # len(fuse_convs) = 53, len(fuse_bns) = 53

      """ Create new model config """
      layers = []
      fused_bn_dict = dict(zip(fuse_bns, fuse_convs))
      fused_conv_dict = dict(zip(fuse_convs, fuse_bns))
      for layer in model_config['config']['layers']:
          if layer["name"] in fuse_convs:
              print(">>>> Fuse conv bn:", layer["name"])
              layer["config"]["use_bias"] = True
          elif layer["name"] in fuse_bns:
              continue

          if len(layer["inbound_nodes"]) != 0:
              for ii in layer["inbound_nodes"][0]:
                  if isinstance(ii, list) and ii[0] in fused_bn_dict:
                      print(">>>> Replace inbound_nodes: {}, {} --> {}".format(layer["name"], ii[0], fused_bn_dict[ii[0]]))
                      ii[0] = fused_bn_dict[ii[0]]
          layers.append(layer)
      model_config['config']['layers'] = layers
      new_model = keras.models.model_from_json(json.dumps(model_config))

      """ New model set layer weights by layer names """
      for layer in new_model.layers:
          if layer.name in fuse_bns:  # This should not happen
              continue

          orign_layer = model.get_layer(layer.name)
          if layer.name in fused_conv_dict:
              orign_bn_layer = model.get_layer(fused_conv_dict[layer.name])
              print(">>>> Fuse conv bn", layer.name, orign_bn_layer.name)
              conv_bn = fuse_conv_bn(orign_layer, orign_bn_layer)
              layer.set_weights(conv_bn.get_weights())
          else:
              layer.set_weights(orign_layer.get_weights())
      return new_model

  """ Verification """
  model = keras.applications.ResNet50(input_shape=(224, 224, 3))
  new_model = convert_to_fused_conv_bn_model(model)

  inputs = tf.ones((1, *model.input_shape[1:]))
  orign_out = model(inputs).numpy()
  fused_out = new_model(inputs).numpy()
  print(f'{np.allclose(orign_out, fused_out, atol=1e-9) = }')
  # np.allclose(orign_out, fused_out, atol=1e-9) = True

  %timeit model(inputs)
  # 69.6 ms ± 209 µs per loop (mean ± std. dev. of 7 runs, 10 loops each) # CPU
  # 29.7 ms ± 172 µs per loop (mean ± std. dev. of 7 runs, 10 loops each) # GPU
  %timeit new_model(inputs)
  # 49.7 ms ± 185 µs per loop (mean ± std. dev. of 7 runs, 10 loops each) # CPU
  # 16.8 ms ± 126 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) # GPU
  ```
  ```py
  import models
  model = keras.models.load_model('./checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_cleaned_bnm09_bne1e4_cos16_batch_float16_basic_agedb_30_epoch_50_0.976333.h5')
  model = models.convert_mixed_float16_to_float32(model)  # Don't use float16 when converting
  new_model = convert_to_fused_conv_bn_model(model)

  inputs = tf.ones((1, *model.input_shape[1:]))
  orign_out = model(inputs).numpy()
  fused_out = new_model(inputs).numpy()
  print(f'{np.allclose(orign_out, fused_out, atol=1e-5) = }')
  # np.allclose(orign_out, fused_out, atol=1e-5) = True

  %timeit model(inputs)
  # 47.6 ms ± 240 µs per loop (mean ± std. dev. of 7 runs, 10 loops each) # GPU
  %timeit new_model(inputs)
  # 35.8 ms ± 278 µs per loop (mean ± std. dev. of 7 runs, 10 loops each) # GPU
  ```
***

# VOLO
  ```py
  sys.path.append('../Keras_volo')
  import volo
  mm = volo.volo_d1(input_shape=(112, 112, 3), num_classes=512, survivals=0.8, mean_classifier_top=True)
  mm.load_weights('../models/volo/d1_224_84.2.h5', by_name=True, skip_mismatch=True)
  bb = keras.models.load_model('../models/volo/d1_224_84.2.h5')
  mm.get_layer('stack_0_positional').load_resized_pos_emb(bb.get_layer('stack_0_positional'))

  bb = keras.models.Model(mm.inputs[0], keras.layers.Activation("linear", dtype="float32", name="embedding")(mm.output))
  bb.save('../models/volo/d1_112_GAP_512_SD_08_84.2.h5')
  ```
  ```py
  sys.path.append('../Keras_volo')
  import volo
  mm = volo.volo_d2(input_shape=(112, 112, 3), num_classes=512, survivals=0.8, mean_classifier_top=True)
  mm.load_weights('../models/volo/d2_384_86.0.h5', by_name=True, skip_mismatch=True)
  bb = keras.models.load_model('../models/volo/d2_384_86.0.h5')
  mm.get_layer('stack_0_positional').load_resized_pos_emb(bb.get_layer('stack_0_positional'))

  bb = keras.models.Model(mm.inputs[0], keras.layers.Activation("linear", dtype="float32", name="embedding")(mm.output))
  bb.save('../models/volo/d2_112_GAP_512_SD_08_86.0.h5')
  ```
  ```py
  import losses, train, models
  import tensorflow_addons as tfa
  keras.mixed_precision.set_global_policy("mixed_float16")

  data_basic_path = '/datasets/ms1m-retinaface-t1'
  data_path = data_basic_path + '_112x112_folders'
  eval_paths = ["IJBB.bin", "IJBC.bin"] + [os.path.join(data_basic_path, ii) for ii in ['lfw.bin', 'cfp_fp.bin', 'agedb_30.bin']]

  sys.path.append('../Keras_volo/')
  import volo
  basic_model = keras.models.load_model('../models/volo/d2_112_GAP_512_SD_08_86.0.h5')

  tt = train.Train(data_path, eval_paths=eval_paths,
      save_path='TT_volo_d2_GAP_SD_08_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16.h5',
      basic_model=basic_model, model=None, lr_base=0.2, lr_decay=0.5, lr_decay_steps=16, lr_min=1e-4,
      batch_size=512, random_status=100, output_weight_decay=1)

  optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
  sch = [
      {"loss": losses.ArcfaceLoss(scale=16), "epoch": 5, "optimizer": optimizer},
      {"loss": losses.ArcfaceLoss(scale=32), "epoch": 5},
      {"loss": losses.ArcfaceLoss(scale=64), "epoch": 40},
  ]
  tt.train(sch, 0)
  ```
***

# AotNet
  ```py
  from keras_cv_attention_models import aotnet
  # resnet50 like
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0)

  # se_ir_resnet50 like
  mm = aotnet.AotNet(num_blocks=[3, 4, 14, 3], expansion=1, input_shape=(112, 112, 3), deep_stem=False, num_classes=0, se_ratio=1/4, stem_down_sample=False, strides=2)

  # Basic
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types=None, deep_stem=True, strides=1)

  # BotNet like
  attn = [None, None, "mhsa", "mhsa"]
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types=attn, deep_stem=True, strides=1)

  # HaloNet like
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types="halo", deep_stem=True, strides=1)

  # Resnest50 like
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types="sa", deep_stem=True, strides=1)

  # Cotnet50 like
  mm = aotnet.AotNet50(inp也是ut_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types="cot", deep_stem=True, strides=1)

  # SECotnetD50 like
  attn = ["sa", "sa", ["cot", "sa"] * 50, "cot"]
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types=attn, deep_stem=True, strides=1)

  # Mixing se and outlook and halo and mhsa
  attn = [None, "outlook", ["mhsa", "halo"] * 50, ["mhsa", "halo"] * 4]
  se_ratio = [0.25, 0.25, 0, 0]
  mm = aotnet.AotNet50(input_shape=(112, 112, 3), num_classes=0, activation='relu', attn_types=attn, se_ratio=se_ratio, deep_stem=True)
  ```
  ```py
  # Most se_ir_resnet50 like
  # first conv, kernel_size=3, strides=1
  # se_module, use_bias=False
  mm = aotnet.AotNet(num_blocks=[3, 4, 14, 3], stack=aotnet.stack1, preact=True, stack_strides=[2, 2, 2, 2], input_shape=(112, 112, 3), num_classes=0, se_ratio=1/16, expansion=1, stem_downsample=False)
  ```
***

# Distillation
  - [PDF 2106.05237 Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/pdf/2106.05237.pdf)
  ```py
  import models, data
  mm = keras.models.load_model('checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_M5_sgd_LA.h5', custom_objects={'NormDense': models.NormDense}, compile=False)
  ds, steps_per_epoch = da也是ta.prepare_dataset('faces_emore_test', batch_size=16)
  aa, bb = ds.as_numpy_iterator().next()
  cc = mm(aa).numpy()
  dd = np.argsort(cc, axis=-1)[:, -10:]
  ss = np.array([ii[jj] for ii, jj in zip(cc, dd)])
  ```
  ```py
  class DistillerLoss(keras.losses.Loss):
      def __init__(self, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=10, **kwargs):
          super(DistillerLoss, self).__init__(**kwargs)
          self.student_loss_fn, self.distillation_loss_fn = student_loss_fn, distillation_loss_fn
          self.alpha, self.temperature = alpha, temperature

      def call(self, y_true, y_pred):
          student_output, teacher_output = tf.split(y_pred, 2, axis=-1)
          student_loss = self.student_loss_fn(y_true, student_output)
          distillation_loss = self.distillation_loss_fn(
              tf.nn.softmax(teacher_output / self.temperature, axis=1),
              tf.nn.softmax(student_output / self.temperature, axis=1),
          )
          loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
          return loss

  def distiller_accuracy(y_true, y_pred):
      student_output, _ = tf.split(y_pred, 2, axis=-1)
      return keras.metrics.sparse_categorical_accuracy(y_true, student_output)

  distiller_loss = DistillerLoss(
      student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      distillation_loss_fn=keras.losses.KLDivergence(),
      alpha=0.1,
      # temperature=100,
      temperature=10,
  )
  ```
  ```py
  # Reference: https://www.tutorialexample.com/implement-kl-divergence-loss-in-tensorflow-tensorflow-tutorial/
  def kl_divergence(true_p, pred_p):
      true_prob = tf.nn.softmax(true_p, axis = 1)
      loss_1 = -tf.nn.softmax_cross_entropy_with_logits(logits=true_p, labels=true_prob)
      loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=pred_p, labels=true_prob)
      loss = loss_1 + loss_2
      return loss

  distillation_loss = self.distillation_loss_fn(
      teacher_predictions / self.temperature,
      student_predictions / self.temperature
  )
  ```
***
