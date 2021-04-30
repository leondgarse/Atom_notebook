# ___2021 - 04 - 07 Keras Insightface Advance___
***

# 目录
***

# BotNet
  - [aravindsrinivas/botnet.py](https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2)
  - [leondgarse/botnet.py](https://gist.github.com/leondgarse/351dba9457c5a36516aea3ce1950ac74)
  - **botnet MHSA**
    ```py
    from icecream import ic
    inputs = keras.layers.Input([14, 16, 1024])
    featuremap = inputs

    print(botnet.MHSA(featuremap, 512, pos_enc_type='relative', heads=4).shape)
    # (None, 14, 16, 512)

    q = botnet.group_pointwise(featuremap, proj_factor=1, name='q_proj', heads=4, target_dimension=512)
    k = botnet.group_pointwise(featuremap, proj_factor=1, name='k_proj', heads=4, target_dimension=512)
    v = botnet.group_pointwise(featuremap, proj_factor=1, name='v_proj', heads=4, target_dimension=512)

    ic(q.shape.as_list(), k.shape.as_list(), v.shape.as_list())
    # q.shape.as_list(): [None, 4, 14, 16, 128]
    print(botnet.relpos_self_attention(q=q, k=k, v=v, relative=True, fold_heads=True).shape)
    # (None, 14, 16, 512)

    relative, fold_heads = True, True
    bs, heads, h, w, dim = q.shape
    int_dim = int(dim)
    q = q * (dim ** -0.5) # scaled dot-product
    logits = tf.einsum('bhHWd,bhPQd->bhHWPQ', q, k)
    if relative:
        logits += botnet.relative_logits(q)
    # weights = tf.reshape(logits, [-1, heads, h, w, h * w])
    # weights = tf.nn.softmax(weights)
    # weights = tf.reshape(weights, [-1, heads, h, w, h, w])
    weights = tf.nn.softmax(logits)
    attn_out = tf.einsum('bhHWPQ,bhPQd->bHWhd', weights, v)
    if fold_heads:
        attn_out = tf.reshape(attn_out, [-1, h, w, heads * dim])
    ic(attn_out.shape.as_list())
    # ic| attn_out.shape.as_list(): [None, 14, 16, 512]
    ```
  - **relative_logits**
    ```py
    def rel_to_abs(x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, h, w, 2*w - 1]
        Output: [bs, heads, h, w, w]
        """
        bs, heads, h, w, dim = x.shape
        col_pad = tf.zeros_like(x[:, :, :, :, :1], dtype=x.dtype)
        x = tf.concat([x, col_pad], axis=-1)
        flat_x = tf.reshape(x, [-1, heads, h, w * 2 * w])
        flat_pad = tf.zeros_like(flat_x[:, :, :, :w-1], dtype=x.dtype)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=-1)
        final_x = tf.reshape(flat_x_padded, [-1, heads, h, w+1, 2*w-1])
        final_x = final_x[:, :, :, :w, w-1:]
        return final_x


    def relative_logits_1d(*, q, rel_k, transpose_mask):
        """
        Compute relative logits along one dimenion.
        `q`: [bs, heads, height, width, dim]
        `rel_k`: [2*width - 1, dim]
        """
        bs, heads, h, w, dim = q.shape
        # rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = tf.matmul(q, tf.transpose(rel_k, [1, 0]))
        rel_logits = rel_to_abs(rel_logits)
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        return rel_logits


    def relative_logits(q):
        bs, heads, h, w, dim = q.shape
        stddev = dim ** -0.5
        rel_emb_w = tf.compat.v1.get_variable('r_width', shape=(2*w - 1, dim), dtype=q.dtype, initializer=tf.random_normal_initializer(stddev=stddev))
        rel_logits_w = relative_logits_1d(q=q, rel_k=rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])

        # Relative logits in height dimension.
        rel_emb_h = tf.compat.v1.get_variable('r_height', shape=(2*h - 1, dim), dtype=q.dtype, initializer=tf.random_normal_initializer(stddev=stddev))
        rel_logits_h = relative_logits_1d(q=tf.transpose(q, [0, 1, 3, 2, 4]), rel_k=rel_emb_h, transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w
    ```
    ```py
    aa = tf.convert_to_tensor(np.arange(45).reshape(1, 1, 3, 3, 5))
    rel_to_abs(aa)
    print(aa[0, 0].numpy())
    # [[[ 0  1  2  3  4]
    #   [ 5  6  7  8  9]
    #   [10 11 12 13 14]]
    #  [[15 16 17 18 19]
    #   [20 21 22 23 24]
    #   [25 26 27 28 29]]
    #  [[30 31 32 33 34]
    #   [35 36 37 38 39]
    #   [40 41 42 43 44]]]
    print(rel_to_abs(aa)[0, 0].numpy())
    # [[[ 2  3  4]
    #   [ 6  7  8]
    #   [10 11 12]]
    #  [[17 18 19]
    #   [21 22 23]
    #   [25 26 27]]
    #  [[32 33 34]
    #   [36 37 38]
    #   [40 41 42]]]
    ```
  - **keras.layers.MultiHeadAttention**
    ```py
    from tensorflow.python.ops import math_ops
    from tensorflow.python.ops import special_math_ops
    from icecream import ic
    inputs = keras.layers.Input([14, 16, 1024])

    nn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)
    ic(nn(inputs, inputs).shape.as_list())
    # ic| nn(inputs, inputs).shape.as_list(): [None, 14, 16, 1024]

    query = nn._query_dense(inputs)
    key = nn._key_dense(inputs)
    value = nn._value_dense(inputs)
    ic(query.shape.as_list(), key.shape.as_list(), value.shape.as_list())
    # ic| query.shape.as_list(): [None, 14, 16, 4, 128]

    # attention_output, attention_scores = nn._compute_attention(query, key, value)
    query = math_ops.multiply(query, 1.0 / math.sqrt(float(nn._key_dim)))
    # 'afgde,abcde->adbcfg', 'bhHWd,bhPQd->bhHWPQ' == 'afgde,adbce->afgdbc'
    attention_scores = special_math_ops.einsum(nn._dot_product_equation, key, query)
    ic(attention_scores.shape.as_list())
    # ic| attention_scores.shape.as_list(): [None, 4, 14, 16, 14, 16]

    if relative:
        query = tf.transpose(query, [0, 3, 1, 2, 4])
        attention_scores += relative_logits(query)
    attention_scores = nn._masked_softmax(attention_scores, None)
    attention_scores_dropout = nn._dropout_layer(attention_scores, training=False)
    attention_output = special_math_ops.einsum(nn._combine_equation, attention_scores_dropout, value)
    ic(attention_output.shape.as_list())
    # ic| attention_output.shape.as_list(): [None, 14, 16, 4, 128]

    attention_output = nn._output_dense(attention_output)
    ic(attention_output.shape.as_list())
    # ic| attention_output.shape.as_list(): [None, 14, 16, 1024]
    ```
    ```py
    def rel_to_abs(x):
        bs, heads, h, w, dim = x.shape
        col_pad = tf.zeros_like(x[:, :, :, :, :1], dtype=x.dtype)
        x = tf.concat([x, col_pad], axis=-1)
        flat_x = tf.reshape(x, [-1, heads, h, w * 2 * w])
        flat_pad = tf.zeros_like(flat_x[:, :, :, :w-1], dtype=x.dtype)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=-1)
        final_x = tf.reshape(flat_x_padded, [-1, heads, h, w+1, 2*w-1])
        final_x = final_x[:, :, :, :w, w-1:]
        return final_x

    def relative_logits_1d(*, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = tf.matmul(q, tf.transpose(rel_k, [1, 0]))
        rel_logits = rel_to_abs(rel_logits)
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, h, 1, 1])
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        return rel_logits

    def relative_logits(q):
        rel_logits_w = relative_logits_1d(q=q, rel_k=rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        rel_logits_h = relative_logits_1d(q=tf.transpose(q, [0, 1, 3, 2, 4]), rel_k=rel_emb_h, transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w
    ```
***

# Backbones
## BoTNet
  **BoTNet50 on MS1MV3**
  ```py
  import json
  hist_path = "checkpoints/"
  pp = {}
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [1, 17, 17, 17, 20]
  pp["skip_epochs"] = 2
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([32, 64, 64, 64], [0.1, 0.1, 0.05, 0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json", fig_label="aa", names=names, **pp)

  pp["axes"] = axes
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_hist.json", fig_label="no bias", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_hist.json", fig_label="no bias, shortcut act none, tmul 2", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_randaug_hist.json", fig_label="no bias, shortcut act none, tmul 2, randaug", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_prelu_shortcut_act_none_GDC_arc_emb512_bs768_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json", fig_label="prelu, init 0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json", fig_label="swish, GDC", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json", fig_label="swish, E, use_bias True", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_true_conv_no_bias_tmul_2_random0_hist.json", fig_label="swish, E, use_bias False", **pp)

  hist_path = "checkpoints/"
  aa = [
      hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json",
      hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_hist.json",
      hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_hist.json",
      hist_path + "TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_randaug_hist.json",
      hist_path + "TT_botnet50_prelu_shortcut_act_none_GDC_arc_emb512_bs768_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json",
      hist_path + "TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json",
      hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_hist.json",
      hist_path + "TT_botnet50_swish_shortcut_act_none_E_dr04__arc_emb512_cos16_batch_restart_2_bias_true_conv_no_bias_tmul_2_random0_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_botnet50_"))
  ```
## GhostNet
  **GhostNet on MS1MV3**
  ```py
  import json
  hist_path = "checkpoints/ghostnet_v1/"
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
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json", **pp)
  pp["axes"] = axes


  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json", **pp, limit_loss_max=80)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_hist.json", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_image_4_hist.json", **pp)

  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist.json", **pp)
  # axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json", **pp, fig_label="PReLU, bias_false")

  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="swish, float16")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, float16")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_ipc10_float16_hist.json", **pp, fig_label="swish, float16, ipc10")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_randaug_float16_hist.json", **pp, fig_label="swish, float16, randaug")

  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, init 0.25, float16")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_kam0_float16_hist.json", **pp, fig_label="swish, float16, keep_as_min 0")

  hist_path = "checkpoints/"
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, float16, strides_1")
  axes, _ = plot.hist_plot_split(hist_path + "TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", **pp, fig_label="PReLU, float16, strides_1, se_relu")
  ```
  ```py
  hist_path = "checkpoints/"
  aa = [
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr04_wd5e4_bs512_ms1m_hist.json",
      hist_path + "TT_ghostnet_pointwise_E_arc_emb512_dr04_wd5e4_bs512_ms1m_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_image_4_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist.json",
      hist_path + "TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist.json",
  ]

  _ = choose_accuracy(aa, skip_name_len=len("TT_ghostnet_prelu_GDC_arc_emb512_dr0_"))
  |                                                             |      lfw |   cfp_fp |   agedb_30 |   epoch |
  |:------------------------------------------------------------|---------:|---------:|-----------:|--------:|
  | _wd5e4_bs512_ms1m_hist                                      | 0.995333 | 0.957714 |   0.956    |      47 |
  | 04_wd5e4_bs512_ms1m_hist                                    | 0.995833 | 0.953571 |   0.959    |      45 |
  | sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist  | 0.997167 | 0.959429 |   0.969333 |      45 |
  | sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_hist    | 0.996167 | 0.961286 |   0.966833 |      46 |
  | sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist         | 0.9965   | 0.965    |   0.97     |      48 |
  | sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_epoch_hist         | 0.996833 | 0.962429 |   0.969    |      53 |
  | sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_hist         | 0.997167 | 0.959857 |   0.968667 |      48 |
  | sgd_l2_1e3_bs1024_ms1m_bnm09_bne1e5_cos7_batch_image_4_hist | 0.996333 | 0.959714 |   0.968    |      47 |
  | sgdw_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist          | 0.9965   | 0.957857 |   0.966    |      45 |
  | sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_hist        | 0.996667 | 0.960429 |   0.968667 |      45 |
  ```
## MobileNet
  - **Mobilenet ArcFace and CurricularFace on Emore**
  ```py
  import json
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  # pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 10, 10, 80]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.01, 0.001])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json", **pp, names=names)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_hist.json", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_batch_hist.json", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_l2_5e4_bs512_emore_sgdw_scale_true_bias_true_cos7_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgdw_scale_true_bias_true_cos7_hist.json", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_batch_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist.json", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_false_bias_true_cos16_batch_hist.json", **pp)
  ```
  ```py
  hist_path = "checkpoints/"
  aa = [
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr04_wd5e4_bs512_emore_sgdw_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_wd5e4_bs512_emore_sgdw_scale_true_bias_true_cos30_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr04_l2_5e4_bs512_emore_sgdw_scale_true_bias_true_cos7_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgdw_scale_true_bias_true_cos7_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_curricular_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos7_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_batch_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_hist.json",
      hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_false_bias_true_cos16_batch_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_mobilenet_GDC_emb512_"))
  ```
  - **Mobilenet randaug**
  ```py
  import json
  hist_path = "checkpoints/"
  pp = {}
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  # pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 7, 33, 80]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.05, 0.025])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json", **pp, names=names, fig_label="scale_true_bias_false, random_0")
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_2_hist.json", **pp, fig_label="scale_true_bias_false, randaug_100")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json", **pp, fig_label="scale_true_bias_false, random_0, ipc_10")

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_false_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json", **pp, fig_label="scale_false_bias_true")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json", **pp, fig_label="scale_true_bias_true, random_0")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_randaug_lrmin_1e4_tmul_2_ipc_10_hist.json", **pp, fig_label="scale_true_bias_true, randaug, ipc_10")
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json", **pp, fig_label="scale_true_bias_true, random_0, ipc_10")


  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_randaug_lrmin_1e4_tmul_2_ipc_10_hist.json 0.995000 | 0.942571 | 0.953833
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_false_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json

  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_tmul_2_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_ipc_10_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_hist.json

  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_E51_randaug_100_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_E51_random_0_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_E51_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_randaug_100_lrmin_1e4_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_randaug_300_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_randaug_300_hist.json

  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_random2_E16_arc_hist.json
  TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_E48_arc_trip_64_E16_arc_hist.json
  ```
  - **Mobilenet CurricularFace on CASIA**
  ```py
  hist_path = "checkpoints/mobilenet_casia_tests/"
  pp = {}
  pp["epochs"] = [5, 5, 10, 10, 40]
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "lr"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64, 64], [0.1, 0.1, 0.1, 0.01, 0.001])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_base_bs400_hist.json", names=names, **pp)
  pp["axes"] = axes

  hist_path = "checkpoints/"
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb256_E_curricular_bs400_scale_false_usebias_true_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb256_E_curricular_bs400_scale_true_usebias_true_hist.json", **pp)
  ```
  - **Mobilenet scale and use_bias on CASIA**
  ```py
  hist_path = "checkpoints/"
  pp = {}
  pp["epochs"] = [5, 5, 30]
  pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "lr"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  names = ["ArcFace Scale 16", "ArcFace Scale 32", "ArcFace Scale 64"]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_false_bias_false_cos30_casia_hist.json", names=names, **pp)
  pp["axes"] = axes

  hist_path = "checkpoints/"
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_false_bias_true_cos30_casia_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_true_bias_false_cos30_casia_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_true_bias_true_cos30_casia_hist.json", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_emb512_GDC_dr04_curricular_bs512_scale_true_bias_true_cos30_casia_hist.json", **pp)

  hist_path = "checkpoints/"
  aa = [
      hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_false_bias_false_cos30_casia_hist.json",
      hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_false_bias_true_cos30_casia_hist.json",
      hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_true_bias_false_cos30_casia_hist.json",
      hist_path + "TT_mobilenet_emb512_GDC_dr04_arc_bs512_scale_true_bias_true_cos30_casia_hist.json",
  ]
  _ = choose_accuracy(aa, skip_name_len=len("TT_mobilenet_emb512_GDC_dr04_arc_bs512_"))
  ```
## MobileNet swish PReLU
  ```py
  import json
  hist_path = "checkpoints/"
  pp = {}
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  # pp["customs"] = ["cfp_fp", "agedb_30", "lfw", "triplet_embedding_loss", "lr", "arcface_loss", "regular_loss"]
  # pp["customs"] = plot.EVALS_NAME + [ii+"_thresh" for ii in plot.EVALS_NAME]
  pp["customs"] = plot.EVALS_NAME + ['lr']
  pp["epochs"] = [5, 5, 7, 33]
  pp["skip_epochs"] = 0
  names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="basic relu", names=names, **pp)
  pp["axes"] = axes

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_true_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="relu scale_true_bias_true", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_prelu_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="PReLU", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GDC_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="swish GDC", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_E_emb512_arc_dr04_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="swish E", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GAP_emb512_arc_dr0_l2_5e4_bs512_emore_sgd_scale_true_bias_false_cos16_batch_random_0_lrmin_1e4_tmul_2_float16_hist.json", fig_label="swish GAP", **pp)

  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist.json", fig_label="swish ms1m, dr0", **pp)
  axes, _ = plot.hist_plot_split(hist_path + "TT_mobilenet_swish_GDC_arc_emb512_dr4_sgd_l2_5e4_bs512_ms1m_bnm09_bne1e4_cos16_batch_fixed_float16_hist.json", fig_label="swish ms1m, dr4", **pp)
  ```
## Resnet
```py
import json
hist_path = "checkpoints/"
pp = {}
pp["customs"] = plot.EVALS_NAME + ['lr']
pp["epochs"] = [5, 5, 7, 33]
pp["skip_epochs"] = 2
names = ["ArcFace Scale %d, learning rate %g" %(ss, lr) for ss, lr in zip([16, 32, 64, 64], [0.1, 0.1, 0.1, 0.05])]
axes, _ = plot.hist_plot_split(hist_path + "TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_hist.json", fig_label="botnet50 relu", names=names, **pp)
pp["axes"] = axes

axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_hist.json", fig_label="resnet101v2 10 warmup", **pp)
axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_2_hist.json", fig_label="resnet101v2 1 warmup", **pp)

axes, _ = plot.hist_plot_split(hist_path + "TT_resnet101v2_pad_same_conv_no_bias_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e4_cos16_float16_hist.json", fig_label="resnet101v2 conv_no_bias", **pp)
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

# IJB Results
|                                                                      |        1e-06 |        1e-05 |       0.0001 |        0.001 |         0.01 |          0.1 |
|:-------------------------------------------------------------------- | ------------:| ------------:| ------------:| ------------:| ------------:| ------------:|
| MS1MV2-ResNet100-Arcface_IJBB_N0D1F1                                 |      0.42814 |     0.908179 |     0.948978 |     0.964654 |     0.976728 |     0.986563 |
| r100-arcface-msfdrop75_IJBB                                          |     0.441772 |     0.905063 |     0.949464 |     0.965823 |     0.978578 |     0.988802 |
| glint360k_r100FC_1.0_fp16_cosface8GPU_model_IJBB                     |     0.460857 | **0.938364** | **0.962317** |     0.970789 |      0.98111 |     0.988023 |
| glint360k_r100FC_1.0_fp16_cosface8GPU_model_average_IJBB             | **0.464849** |     0.937001 |      0.96222 |     0.970789 |     0.981597 |     0.988023 |
| glint360k_r100FC_0.1_fp16_cosface8GPU_model_IJBB                     |     0.450536 |     0.931938 |     0.961928 |     0.972639 |     0.981986 |     0.989679 |
| glint360k_r100FC_0.1_fp16_cosface8GPU_model_average_IJBB             |      0.44742 |     0.932619 |     0.961831 | **0.972833** | **0.982278** | **0.989971** |
| GhostNet_x1.3_Arcface_Epoch_24_IJBB                                  |     0.352678 |     0.881694 |     0.928724 |     0.954041 |     0.972055 |     0.985784 |

|                                                                                                                                                                            |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
| TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_basic_agedb_30_epoch_15_0.970000_IJBB_11                         | 0.344888 | 0.864752 | 0.920156 | 0.951315 | 0.969912 | 0.984129 |
| TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_E48_arc_trip_basic_agedb_30_epoch_15_batch_4000_0.971500_IJBB_11 |  0.39075 | 0.822785 | 0.911879 | 0.951607 | 0.974586 | 0.988413 |
| TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_E48_arc_trip_E17_arc_basic_agedb_30_epoch_4_0.970333_IJBB_11     | 0.372055 | 0.849757 | 0.920351 | 0.952386 |  0.97186 | 0.986076 |
| TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_49_batch_2000_0.970000_IJBB_11                     | 0.337001 | 0.856573 | 0.922006 |  0.95297 |  0.97147 | 0.983544 |
| TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_LH_sgd_lr25e3_float16_basic_agedb_30_epoch_14_batch_4000_0.971000_IJBB_11         | 0.342162 | 0.863778 | 0.923466 | 0.953749 | 0.970886 |  0.98296 |
| TT_ghostnet_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_E50_sgd_lr25e3_float16_basic_agedb_30_epoch_14_batch_2000_0.971000_IJBB_11            | 0.346641 | 0.857157 | 0.922687 | 0.954041 | 0.970691 | 0.983155 |
| TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_47_0.980000_IJBB_11                   | 0.360467 | 0.879065 | 0.931159 | 0.957644 |  0.97186 | 0.985102 |
| TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_50_0.976333_IJBB_11           | 0.327069 | 0.887244 | 0.932425 |  0.95482 | 0.972541 |  0.98481 |
| TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_basic_agedb_30_epoch_16_0.978167_IJBB_11                          | 0.360759 | 0.899318 | 0.941967 | 0.960273 | 0.972444 | 0.984129 |
| TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_basic_agedb_30_epoch_48_batch_4000_0.978833_IJBB_11  | 0.317235 | 0.896981 | 0.941675 | 0.960954 | 0.971762 | 0.983836 |
| TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_conv_no_bias_basic_agedb_30_epoch_50_batch_2000_0.979000_IJBB_11  | 0.313632 | 0.898832 | 0.941675 | 0.960759 | 0.971957 | 0.984031 |
| TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_randaug_basic_agedb_30_epoch_47_0.979333_IJBB_11                    | 0.381694 | 0.894547 | 0.941967 | 0.963875 | 0.975657 | 0.984615 |
| TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_basic_agedb_30_epoch_48_0.979667_IJBB_11                            | 0.384226 |  0.89591 | 0.940019 | 0.958325 | 0.973126 | 0.984323 |
| TT_botnet50_swish_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_2_bias_false_conv_no_bias_tmul_2_random0_basic_agedb_30_epoch_45_batch_4000_0.980167_IJBB_11        | 0.349172 | 0.904284 | 0.944693 | 0.962707 | 0.974878 | 0.983739 |
| TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_46_batch_2000_0.978833_IJBB_11               | 0.395618 | 0.897955 | 0.943622 | 0.962707 | 0.973515 | 0.983544 |
| TT_resnet101v2_pad_same_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_2_basic_agedb_30_epoch_50_IJBB_11                                 | 0.369815 | 0.895618 | 0.944109 | 0.961733 | 0.973612 | 0.983642 |


|                                                                                                                                                                  |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------:| --------:| --------:| --------:| --------:| --------:|
| glint360k_r100FC_1.0_fp16_cosface8GPU_IJBC                                                                                                                       | 0.872066 | 0.961497 | 0.973871 | 0.980672 | 0.987421 | 0.991819 |
| GhostNet_x1.3_Arcface_Epoch_24_IJBC_11                                                                                                                           | 0.876259 | 0.922023 | 0.945748 |  0.96477 | 0.978985 | 0.990336 |
| TT_mobilenet_pointwise_distill_128_emb512_dr04_arc_bs400_r100_glint_fp16_basic_agedb_30_epoch_26_batch_10000_0.972500_IJBC_11                                    | 0.716981 | 0.886077 | 0.938743 | 0.960986 | 0.977604 | 0.987933 |
| TT_mobilenet_pointwise_distill_128_emb512_dr04_arc_bs400_r100_glint_fp16_basic_agedb_30_epoch_22_batch_10000_0.972000_IJBC_11                                    | 0.774608 | 0.890985 | 0.939357 | 0.960986 | 0.977144 | 0.987779 |
| keras_mobilenet_emore_adamw_5e5_soft_baseline_before_arc_E80_BTO_E2_arc_sgdw_basic_agedb_30_epoch_119_0.959333_IJBC_11                                           | 0.741423 | 0.848699 | 0.911745 | 0.951629 | 0.974127 |  0.98737 |
| TT_mobilenet_pointwise_distill_128_arc_emb512_dr04_wd5e4_bs400_r100_ms1m_fp16_basic_agedb_30_epoch_45_0.972833_IJBC_11                                           | 0.848545 | 0.896457 | 0.935573 | 0.961651 | 0.977706 | 0.990438 |
| TT_ghostnet_prelu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_basic_agedb_30_epoch_46_0.969333_IJBC_11                              | 0.817406 | 0.889656 | 0.934499 |  0.96119 | 0.977451 | 0.988495 |
| TT_ghostnet_strides_1_prelu_25_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_47_0.980000_IJBC_11         | 0.873038 | 0.921563 |  0.94943 | 0.967684 | 0.979189 | 0.989569 |
| TT_ghostnet_strides_1_prelu_25_se_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_fixed_float16_basic_agedb_30_epoch_50_0.976333_IJBC_11 | 0.872833 | 0.922892 | 0.949328 | 0.966457 | 0.980263 | 0.989262 |
| TT_botnet50_relu_GDC_arc_emb512_dr0_sgd_l2_5e4_bs1024_ms1m_bnm09_bne1e5_cos16_batch_restart_3_bias_false_basic_agedb_30_epoch_16_0.978167_IJBC_11                | 0.880043 | 0.934499 | 0.955924 | 0.970241 | 0.980161 | 0.988597 |
| TT_botnet50_relu_shortcut_act_none_GDC_arc_emb512_cos16_batch_restart_3_bias_false_conv_no_bias_tmul_2_basic_agedb_30_epoch_48_0.979667_IJBC_11                  |   0.8894 | 0.933834 |  0.95577 | 0.970292 | 0.981439 | 0.988546 |

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
***
