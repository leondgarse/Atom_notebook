# ___2021 - 07 - 06 Transformer___
***

# Attention
  - [遍地开花的 Attention ，你真的懂吗？](https://developer.aliyun.com/article/713354)
  - [综述---图像处理中的注意力机制](https://blog.csdn.net/xys430381_1/article/details/89323444)
  - [全连接的图卷积网络(GCN)和self-attention这些机制有什么区别联系](https://www.zhihu.com/question/366088445/answer/1023290162)
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - [《Attention is All You Need》浅读（简介+代码）](https://spaces.ac.cn/archives/4765)
  - [3W字长文带你轻松入门视觉transformer](https://zhuanlan.zhihu.com/p/308301901)
  - `keras.layers.Attention` a.k.a. Luong-style attention.
  - `keras.layers.AdditiveAttention` a.k.a. Bahdanau-style attention. [Eager 执行环境与 Keras 定义 RNN 模型使用注意力机制为图片命名标题](https://github.com/leondgarse/Atom_notebook/blob/master/public/2018/09-06_tensorflow_tutotials.md#eager-%E6%89%A7%E8%A1%8C%E7%8E%AF%E5%A2%83%E4%B8%8E-keras-%E5%AE%9A%E4%B9%89-rnn-%E6%A8%A1%E5%9E%8B%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%BA%E5%9B%BE%E7%89%87%E5%91%BD%E5%90%8D%E6%A0%87%E9%A2%98)
  - `keras.layers.MultiHeadAttention` multi-headed scaled dot-product attention based on "Attention is all you Need"
  - [Github Keras Attention Augmented Convolutions](https://github.com/titu1994/keras-attention-augmented-convs)
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

# Word2Vec
  ```py
  dd = {ii : np.stack([tf.random.log_uniform_candidate_sampler(true_classes=[[ii]], num_true=1, num_sampled=4, unique=True, range_max=8, seed=42)[0].numpy() for jj in range(1000)]) for ii in range(8)}
  cc = {ii: pd.value_counts(dd[ii].flatten()).to_dict() for ii in dd}
  print(cc[0])
  # {0: 864, 1: 715, 2: 569, 3: 481, 4: 389, 5: 371, 6: 314, 7: 297}
  print({ii : np.mean([tf.random.log_uniform_candidate_sampler(true_classes=[[ii]], num_true=1, num_sampled=4, unique=True, range_max=8, seed=42)[1].numpy()[0][0] for jj in range(1000)]) for ii in range(8)})
  # {0: 0.99967235, 1: 0.7245632, 2: 0.5737029, 3: 0.47004792, 4: 0.3987442, 5: 0.34728608, 6: 0.3084587, 7: 0.27554017}
  print({ii : np.mean([tf.random.log_uniform_candidate_sampler(true_classes=[[ii]], num_true=1, num_sampled=4, unique=True, range_max=8, seed=42)[2].numpy() for jj in range(1000)]) for ii in range(8)})
  # {0: 0.59829926, 1: 0.59250146, 2: 0.59380203, 3: 0.59515625, 4: 0.5941352, 5: 0.60234785, 6: 0.5936593, 7: 0.5999326}

  pairs, labels = tf.keras.preprocessing.sequence.skipgrams([1, 2, 3, 4, 5, 1, 6, 7], vocabulary_size=8, window_size=2, negative_samples=4)
  # list(zip(pairs, labels))
  pairs, labels = np.array(pairs), np.array(labels)
  negs, poses = pairs[labels == 0], pairs[labels == 1]
  poses = [tuple(ii) for ii in poses]
  neg_in_pos = np.sum([tuple(ii) in poses for ii in negs])
  print(neg_in_pos, neg_in_pos / negs.shape[0])
  # 62 0.5961538461538461

  rr_contexts = np.array(contexts)[:, :, 0]
  rr = [rr_contexts[ii][0] in rr_contexts[ii][1:] for ii in range(rr_contexts.shape[0])]
  print("Total negatives containing positive:", np.sum(rr), "ratio:", np.sum(rr) / rr_contexts.shape[0])
  # Total negatives containing positive: 2226 ratio: 0.03430843685459758
  print("Sample:", rr_contexts[np.array(rr)][:5].tolist())
  # Sample: [[1, 3, 0, 73, 1], [1, 1, 2, 47, 625], [4, 9, 717, 11, 4], [8, 15, 37, 26, 8], [1, 97, 1, 4, 120]]

  ff = np.logical_not(rr)
  dataset = tf.data.Dataset.from_tensor_slices(((targets[ff], contexts[ff]), labels[ff]))

  targets, contexts, labels = np.array(targets), np.array(contexts), np.array(labels)
  dd = pd.DataFrame({"targets": targets.tolist(), "pos": contexts[:, 0, 0].tolist(), "neg": contexts[:, 1:, 0].tolist()})
  cc = dd.groupby('targets').apply(lambda ii: np.sum([jj in ii['pos'].values for jj in np.concatenate(ii['neg'].values)]))
  print("Total negatives pairs containing positive pairs:", cc.sum(), "ratio:", cc.sum() / dd.shape[0])
  # Total negatives pairs containing positive pairs: 38660 ratio: 0.5953095887035925

  checkpoint = tf.train.Checkpoint(embedding=tf.Variable(word2vec.get_layer('w2v_embedding').get_weights()[0]))
  checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
  ```
  ```py
  unigrams = [0.99967235, 0.7245632, 0.5737029, 0.47004792, 0.3987442, 0.34728608, 0.3084587, 0.27554017]
  sample_func = lambda ii: tf.nn.fixed_unigram_candidate_sampler(true_classes=[[ii]], num_true=1, num_sampled=4, unique=True, range_max=8, unigrams=unigrams)
  dd = {ii : np.stack([sample_func(ii)[0].numpy() for jj in range(1000)]) for ii in range(8)}
  ```
# Imagenet
  ```py
  sys.path.append('/home/tdtest/workspace/Keras_insightface/')
  import data, models, myCallbacks
  input_shape=(64, 64, 3)
  image_classes_rule = data.ImageClassesRule_map('/datasets/ILSVRC2012_img_train')
  ds, steps_per_epoch = data.prepare_dataset('/datasets/ILSVRC2012_img_train', image_names_reg = "*/*", image_classes_rule=image_classes_rule, img_shape=input_shape, batch_size=512)

  # aa, bb = ds.as_numpy_iterator().next()
  # aa = (aa + 1) / 2
  # plt.imshow(np.vstack([np.hstack([aa[ii * 8 + jj] for jj in range(8)]) for ii in range(8)]))
  # plt.axis('off')
  # plt.tight_layout()
  # print([image_classes_rule.indices_2_classes[ii] for ii in bb.argmax(-1)][:10])
  # ['n02493793', 'n04204347', 'n01981276', 'n02027492', 'n03937543', 'n03743016', 'n04344873', 'n03590841', 'n04423845', 'n03720891']

  model = keras.applications.ResNet50V2(input_shape=input_shape, weights=None)
  model = models.add_l2_regularizer_2_model(model, weight_decay=5e-4, apply_to_batch_normal=False)
  optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
  # callbacks = myCallbacks.basic_callbacks(checkpoint="keras_checkpoints.h5", lr=0.1, lr_decay=0.1, lr_min=1e-5, lr_decay_steps=[20, 30, 40])
  lr_schduler = myCallbacks.CosineLrScheduler(0.1, first_restart_step=16, m_mul=0.5, t_mul=2.0, lr_min=1e-05, warmup=0, steps_per_epoch=steps_per_epoch)
  callbacks = [lr_schduler]

  model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(), metrics=['acc'])
  model.fit(ds, epochs=50, verbose=1, callbacks=callbacks, initial_epoch=0, steps_per_epoch=steps_per_epoch, use_multiprocessing=True, workers=4)
  ```
# BiT
  - [Colabs big_transfer_tf2.ipynb](https://colab.research.google.com/github/google-research/big_transfer/blob/master/colabs/big_transfer_tf2.ipynb)
  - [tfhub bit/m-r50x1](https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1)
  ```py
  def add_name_prefix(name, prefix=None):
    return prefix + "/" + name if prefix else name

  class StandardizedConv2D(tf.keras.layers.Conv2D):
    """Implements the abs/1903.10520 technique (see go/dune-gn).
    You can simply replace any Conv2D with this one to use re-parametrized
    convolution operation in which the kernels are standardized before conv.
    Note that it does not come with extra learnable scale/bias parameters,
    as those used in "Weight normalization" (abs/1602.07868). This does not
    matter if combined with BN/GN/..., but it would matter if the convolution
    was used standalone.
    Author: Lucas Beyer
    """

    def build(self, input_shape):
      super(StandardizedConv2D, self).build(input_shape)
      # Wrap a standardization around the conv OP.
      default_conv_op = self._convolution_op

      def standardized_conv_op(inputs, kernel):
        # Kernel has shape HWIO, normalize over HWI
        mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
        # Author code uses std + 1e-5
        return default_conv_op(inputs, (kernel - mean) / tf.sqrt(var + 1e-10))

      self._convolution_op = standardized_conv_op
      self.built = True

  class PaddingFromKernelSize(tf.keras.layers.Layer):
    """Layer that adds padding to an image taking into a given kernel size."""

    def __init__(self, kernel_size, **kwargs):
      super(PaddingFromKernelSize, self).__init__(**kwargs)
      self.kernel_size = kernel_size
      pad_total = kernel_size - 1
      self._pad_beg = pad_total // 2
      self._pad_end = pad_total - self._pad_beg

    def compute_output_shape(self, input_shape):
      batch_size, height, width, channels = tf.TensorShape(input_shape).as_list()
      if height is not None:
        height = height + self._pad_beg + self._pad_end
      if width is not None:
        width = width + self._pad_beg + self._pad_end
      return tf.TensorShape((batch_size, height, width, channels))

    def call(self, x):
      padding = [
          [0, 0],
          [self._pad_beg, self._pad_end],
          [self._pad_beg, self._pad_end],
          [0, 0]]
      return tf.pad(x, padding)

    def get_config(self):
      config = super(PaddingFromKernelSize, self).get_config()
      config.update({"kernel_size": self.kernel_size})
      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)

  class BottleneckV2Unit(tf.keras.layers.Layer):
    """Implements a standard ResNet's unit (version 2).
    """

    def __init__(self, num_filters, stride=1, **kwargs):
      """Initializer.
      Args:
        num_filters: number of filters in the bottleneck.
        stride: specifies block's stride.
        **kwargs: other tf.keras.layers.Layer keyword arguments.
      """
      super(BottleneckV2Unit, self).__init__(**kwargs)
      self._num_filters = num_filters
      self._stride = stride

      self._proj = None
      self._unit_a = tf.keras.Sequential([
          normalization.GroupNormalization(),
          ReLU(),
      ])
      self._unit_a_conv = StandardizedConv2D(
          filters=num_filters,
          kernel_size=1,
          use_bias=False,
          padding="VALID",
          trainable=self.trainable,)

      self._unit_b = tf.keras.Sequential([
          normalization.GroupNormalization(),
          ReLU(),
          PaddingFromKernelSize(kernel_size=3),
          StandardizedConv2D(
              filters=num_filters,
              kernel_size=3,
              strides=stride,
              use_bias=False,
              padding="VALID",
              trainable=self.trainable,)
      ])

      self._unit_c = tf.keras.Sequential([
          normalization.GroupNormalization(),
          ReLU(),
          StandardizedConv2D(
              filters=4 * num_filters,
              kernel_size=1,
              use_bias=False,
              padding="VALID",
              trainable=self.trainable)
      ])

    def build(self, input_shape):
      input_shape = tf.TensorShape(input_shape).as_list()

      # Add projection layer if necessary.
      if (self._stride > 1) or (4 * self._num_filters != input_shape[-1]):
        self._proj = StandardizedConv2D(
            filters=4 * self._num_filters,
            kernel_size=1,
            strides=self._stride,
            use_bias=False,
            padding="VALID",
            trainable=self.trainable)
      self.built = True

    def compute_output_shape(self, input_shape):
      current_shape = self._unit_a.compute_output_shape(input_shape)
      current_shape = self._unit_a_conv.compute_output_shape(current_shape)
      current_shape = self._unit_b.compute_output_shape(current_shape)
      current_shape = self._unit_c.compute_output_shape(current_shape)
      return current_shape

    def call(self, x):
      x_shortcut = x
      # Unit "a".
      x = self._unit_a(x)
      if self._proj is not None:
        x_shortcut = self._proj(x)
      x = self._unit_a_conv(x)
      # Unit "b".
      x = self._unit_b(x)
      # Unit "c".
      x = self._unit_c(x)

      return x + x_shortcut

    def get_config(self):
      config = super(BottleneckV2Unit, self).get_config()
      config.update({"num_filters": self._num_filters, "stride": self._stride})
      return config

    @classmethod
    def from_config(cls, config):
      return cls(**config)
  ```
  ```py
  from tensorflow.keras.layers import ReLU
  import tensorflow_addons.layers.normalizations as normalization

  num_units = (3, 4, 6, 3)
  num_outputs = 1000
  filters_factor = 4
  strides = (1, 2, 2, 2)
  trainable = False

  num_blocks = len(num_units)
  num_filters = tuple(16 * filters_factor * 2**b for b in range(num_blocks))

  def create_root_block(num_filters, conv_size=7, conv_stride=2, pool_size=3, pool_stride=2):
      layers = [
          PaddingFromKernelSize(conv_size),
          StandardizedConv2D(filters=num_filters, kernel_size=conv_size, strides=conv_stride, trainable=trainable, use_bias=False, name="standardized_conv2d"),
          PaddingFromKernelSize(pool_size),
          tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_stride, padding="valid")
      ]
      return tf.keras.Sequential(layers, name="root_block")

  def create_block(num_units, num_filters, stride, name):
      layers = []
      for i in range(1, num_units + 1):
          layers.append(BottleneckV2Unit(num_filters=num_filters, stride=(stride if i == 1 else 1), name=name + "_unit%02d" % i))
      return tf.keras.Sequential(layers, name=name)

  root = create_root_block(num_filters=num_filters[0])
  blocks = []
  for b, (f, u, s) in enumerate(zip(num_filters, num_units, strides), 1):
      n = "block{}".format(b)
      blocks.append(create_block(num_units=u, num_filters=f, stride=s, name=n))

  pre_head = [
      normalization.GroupNormalization(name="group_norm"),
      ReLU(),
      tf.keras.layers.GlobalAveragePooling2D()
  ]

  xx = keras.Input([None, None, 3])
  nn = root(xx)
  for block in blocks:
      nn = block(nn)
  for layer in pre_head:
      nn = layer(nn)
  dd = keras.models.Model(xx, nn)

  mm = keras.models.load_model('../module_cache/4ff8fefe176c863be939e3880dfa769989df4e32')
  mm.save_weights('aa.h5')

  dd.load_weights('aa.h5')
  dd.save('dd.h5', include_optimizer=False)
  ```
  ```py
  from tensorflow.keras.layers import ReLU
  import tensorflow_addons.layers.normalizations as normalization

  num_units = (3, 4, 6, 3)
  num_outputs = 1000
  filters_factor = 4
  strides = (1, 2, 2, 2)
  trainable = False

  num_blocks = len(num_units)
  num_filters = tuple(16 * filters_factor * 2**b for b in range(num_blocks))

  xx = keras.Input([None, None, 3])
  nn = PaddingFromKernelSize(7)(xx)
  nn = StandardizedConv2D(filters=num_filters[0], kernel_size=7, strides=2, trainable=trainable, use_bias=False, name="standardized_conv2d")(nn)
  nn = PaddingFromKernelSize(3)(nn)
  nn = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid")(nn)

  for bb, (ff, uu, ss) in enumerate(zip(num_filters, num_units, strides), 1):
      name = "block{}".format(bb)
      # create_block(num_units=u, num_filters=f, stride=s, name=n)
      for ii in range(1, uu + 1):
          nn = BottleneckV2Unit(num_filters=ff, stride=(ss if ii == 1 else 1), name=name + "_unit%02d" % ii)(nn)

  nn = normalization.GroupNormalization(name="group_norm")(nn)
  nn = ReLU()(nn)
  nn = tf.keras.layers.GlobalAveragePooling2D()(nn)

  dd = keras.models.Model(xx, nn)

  mm = keras.models.load_model('../module_cache/4ff8fefe176c863be939e3880dfa769989df4e32')
  mm.save_weights('aa.h5')

  dd.load_weights('aa.h5')
  dd.save('dd.h5', include_optimizer=False)
  ```
# Tape
  ```py
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  x_train, x_test = tf.expand_dims(x_train, -1) / 255, tf.expand_dims(x_test, -1) / 255
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

  class MyModel(keras.models.Model):
      def __init__(self, **kwargs):
          super(MyModel, self).__init__(**kwargs)
          self.conv = keras.layers.Conv2D(32, 3)
          self.flatten = keras.layers.Flatten()
          self.dense = keras.layers.Dense(10)
      def call(self, xx):
          xx = self.conv(xx)
          xx = self.flatten(xx)
          xx = self.dense(xx)
          return xx

      @tf.function
      def train_step(self, data):
          images, labels = data
          with tf.GradientTape() as tape:
              predictions = self(images, training=True)
              loss = self.compiled_loss(labels, predictions)
          gradients = tape.gradient(loss, self.trainable_variables)
          self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
          return {"loss": loss, "predictions": predictions}

      @tf.function
      def test_step(self, data):
          images, labels = data
          predictions = self(images, training=False)
          loss = self.compiled_loss(labels, predictions)
          return {"loss": loss, "predictions": predictions}

  model = MyModel()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam()
  model.compile(loss=loss_object, optimizer=optimizer)

  # Metrics
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  # fit: model.fit(train_ds, validation_data=test_ds, epochs=10)
  EPOCHS = 10
  for epoch in range(EPOCHS):
      start = time.time()
      train_loss.reset_states()
      train_accuracy.reset_states()
      train_loss.reset_states()
      test_accuracy.reset_states()
      for batch_n, (images, labels) in enumerate(train_ds):
          logs = model.train_step([images, labels])
          train_loss.update_state(logs['loss'])
          train_accuracy.update_state(labels, logs['predictions'])
          if batch_n % 100 == 0:
              print(f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}")

      for test_images, test_labels in test_ds:
          logs = model.test_step([test_images, test_labels])
          test_loss.update_state(logs['loss'])
          test_accuracy.update_state(test_labels, logs['predictions'])

      # if (epoch + 1) % 5 == 0:
      #     model.save_weights(checkpoint_prefix.format(epoch=epoch))

      print(
          f'Epoch {epoch + 1}, '
          f'Loss: {train_loss.result()}, '
          f'Accuracy: {train_accuracy.result() * 100}, '
          f'Test Loss: {test_loss.result()}, '
          f'Test Accuracy: {test_accuracy.result() * 100}'
      )
      print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
      print("_" * 80)
  ```
***

# Transformer
## 位置编码 Positional encoding
  ```sh
  PE(pos, 2i) = sin(pos / 10000 ** (2i / d_model))
  PE(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model))
  ```
  ```py
  def positional_encoding(position, d_model):
      """
      position, d_model = 50, 512
      d_model_range = np.expand_dims(np.arange(d_model), 0) --> [0, 1, 2, 3,..., 509, 510, 511]
      (2 * (d_model_range // 2)) / np.float32(d_model) --> [0, 0, 2, 2,..., 508, 510, 510] / 512
      np.power(1e4, (2 * (d_model_range // 2)) / np.float32(d_model)) --> ~ [1, 1, ..., 1e4, 1e4]
      angle_rads --> ~ [1, 1, ..., 1e-4, 1e-4]
      angle_rads --> [angle_rads * 0, angle_rads * 1, angle_rads * 2, ..., angle_rads * 49]
      [sin] angle_rads[0] --> [0]
      [sin] angle_rads[1] --> ~[sin(1), sin(1), ..., 0, 0]
      [cos] angle_rads[0] --> [1]
      [cos] angle_rads[1] --> ~[cos(1), cos(1), ..., 1, 1]
      """
      d_model_range = np.expand_dims(np.arange(d_model), 0)
      angle_rads = 1 / np.power(1e4, (2 * (d_model_range // 2)) / np.float32(d_model))
      angle_rads = np.expand_dims(np.arange(position), 1) * angle_rads

      # 将 sin 应用于数组中的偶数索引（indices）；2i
      angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
      # 将 cos 应用于数组中的奇数索引；2i+1
      angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

      pos_encoding = np.expand_dims(angle_rads, 0)
      return tf.cast(pos_encoding, dtype=tf.float32)

  pos_encoding = positional_encoding(50, 512)
  print(f'{pos_encoding.shape = }') # pos_encoding.shape = TensorShape([1, 50, 512])
  plt.pcolormesh(pos_encoding[0], cmap='RdBu')
  plt.xlabel('Depth')
  plt.xlim((0, 512))
  plt.ylabel('Position')
  plt.colorbar()
  ```
  ![](images/position_encoding_colormesh.svg)
  ```py
  for ii in range(0, 50, 10):
      plt.plot(pos_encoding[0, ii, ::2], label="sin, {}".format(ii))
      plt.plot(pos_encoding[0, ii, 1::2], label="cos, {}".format(ii))
  plt.legend()
  plt.title("Position values")
  ```
  ![](images/position_encoding_position_values.svg)
  ```py
  print((pos_encoding.numpy()[0] ** 2).sum(1))
  # [256] * 50

  print(np.dot(pos_encoding.numpy()[0], pos_encoding.numpy()[0, 0]))
  # [256.      249.10211 231.73363 211.74947 196.68826 189.59668 188.2482 187.86502 184.96516 179.45654 173.78973 170.35315 169.44649 169.34525
  #  168.0597  165.0636  161.53304 159.08392 158.2816  158.24513 157.57397 155.64383 153.08748 151.10722 150.33557 150.30371 149.94781 148.61731
  #  146.63585 144.93758 144.17035 144.1174  143.94557 143.00458 141.41406 139.9111  139.13742 139.0499  138.99149 138.32632 137.02701 135.67337
  #  134.88887 134.7587  134.77036 134.31155 133.24333 132.01273 131.21637 131.03839]

  for ii in range(0, 50, 10):
      plt.plot(np.dot(pos_encoding.numpy()[0], pos_encoding.numpy()[0, ii]), label=str(ii))
  plt.legend()
  plt.title("Dists between values")
  plt.xlabel('position id')
  plt.ylabel('dist')
  ```
  ![](images/position_encoding_dist_values.svg)
## Scaled dot product attention
  - 点积注意力被缩小了深度的平方根倍。这样做是因为对于较大的深度值，点积的大小会增大，从而推动 softmax 函数往仅有很小的梯度的方向靠拢，导致了一种很硬的（hard）softmax。
  ```py
  def scaled_dot_product_attention(q, k, v, mask):
      """计算注意力权重。
      q, k, v 必须具有匹配的前置维度。
      k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
      虽然 mask 根据其类型（填充或前瞻）有不同的形状，
      但是 mask 必须能进行广播转换以便求和。

      参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
              (..., seq_len_q, seq_len_k)。默认为None。

      返回值:
        输出，注意力权重
      """
      matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

      # 缩放 matmul_qk
      dk = tf.cast(tf.shape(k)[-1], tf.float32)
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

      # 将 mask 加入到缩放的张量上。
      if mask is not None:
          scaled_attention_logits += (mask * -1e9)  

      # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
      output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
      return output, attention_weights
  ```
  ```py
  temp_k = tf.expand_dims([1., 2, 3, 4], 1)
  temp_v = tf.expand_dims([1., 2, 3, 4], 1)
  print("Output: {}, Attention: {}".format(*scaled_dot_product_attention(tf.constant([[5.]]), temp_k, temp_v, None)))
  # Output: [[3.9932165]], Attention: [[0.0000003  0.00004509 0.00669255 0.9932621 ]]
  print("Output: {}, Attention: {}".format(*scaled_dot_product_attention(tf.constant([[50.]]), temp_k, temp_v, None)))
  # Output: [[4.]], Attention: [[0. 0. 0. 1.]]

  temp_k = tf.constant([[10, 0, 0], [0, 10, 0], [0, 0, 10], [0, 0, 10]], dtype=tf.float32)  # (4, 3)
  temp_v = tf.constant([[1, 0], [10, 0], [100, 5], [1000, 6]], dtype=tf.float32)  # (4, 2)

  # 这条 `请求（query）符合第二个`主键（key）`，因此返回了第二个`数值（value）`。
  temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
  print("Output: {}, Attention: {}".format(*scaled_dot_product_attention(temp_q, temp_k, temp_v, None)))
  # Output: [[10.  0.]], Attention: [[0. 1. 0. 0.]]

  # 这条请求符合重复出现的主键（第三第四个），因此，对所有的相关数值取了平均。
  temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
  print("Output: {}, Attention: {}".format(*scaled_dot_product_attention(temp_q, temp_k, temp_v, None)))
  # Output: [[550.    5.5]], Attention: [[0.  0.  0.5 0.5]]

  # 这条请求符合第一和第二条主键，因此，对它们的数值去了平均。
  temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
  print("Output: {}, Attention: {}".format(*scaled_dot_product_attention(temp_q, temp_k, temp_v, None)))
  # Output: [[5.5 0. ]], Attention: [[0.5 0.5 0.  0. ]]

  temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
  print("Output: {}, Attention: {}".format(*scaled_dot_product_attention(temp_q, temp_k, temp_v, None)))
  # Output: [[550.    5.5] [ 10.    0. ] [  5.5   0. ]], Attention: [[0.  0.  0.5 0.5] [0.  1.  0.  0. ] [0.5 0.5 0.  0. ]]
  ```
## 多头注意力 Multi-head attention
  多头注意力由四部分组成：

  线性层并分拆成多头。
  按比缩放的点积注意力。
  多头及联。
  最后一层线性层。
  每个多头注意力块有三个输入：Q（请求）、K（主键）、V（数值）。这些输入经过线性（Dense）层，并分拆成多头。

  将上面定义的 scaled_dot_product_attention 函数应用于每个头（进行了广播（broadcasted）以提高效率）。注意力这步必须使用一个恰当的 mask。然后将每个头的注意力输出连接起来（用tf.transpose 和 tf.reshape），并放入最后的 Dense 层。

  Q、K、和 V 被拆分到了多个头，而非单个的注意力头，因为多头允许模型共同注意来自不同表示空间的不同位置的信息。在分拆后，每个头部的维度减少，因此总的计算成本与有着全部维度的单个注意力头相同。
  ```py
  query: Query `Tensor` of shape `[B, T, dim]`
  value: Value `Tensor` of shape `[B, S, dim]`
  key: Optional key `Tensor` of shape `[B, S, dim]`
  attention_mask: a boolean mask of shape `[B, T, S]`
  attention_output: The result of the computation, of shape [B, T, E] where `T` is for target sequence shapes and `E` is the query input last dimension

  N = `num_attention_heads`
  H = `size_per_head`
  `query` = [B, T, N ,H]
  `key` = [B, S, N, H]
  `value` = [B, S, N, H]
  ```
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

  attention_scores = nn._masked_softmax(attention_scores, None)
  attention_scores_dropout = nn._dropout_layer(attention_scores, training=False)
  attention_output = special_math_ops.einsum(nn._combine_equation, attention_scores_dropout, value)
  ic(attention_output.shape.as_list())
  # ic| attention_output.shape.as_list(): [None, 14, 16, 4, 128]

  attention_output = nn._output_dense(attention_output)
  ic(attention_output.shape.as_list())
  # ic| attention_output.shape.as_list(): [None, 14, 16, 1024]
  ```
***

# VOLO
## PyTorch fold and unfold and conv2d
  - [torch.nn.Unfold](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)
  - Torch `unfold --> fold` will cumulative sum values on overlapped areas.
    ```py
    import torch
    import torch.nn.functional as F
    aa = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)  # NCHW
    bb = F.unfold(aa, 3, padding=1, stride=2)
    print(f"{bb.shape = }")
    # bb.shape = torch.Size([1, 9, 4])
    print(bb[0].reshape(3, 3, 2, 2).permute(2, 0, 3, 1).reshape(6, 6).numpy())
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  1.  1.  2.  3.]
    #  [ 0.  4.  5.  5.  6.  7.]
    #  [ 0.  4.  5.  5.  6.  7.]
    #  [ 0.  8.  9.  9. 10. 11.]
    #  [ 0. 12. 13. 13. 14. 15.]]
    print(F.fold(bb, (4, 4), 3, padding=1, stride=2)[0, 0].numpy())
    # [[ 0.  2.  2.  3.]
    #  [ 8. 20. 12. 14.]
    #  [ 8. 18. 10. 11.]
    #  [12. 26. 14. 15.]]
    print(F.fold(bb, (3, 3), 3, padding=1, stride=2)[0, 0].numpy())
    # [[ 0.  2.  2.]
    #  [ 8. 20. 12.]
    #  [ 8. 18. 10.]]

    from torch import nn
    inputs = torch.arange(128, dtype=torch.float32).reshape(1, 8, 4, 4)  # NCHW
    fold_params = dict(kernel_size=3, dilation=1, padding=1, stride=2)
    fold = nn.Fold(output_size=inputs.shape[2:4], **fold_params)
    unfold = nn.Unfold(**fold_params)

    divisor = fold(unfold(torch.ones_like(inputs)))  # Overlapped area will be > 1
    print(f"{np.allclose(fold(unfold(inputs)), divisor * inputs) = }")
    # np.allclose(fold(unfold(inputs)), divisor * inputs) = True
    ```
    ![](images/fold_unfold.png)
  - Result by `tf.image.extract_patches`
    ```py
    aa = np.arange(16, dtype="float32").reshape(1, 4, 4, 1)
    bb = tf.pad(aa, [[0, 0], [1, 1], [1, 1], [0, 0]])
    cc = tf.image.extract_patches(bb, [1, 3, 3, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding="VALID").numpy()
    print(f"{cc.shape = }")
    # cc.shape = (1, 2, 2, 9)
    print(cc[0].reshape(2, 2, 3, 3).transpose(0, 2, 1, 3).reshape(6, 6))
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  1.  1.  2.  3.]
    #  [ 0.  4.  5.  5.  6.  7.]
    #  [ 0.  4.  5.  5.  6.  7.]
    #  [ 0.  8.  9.  9. 10. 11.]
    #  [ 0. 12. 13. 13. 14. 15.]]
    ```
  - Torch `unfold --> matmul --> fold` is equal to `conv2d` if no overlap. [ ??? ]
    ```py
    inp = torch.randn(1, 3, 10, 12)
    w = torch.randn(2, 3, 4, 5) # [output_channel, input_channel, kernel_size_1, kernel_size_2]

    inp_unf = torch.nn.functional.unfold(inp, (4, 5)) # [1, 60, 56]
    inp_unf = inp_unf.transpose(1, 2) # [1, 56, 60]
    ww = w.view(w.size(0), -1).t()  # [60, 2]
    out_unf = inp_unf.matmul(ww)  # [1, 56, 2]
    out_unf = out_unf.transpose(1, 2) # ([1, 2, 56]
    out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1)) # [1, 2, 7, 8]

    print(f"{np.allclose(torch.nn.functional.conv2d(inp, w), out, atol=1e-6) = }")
    # np.allclose(torch.nn.functional.conv2d(inp, w), out, atol=1e-6) = True
    ```
    ```py
    inp = torch.randn(1, 3, 10, 12)
    w = torch.randn(2, 3, 4, 5) # [output_channel, input_channel, kernel_size_1, kernel_size_2]

    inp_unf = torch.nn.functional.unfold(inp, (4, 5), stride=2, padding=1) # [1, 60, 25]
    inp_unf = inp_unf.transpose(1, 2) # [1, 25, 60]
    ww = w.view(w.size(0), -1).t()  # [60, 2]
    out_unf = inp_unf.matmul(ww)  # [1, 25, 2]
    out_unf = out_unf.transpose(1, 2) # ([1, 2, 25]
    out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1), stride=2, padding=1) # [1, 2, 7, 8]

    conv_out = torch.nn.functional.conv2d(inp, w, stride=2, padding=1)
    print(f"{out.shape = }, {conv_out.shape = }")
    # out.shape = torch.Size([1, 2, 7, 8]), conv_out.shape = torch.Size([1, 2, 5, 5])
    ```
## TF extract patches and Torch unfold
  - **extract_patches output shape**
    ```py
    tf.pad(tf.ones([1, 28, 28, 72]), [[0, 0], [1, 1], [1, 1], [0, 0]]).shape
    # TensorShape([1, 30, 30, 72])
    ```
    ```py
    img = np.random.uniform(size=[1, 28, 28, 192])
    kernel_size, strides = 4, 4
    patches = tf.image.extract_patches(img, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 7, 7, 3072]) --> 4 * 7 == 28

    img2 = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]])
    print(f"{img2.shape = }")
    # img2.shape = TensorShape([1, 30, 30, 192])
    kernel_size, strides = 3, 3

    patches = tf.image.extract_patches(img2, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 10, 10, 1728])  --> 3 * 10 == 30

    kernel_size = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    print(f"{tf.image.extract_patches(tf.ones([1, 27, 27, 3]), kernel_size, strides, [1, 1, 1, 1], padding='VALID').shape = }")
    # [1, 13, 13, 27] --> np.ceil(27 / 2) - 1 == 13
    print(f"{tf.image.extract_patches(tf.ones([1, 28, 28, 3]), kernel_size, strides, [1, 1, 1, 1], padding='VALID').shape = }")
    # [1, 13, 13, 27] --> np.ceil(28 / 2) - 1 == 13
    print(f"{tf.image.extract_patches(tf.ones([1, 29, 29, 3]), kernel_size, strides, [1, 1, 1, 1], padding='VALID').shape = }")
    # [1, 14, 14, 27] --> np.ceil(29 / 2) - 1 == 14
    print(f"{tf.image.extract_patches(tf.ones([1, 30, 30, 3]), kernel_size, strides, [1, 1, 1, 1], padding='VALID').shape = }")
    # [1, 14, 14, 27] --> np.ceil(30 / 2) - 1 == 14
    ```
  - **Comparing Torch unfold** `extract_patches` output channels like `RGBRGB...`, while `toch.nn.Unfold` like `RRR...GGG...BBB...`.
    ```py
    import torch
    from torch import nn

    image = np.zeros([256, 256, 3])
    image[:, :, 1] += 128 # G
    image[:, :, 2] += 256 # B

    aa = np.expand_dims(image.astype("float32"), 0)
    cc = nn.Unfold(kernel_size=3, padding=1, stride=2)(torch.from_numpy(aa).permute(0, 3, 1, 2)).permute(0, 2, 1)

    # bb = keras.layers.ZeroPadding2D(1)(aa)
    bb = tf.pad(aa, [[0, 0], [1, 1], [1, 1], [0, 0]])
    bb = tf.image.extract_patches(bb, sizes=[1, 3, 3, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='VALID')

    torch_stack = cc.numpy().reshape(1, 128, 128, 27).transpose(0, 3, 1, 2) / 255 # RRR...GGG...BBB...
    tf_stack = bb.numpy().transpose(0, 3, 1, 2) / 255  # RGBRGB...

    print(f"{np.allclose(torch_stack[0, 0], tf_stack[0, 0], atol=1e-7) = }")
    # np.allclose(torch_stack[0], tf_stack[0], atol=1e-7) = True
    channel = image.shape[-1]
    tf_transpose_channel = tf_stack.reshape(-1, 9, channel, 128, 128).transpose(0, 2, 1, 3, 4).reshape(-1, 27, 128, 128)
    print(f"{np.allclose(tf_transpose_channel, torch_stack, atol=1e-7) = }")
    # np.allclose(tf_picked_stack, torch_stack, atol=1e-7) = True

    fig, axes = plt.subplots(2, 1, figsize=(12, 2))
    axes[0].imshow(np.hstack(torch_stack[0]))
    axes[0].axis('off')
    axes[0].set_title("Torch image stack", fontsize='small')
    axes[1].imshow(np.hstack(tf_stack[0]))
    axes[1].axis('off')
    axes[1].set_title("TF image stack", fontsize='small')
    fig.tight_layout()
    ```
    ![](images/unfold_and_extract_patches.png)
## TF reverse extracted patches without overlap
  - Without overlap means `kernel_size == strides`, not considering `kernel_size < strides`.
  - `tf.nn.space_to_depth` works same with `extract_patches` if no overlap.
    ```py
    img = np.random.uniform(size=[2, 28, 28, 192])
    img_padded = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]])
    block_size = 3
    print(f"{tf.nn.space_to_depth(img_padded, block_size).shape = }")
    # [2, 10, 10, 1728]

    patches = tf.image.extract_patches(img_padded, [1, block_size, block_size, 1], [1, block_size, block_size, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{np.allclose(tf.nn.space_to_depth(img_padded, block_size), patches) = }")
    # np.allclose(tf.nn.space_to_depth(img_padded, block_size), patches) = True
    ```
  - `tf.nn.depth_to_space` is the reverse of `space_to_depth`
    ```py
    img = np.random.uniform(size=[2, 28, 28, 192])
    img_padded = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]])
    block_size = 3
    img_to_depth = tf.nn.space_to_depth(img_padded, block_size)
    print(f"{img_to_depth.shape = }")
    # img_to_depth.shape = TensorShape([2, 10, 10, 1728])
    depth_to_img = tf.nn.depth_to_space(img_to_depth, block_size)
    print(f"{depth_to_img.shape = }")
    # depth_to_img.shape = TensorShape([2, 30, 30, 192])
    print(f"{np.allclose(depth_to_img[:, 1:-1, 1:-1, :], img) = }")
    # np.allclose(depth_to_img[:, 1:-1, 1:-1, :], img) = True
    ```
  - This also can be done just using `transpose` and `reshape`
    ```py
    ww, hh = img_padded.shape[1] // block_size, img_padded.shape[2] // block_size
    bb = img_padded.numpy().reshape(-1, ww, block_size, hh, block_size, img_padded.shape[-1])
    cc = bb.transpose(0, 1, 3, 2, 4, 5).reshape(-1, ww, hh, block_size * block_size * img_padded.shape[-1])
    print(f"{cc.shape = }")
    # cc.shape = (2, 10, 10, 1728)
    print(f"{np.allclose(cc, patches) = }")
    # np.allclose(cc, patches) = True

    """ Reverse of `extract_patches` using `transpose` and `reshape` """
    dd = patches.numpy().reshape(-1, patches.shape[1], patches.shape[2], block_size, block_size, img_padded.shape[-1])
    print(f"{dd.shape = }")
    # dd.shape = (2, 10, 10, 3, 3, 192)
    ee = dd.transpose(0, 1, 3, 2, 4, 5).reshape(-1, *img_padded.shape[1:])
    print(f"{ee.shape = }")
    # ee.shape = (2, 30, 30, 192)
    print(f"{np.allclose(img, ee[:, 1:-1, 1:-1, :]) = }")
    # np.allclose(img, ee[:, 1:-1, 1:-1, :]) = True
    ```
  - `tf.space_to_batch` is same with `tf.space_to_batch_nd`
    ```py
    # /opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py
    def space_to_batch_v2(input, block_shape, paddings, name=None):
        return space_to_batch_nd(input, block_shape, paddings, name)
    ```
  - `tf.space_to_batch_nd` works similar to `tf.nn.space_to_depth`, just stacked `kernels` on `batch` dimension.
    ```py
    aa = np.arange(32, dtype="float32").reshape(2, 4, 4, 1)
    aa_padded = tf.pad(aa, [[0, 0], [1, 1], [1, 1], [0, 0]])
    aa_to_batch = tf.space_to_batch_nd(aa_padded, [3, 3], [[0, 0], [0, 0]])
    print(f"{aa_to_batch.shape = }")
    # aa_to_batch.shape = TensorShape([18, 2, 2, 1])

    aa_to_depth = tf.nn.space_to_depth(aa_padded, 3)
    print(f"{aa_to_depth.shape = }")
    # aa_to_depth.shape = TensorShape([2, 2, 2, 9])
    print(f"{np.allclose(tf.reshape(aa_to_depth, [-1, 4, 9]), tf.transpose(tf.reshape(aa_to_batch, [9, -1, 4]), [1, 2, 0])) = }")
    # np.allclose(tf.reshape(aa_to_depth, [-1, 4, 9]), tf.transpose(tf.reshape(aa_to_batch, [9, -1, 4]), [1, 2, 0])) = True
    ```
    ```py
    img = np.random.uniform(size=[2, 28, 28, 192])
    kernel_size = 3
    img_to_batch = tf.space_to_batch_nd(img, [kernel_size, kernel_size], [[1, 1], [1, 1]])
    print(f"{img_to_batch.shape = }")
    # img_to_batch.shape = TensorShape([18, 10, 10, 192])

    img_to_depth = tf.nn.space_to_depth(tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]]), kernel_size)
    print(f"{img_to_depth.shape = }")
    # img_to_depth.shape = TensorShape([2, 10, 10, 1728])

    bb = tf.transpose(tf.reshape(img_to_batch, [kernel_size * kernel_size, -1, *img_to_batch.shape[1:]]), [1, 2, 3, 0, 4])
    img_to_batch_reshaped = tf.reshape(bb, [-1, bb.shape[1], bb.shape[2], bb.shape[3] * bb.shape[4]])
    print(f"{img_to_batch_reshaped.shape = }")
    # img_to_batch_reshaped.shape = TensorShape([2, 10, 10, 1728])
    print(f"{np.allclose(img_to_depth, img_to_batch_reshaped) = }")
    # np.allclose(img_to_depth, img_to_batch_reshaped) = True
    ```
  - `tf.batch_to_space` is the reverse of `tf.space_to_batch_nd`
    ```py
    img = np.random.uniform(size=[2, 28, 28, 192])
    img_to_batch = tf.space_to_batch_nd(img, [3, 3], [[1, 1], [1, 1]])
    print(f"{img_to_batch.shape = }")
    # img_to_batch.shape = TensorShape([18, 10, 10, 192])

    batch_to_img = tf.batch_to_space(img_to_batch, [3, 3], [[1, 1], [1, 1]])
    print(f"{batch_to_img.shape = }")
    # batch_to_img.shape = TensorShape([2, 28, 28, 192])
    print(f"{np.allclose(img, batch_to_img) = }")
    # np.allclose(img, batch_to_img) = True
    ```
## TF reverse extracted patches with overlap
  - For case of overlap happens once, like `kernel_size=3, strides=2`, split `patches` into `4` groups, and sum them up to handle overlapped areas. Others scenarios like `kernel_size=3, strides=1`, overlap happens twice, will need split into `9` groups.
    ```py
    img = np.random.uniform(size=[1, 28, 28, 192])
    kernel_size, strides = 3, 2
    channel = img.shape[-1]
    img_pad = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]])
    patches = tf.image.extract_patches(img_pad, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 14, 14, 1728])

    patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], kernel_size, kernel_size, channel))
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 14, 14, 3, 3, 192])

    overlap_pad = 2 * strides - kernel_size
    aa = tf.pad(patches, [[0, 0], [0, 0], [0, 0], [0, overlap_pad], [0, overlap_pad], [0, 0]])
    print(f"{aa.shape = }")
    # aa.shape = TensorShape([1, 14, 14, 4, 4, 192])

    bb = tf.transpose(aa, [0, 1, 3, 2, 4, 5])
    print(f"{bb.shape = }")
    # bb.shape = TensorShape([1, 14, 4, 14, 4, 192])
    ww, hh = bb.shape[1] // 2, bb.shape[3] // 2
    output_shape = [-1, ww * bb.shape[2], hh * bb.shape[4], bb.shape[5]]
    tf_fold = tf.pad(tf.reshape(bb[:, ::2, :, ::2, :, :], output_shape), [[0, 0], [0, strides], [0, strides], [0, 0]])
    print(f"{tf_fold.shape = }")
    # tf_fold.shape = TensorShape([1, 30, 30, 192])
    tf_fold += tf.pad(tf.reshape(bb[:, ::2, :, 1::2, :, :], output_shape), [[0, 0], [0, strides], [strides, 0], [0, 0]])
    tf_fold += tf.pad(tf.reshape(bb[:, 1::2, :, ::2, :, :], output_shape), [[0, 0], [strides, 0], [0, strides], [0, 0]])
    tf_fold += tf.pad(tf.reshape(bb[:, 1::2, :, 1::2, :, :], output_shape), [[0, 0], [strides, 0], [strides, 0], [0, 0]])

    print(f"{tf_fold.shape = }")
    # tf_fold.shape = TensorShape([1, 30, 30, 192])

    """ Torch unfold --> fold """
    import torch
    torch_inputs = torch.from_numpy(img).permute(0, 3, 1, 2)  # NCHW
    unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=strides)
    fold = torch.nn.Fold(output_size=torch_inputs.shape[2:4], kernel_size=kernel_size, dilation=1, padding=1, stride=strides)
    torch_fold = fold(unfold(torch_inputs))
    print(f"{torch_fold.shape = }")
    # torch_fold.shape = torch.Size([1, 192, 28, 28])

    print(f"{np.allclose(torch_fold.permute(0, 2, 3, 1), tf_fold[:, 1:-1, 1:-1, :]) = }")
    # np.allclose(torch_fold.permute(0, 2, 3, 1), tf_fold[:, 1:-1, 1:-1, :]) = True
    ```
    ![](images/fold_overlap_1.png)
  - **Collect into functions**
    ```py
    def pad_overlap_1(patches, start_x, start_y, strides):
        bb = patches[:, start_x::2, :, start_y::2, :, :]  # [1, 7, 4, 7, 4, 192]
        bb = tf.reshape(bb, [-1, bb.shape[1] * bb.shape[2], bb.shape[3] * bb.shape[4], bb.shape[-1]])  # [1, 28, 28, 192]
        pad_x = [0, strides] if start_x == 0 else [strides, 0]
        pad_y = [0, strides] if start_y == 0 else [strides, 0]
        padding = [[0, 0], pad_x, pad_y, [0, 0]]
        bb = tf.pad(bb, padding)
        return bb

    def fold_overlap_1(patches, output_size, kernel_size, strides, head_padding=[0, 0]):
        channel = patches.shape[-1] // (kernel_size * kernel_size)
        patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], kernel_size, kernel_size, channel)) # [1, 14, 14, 3, 3, 192]
        stride_pad = 2 * strides - kernel_size
        aa = tf.pad(patches, [[0, 0], [0, 0], [0, 0], [0, stride_pad], [0, stride_pad], [0, 0]]) # [1, 14, 14, 4, 4, 192], 14 // 2 * 4 == 28
        aa = tf.transpose(aa, [0, 1, 3, 2, 4, 5])   # [1, 14, 4, 14, 4, 192]
        bb = pad_overlap_1(aa, 0, 0, strides) + pad_overlap_1(aa, 0, 1, strides) + pad_overlap_1(aa, 1, 0, strides) + pad_overlap_1(aa, 1, 1, strides)
        out_sx, out_sy = head_padding
        out_ex, out_ey = out_sx + output_size[0], out_sy + output_size[1]
        return bb[:, out_sx:out_ex, out_sy:out_ey, :]

    img = np.random.uniform(size=(1, 28, 28, 192)).astype('float32')
    kernel_size, strides = 3, 2

    img_pad = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]])
    patches = tf.image.extract_patches(img_pad, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    tf_fold = fold_overlap_1(patches, img.shape[1:], kernel_size, strides, head_padding=[1, 1])
    print(f"{tf_fold.shape = }")
    # tf_fold.shape = TensorShape([1, 28, 28, 192])

    """ Torch unfold --> fold """
    import torch
    torch_inputs = torch.from_numpy(img).permute(0, 3, 1, 2)  # NCHW
    unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=strides)
    fold = torch.nn.Fold(output_size=torch_inputs.shape[2:4], kernel_size=kernel_size, dilation=1, padding=1, stride=strides)
    torch_fold = fold(unfold(torch_inputs))
    print(f"{torch_fold.shape = }")
    # torch_fold.shape = torch.Size([1, 192, 28, 28])

    print(f"{np.allclose(torch_fold.permute(0, 2, 3, 1), tf_fold) = }")
    # np.allclose(torch_fold.permute(0, 2, 3, 1), tf_fold) = True
    ```
## TF reverse extracted patches with overlap NOT cumulative sum on overlapped areas
  - If use a bigger padding to `extract_patches`, we can drop the overlappped areas, and use `new_kernel_size = strides` to reverse.
    ```py
    img = np.random.uniform(size=[1, 28, 28, 192])
    kernel_size, strides = 5, 3
    channel = img.shape[-1]
    img_pad = tf.pad(img, [[0, 0], [1, 1 + strides], [1, 1 + strides], [0, 0]])
    patches = tf.image.extract_patches(img_pad, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 15, 15, 4800])

    patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], kernel_size, kernel_size, channel))
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 10, 10, 5, 5, 192])
    non_overlap_patches = tf.transpose(patches[:, :, :, :strides, :strides, :], [0, 1, 3, 2, 4, 5])
    print(f"{non_overlap_patches.shape = }")
    # non_overlap_patches.shape = TensorShape([1, 10, 3, 10, 3, 192])
    _, ww, kww, hh, khh, cc = non_overlap_patches.shape
    reconstructed = tf.reshape(non_overlap_patches, [-1, ww * kww, hh * khh, cc])
    print(f"{reconstructed.shape = }")
    # reconstructed.shape = TensorShape([1, 30, 30, 192])
    print(f"{np.allclose(reconstructed[:, 1: 1 + img.shape[1], 1: 1 + img.shape[2], :], img) = }")
    # np.allclose(reconstructed[:, 1: 1 + img.shape[1], 1: 1 + img.shape[2], :], img) = True
    ```
  - Else we need to create the last patch. If bigger gap between `kernel_size` and `strides`, like `kernel_size=10, strides=2`, need to create more patches.
    ```py
    img = np.random.uniform(size=[1, 28, 28, 192])
    kernel_size, strides = 7, 3
    channel = img.shape[-1]
    padded = kernel_size // 2
    img_pad = tf.pad(img, [[0, 0], [padded, padded], [padded, padded], [0, 0]])
    patches = tf.image.extract_patches(img_pad, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 9, 9, 4800])

    patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], kernel_size, kernel_size, channel))
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 9, 5, 9, 5, 192])

    """ Create the last patch """
    patches = tf.concat([patches, tf.pad(patches[:, -1:, strides:], [[0, 0], [0, 0], [0, strides]] + [[0, 0]] * 3)], axis=1)
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 10, 5, 9, 5, 192])
    patches = tf.concat([patches, tf.pad(patches[:, :, :, -1:, strides:], [[0, 0]] * 3 + [[0, 0], [0, strides], [0, 0]])], axis=3)
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 10, 5, 10, 5, 192])

    non_overlap_patches = patches[:, :, :strides, :, :strides, :]
    print(f"{non_overlap_patches.shape = }")
    # non_overlap_patches.shape = TensorShape([1, 10, 3, 10, 3, 192])
    _, ww, kww, hh, khh, cc = non_overlap_patches.shape
    reconstructed = tf.reshape(non_overlap_patches, [-1, ww * kww, hh * khh, cc])
    print(f"{reconstructed.shape = }")
    # reconstructed.shape = TensorShape([1, 30, 30, 192])

    print(f"{np.allclose(reconstructed[:, padded: padded + img.shape[1], padded: padded + img.shape[2], :], img) = }")
    # np.allclose(reconstructed[:, padded: padded + img.shape[1], padded: padded + img.shape[2], :], img) = True
    ```
  - Or concatenate the cut values in the last patch to the result.
    ```py
    img = np.random.uniform(size=[1, 28, 28, 192])
    kernel_size, strides = 10, 2
    channel = img.shape[-1]
    padded = kernel_size // 2
    img_pad = tf.pad(img, [[0, 0], [padded, padded], [padded, padded], [0, 0]])
    patches = tf.image.extract_patches(img_pad, [1, kernel_size, kernel_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding='VALID')
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 15, 15, 19200])

    patches = tf.reshape(patches, (-1, patches.shape[1], patches.shape[2], kernel_size, kernel_size, channel))
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    print(f"{patches.shape = }")
    # patches.shape = TensorShape([1, 15, 10, 15, 10, 192])
    non_overlap_patches = patches[:, :, :strides, :, :strides, :]
    print(f"{non_overlap_patches.shape = }")
    # non_overlap_patches.shape = TensorShape([1, 15, 2, 15, 2, 192])
    _, ww, kww, hh, khh, cc = non_overlap_patches.shape
    reconstructed = tf.reshape(non_overlap_patches, [-1, ww * kww, hh * khh, cc])
    print(f"{reconstructed.shape = }")
    # reconstructed.shape = TensorShape([1, 30, 30, 192])

    """ Handle the cut values in the last patch """
    ww_cut = patches[:, -1:, strides:, :, :strides, :]
    hh_cut = patches[:, :, :strides, -1:, strides:, :]
    corner_cut = patches[:, -1:, strides:, -1:, strides:, :]
    print(f"{ww_cut.shape = }, {hh_cut.shape = }, {corner_cut.shape = }")
    # ww_cut.shape = TensorShape([1, 1, 8, 15, 2, 192]), hh_cut.shape = TensorShape([1, 15, 2, 1, 8, 192]), corner_cut.shape = TensorShape([1, 1, 8, 1, 8, 192])

    ww_cut = tf.reshape(ww_cut, [-1, kernel_size - strides, ww * kww, cc])
    hh_cut = tf.reshape(hh_cut, [-1, ww * kww, kernel_size - strides, cc])
    corner_cut = tf.reshape(corner_cut, [-1, kernel_size - strides, kernel_size - strides, cc])
    print(f"{ww_cut.shape = }, {hh_cut.shape = }, {corner_cut.shape = }")
    # ww_cut.shape = TensorShape([1, 8, 30, 192]), hh_cut.shape = TensorShape([1, 30, 8, 192]), corner_cut.shape = TensorShape([1, 8, 8, 192])

    """ Attach them to the reconstructed """
    reconstructed = tf.concat([reconstructed, ww_cut], axis=1)
    new_hh = tf.concat([hh_cut, corner_cut], axis=1)
    reconstructed = tf.concat([reconstructed, new_hh], axis=2)
    print(f"{reconstructed.shape = }")
    # reconstructed.shape = TensorShape([1, 38, 38, 192])

    print(f"{np.allclose(reconstructed[:, padded: padded + img.shape[1], padded: padded + img.shape[2], :], img) = }")
    # np.allclose(reconstructed[:, padded: padded + img.shape[1], padded: padded + img.shape[2], :], img) = True
    ```
## Volo outlook attention
  ```py
  kernel_size, padding, stride, num_heads, embed_dim = 3, 1, 2, 6, 192
  aa = np.ones([1, 28, 28, 192], dtype="float32")
  ww, hh = int(np.ceil(aa.shape[1] / stride)), int(np.ceil(aa.shape[2] / stride)) # 14, 14
  attn = np.random.uniform(size=[1, ww, hh, kernel_size ** 4 * num_heads]).astype("float32")
  qk_scale = np.sqrt(embed_dim // num_heads)
  num_head, strides = num_heads, stride

  """ PyTorch unfold """
  import torch

  torch_inputs = torch.from_numpy(aa).permute(0, 3, 1, 2) # B, C, H, W
  unfold = torch.nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
  fold = torch.nn.Fold(output_size=torch_inputs.shape[2:4], kernel_size=kernel_size, padding=padding, stride=stride)

  torch_unfold = unfold(torch_inputs) # [1, 1728, 196]
  torch_vv = torch_unfold.reshape(1, num_heads, embed_dim // num_heads, kernel_size * kernel_size, ww * hh) # [1, 6, 32, 9, 196]
  torch_vv = torch_vv.permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H [1, 6, 196, 9, 32]

  """ PyTorch attention """
  # attn = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [1, 14, 14, 192]
  torch_attn = torch.from_numpy(attn)
  torch_attn = torch_attn.reshape(1, ww * hh, num_heads, kernel_size * kernel_size, kernel_size * kernel_size).permute(0, 2, 1, 3, 4)  #
  torch_attn = torch_attn / qk_scale
  torch_attn = torch_attn.softmax(dim=-1) # [1, 6, 196, 9, 9]

  """ PyTorch fold """
  torch_before_fold_1 = (torch_attn @ torch_vv)  # [1, 6, 196, 9, 32]
  torch_before_fold_1 = torch_before_fold_1.permute(0, 1, 4, 3, 2)  # [1, 6, 32, 9, 196]
  torch_before_fold = torch_before_fold_1.reshape(1, embed_dim * kernel_size * kernel_size, hh * ww) # [1, 1728, 196]
  # 196 == ceil(aa.shape[1] / stride) * ceil(aa.shape[2] / stride), 1728 == 192 * kernel_size * kernel_size
  xx = fold(torch_before_fold)  # [1, 192, 28, 28]
  xx = xx.permute(0, 2, 3, 1) # [1, 28, 28, 192]
  torch_out = xx.numpy()

  """ TF unfold """
  img_padded = keras.layers.ZeroPadding2D(1)(aa)
  tf_unfold = tf.image.extract_patches(img_padded, sizes=[1, 3, 3, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='VALID') # [1, 14, 14, 1728]

  print(f"{np.allclose(torch_unfold.permute(0, 2, 1).reshape(*tf_unfold.shape)[0, :, :, 0], tf_unfold[0, :, :, 0]) = }")
  # np.allclose(torch_unfold.permute(0, 2, 1).reshape(*tf_unfold.shape)[0, :, :, 0], tf_unfold[0, :, :, 0]) = True

  tf_vv = tf.reshape(tf_unfold, [-1, ww, hh, kernel_size * kernel_size, num_head, embed_dim // num_head]) # [1, 14, 14, 9, 6, 32], the last dimension is channel
  print(f"{np.allclose(torch_vv.permute(0, 2, 3, 1, 4).reshape(*tf_vv.shape), tf_vv) = }")
  # np.allclose(torch_vv.permute(0, 2, 3, 1, 4).reshape(*tf_vv.shape), tf_vv) = True

  """ TF attention """
  tf_attn = tf.reshape(attn, (-1, ww, hh, num_head, kernel_size * kernel_size, kernel_size * kernel_size)) / qk_scale # [1, 14, 14, 6, 9, 9]
  attention_weights = tf.nn.softmax(tf_attn, axis=-1)  # [1, 14, 14, 6, 9, 9]
  print(f"{np.allclose(torch_attn.permute(0, 2, 1, 3, 4).reshape(*attention_weights.shape), attention_weights) = }")
  # np.allclose(torch_attn.permute(0, 2, 1, 3, 4).reshape(*attention_weights.shape), attention_weights) = True

  """ TF fold """
  tf_before_fold_1 = tf.matmul(attention_weights, tf.transpose(tf_vv, [0, 1, 2, 4, 3, 5]))  # [1, 14, 14, 6, 9, 32],  The last two dimensions [9, 9] @ [9, 32] --> [9, 32]
  tf_before_fold_1 = tf.transpose(tf_before_fold_1, [0, 1, 2, 4, 3, 5]) # [1, 14, 14, 9, 6, 32]
  print(f"{np.allclose(torch_before_fold_1.permute(0, 4, 3, 1, 2).reshape(*tf_before_fold_1.shape), tf_before_fold_1, atol=1e-5) = }")
  # np.allclose(torch_before_fold_1.permute(0, 4, 3, 1, 2).reshape(*tf_before_fold_1.shape), tf_before_fold_1, atol=1e-5) = True

  tf_before_fold = tf.reshape(tf_before_fold_1, [-1, ww, hh, kernel_size * kernel_size * embed_dim]) # [1, 14, 14, 1728]
  tf_out = fold_overlap_1(tf_before_fold, aa.shape[1:3], kernel_size, strides, head_padding=[1, 1]) # [1, 28, 28, 192]
  print(f"{np.allclose(torch_out, tf_out) = }")
  # np.allclose(torch_out, tf_out) = True
  ```
## Volo load torch weights
  - [Github sail-sg/volo](https://github.com/sail-sg/volo)
  ```py
  import torch
  from torchsummary import summary
  from models import volo

  net = volo.volo_d1()
  net.eval()

  summary(net, (3, 224, 224))
  traced_cell = torch.jit.trace(net, (torch.randn(10, 3, 224, 224)))
  torch.jit.save(traced_cell, 'd1.pth')

  # RuntimeError: Exporting the operator col2im to ONNX opset version 13 is not supported.
  # torch.onnx.export(net, torch.randn(10, 3, 224, 224), "d1.onnx", verbose=False, keep_initializers_as_inputs=True, training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=True, opset_version=13)
  ```
  ```py
  import torch
  from torchsummary import summary

  sys.path.append('../volo')
  import models.volo as torch_volo

  model_path = "../models/volo/d1_224_84.2.pth.tar"
  model_type = "volo_" + os.path.basename(model_path).split("_")[0]
  input_shape = int(os.path.basename(model_path).split("_")[1])
  print(f">>>> {model_path = }, {model_type = }, {input_shape = }")

  torch_model = getattr(torch_volo, model_type)(img_size=input_shape)
  torch_model.eval()

  summary(torch_model, (3, input_shape, input_shape))

  from utils import load_pretrained_weights
  load_pretrained_weights(torch_model, model_path, use_ema=False, strict=True, num_classes=1000)

  torch_params = {kk: np.cumproduct(vv.shape)[-1] for kk, vv in torch_model.state_dict().items() if ".num_batches_tracked" not in kk}
  print("torch_model total_parameters :", np.sum(list(torch_params.values())))

  import volo
  mm = getattr(volo, model_type)(input_shape=(input_shape, input_shape, 3), classfiers=2, num_classes=1000)
  keras_params = {ii.name: int(sum([np.cumproduct(jj.shape)[-1] for jj in ii.weights])) for ii in mm.layers}
  keras_params = {kk: vv for kk, vv in keras_params.items() if vv != 0}
  print("keras_model total_parameters :", np.sum(list(keras_params.values())))

  input_output_rr = {
      "patch_embed.conv.0" : "stem_1_conv",
      'patch_embed.conv.1': 'stem_1_bn',
      'patch_embed.conv.3': 'stem_2_conv',
      "patch_embed.conv.4": "stem_2_bn",
      "patch_embed.conv.6": "stem_3_conv",
      "patch_embed.conv.7": "stem_3_bn",
      "patch_embed.proj": "stem_patch_conv",
      "norm": "pre_out_LN",
      "head": "token_head",
      "aux_head": "aux_head",
      "cls_token": "class_token",
      "pos_embed": "stack_0_positional",
      "network.1.proj": "stack_0_downsample",
  }
  network_stack_rr = {'0': 'stack0_', '2': 'stack1_', '3': 'stack2_', '4': 'stack3_'}
  network_block_rr = {"{}".format(ii): "block{}_".format(ii) for ii in range(30)}
  layer_rr = {
      "norm1": "LN",
      "attn.v": "attn_v",
      "attn.q": "attn_q",
      "attn.kv": "attn_kv",
      "attn.qkv": "attn_qkv",
      "attn.attn": "attn_attn",
      "attn.proj": "attn_out",
      "norm2": "mlp_LN",
      "mlp.fc1": "mlp_dense_1",
      "mlp.fc2": "mlp_dense_2",
  }
  post_network_block_rr = {"0": "classfiers0_", "1": "classfiers1_"}

  def match_layer_name(torch_layer_name):
      splitted_name = torch_layer_name.split('.')
      layer_name = ".".join(splitted_name[:-1] if len(splitted_name) > 1 else splitted_name)
      if layer_name in input_output_rr:
           return input_output_rr[layer_name]
      elif splitted_name[0] == "network":
          stack_nn, block_nn = splitted_name[1], splitted_name[2]
          layer_nn = ".".join(splitted_name[3:-1])
          return "".join([network_stack_rr[stack_nn], network_block_rr[block_nn], layer_rr[layer_nn]])
      elif splitted_name[0] == "post_network":
          block_nn = splitted_name[1]
          layer_nn = ".".join(splitted_name[2:-1])
          return "".join([post_network_block_rr[block_nn], layer_rr[layer_nn]])
      else:
          return None

  aa = torch_model.state_dict()
  bb = {ii: match_layer_name(ii) for ii in aa.keys()}
  cc = set(bb.values())
  print("TF layers not contained in torch:", [ii.name for ii in mm.layers if ii.name not in cc])
  print("torch layers not contained in TF:", [ii for ii in cc if ii not in keras_params])
  # torch layers not contained in TF: []

  dd = {kk: (aa[kk].shape, mm.get_layer(vv).weights[0 if "weight" in kk else -1].shape) for kk, vv in bb.items() if "num_batches_tracked" not in kk}
  # 'patch_embed.conv.0.weight': (torch.Size([64, 3, 7, 7]), TensorShape([7, 7, 3, 64])),
  # 'network.0.0.attn.attn.weight': (torch.Size([486, 192]), TensorShape([192, 486])),
  # 'network.0.0.attn.proj.weight': (torch.Size([192, 192]), TensorShape([192, 192])),

  tf_weights_dict = {"weight": 0, "bias": 1, "running_mean": 2, "running_var": 3, "pos_embed": 0, "cls_token": 0}
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
              torch_weight = torch_weight.T

      tf_weights[tf_weight_pos] = torch_weight
      tf_layer.set_weights(tf_weights)

  save_path = os.path.basename(model_path).replace(".pth.tar", ".h5")
  mm.save(save_path)
  print("Saved model:", save_path)

  torch_out = torch_model(torch.from_numpy(np.ones([1, 3, input_shape, input_shape], dtype='float32'))).detach().numpy()
  keras_out = mm(np.ones([1, input_shape, input_shape, 3], dtype='float32'))
  print(f"{np.allclose(torch_out, keras_out, atol=5e-3) = }")
  ```
  ```py
  import volo
  import os
  from tensorflow import keras
  import tensorflow as tf
  import sys

  index = 0 if len(sys.argv) == 1 else int(sys.argv[1])

  model_paths = [
      "../models/volo/d1_224_84.2.h5",
      "../models/volo/d1_384_85.2.h5",
      "../models/volo/d2_224_85.2.h5",
      "../models/volo/d2_384_86.0.h5",
      "../models/volo/d3_224_85.4.h5",
      "../models/volo/d3_448_86.3.h5",
      "../models/volo/d4_224_85.7.h5",
      "../models/volo/d4_448_86.79.h5",
      "../models/volo/d5_224_86.10.h5",
      "../models/volo/d5_448_87.0.h5",
      "../models/volo/d5_512_87.07.h5",
  ]
  model_path = model_paths[index]
  model_type = "volo_" + os.path.basename(model_path).split("_")[0]
  input_shape = int(os.path.basename(model_path).split("_")[1])
  print(f">>>> {model_path = }, {model_type = }, {input_shape = }")

  mm = getattr(volo, model_type)(input_shape=(input_shape, input_shape, 3), classfiers=2, num_classes=1000)
  mm.load_weights(model_path)

  from skimage.data import chelsea
  imm = chelsea()  # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, (input_shape, input_shape)) / 255, 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])

  mm.save(model_path)
  print(">>>> saved:", model_path)
  ```
  ```py
  index = 0
  model_paths = [
      "../models/volo/d1_224_84.2.pth.tar",
      "../models/volo/d1_384_85.2.pth.tar",
      "../models/volo/d2_224_85.2.pth.tar",
      "../models/volo/d2_384_86.0.pth.tar",
      "../models/volo/d3_224_85.4.pth.tar",
      "../models/volo/d3_448_86.3.pth.tar",
      "../models/volo/d4_224_85.7.pth.tar",
      "../models/volo/d4_448_86.79.pth.tar",
      "../models/volo/d5_224_86.10.pth.tar",
      "../models/volo/d5_448_87.0.pth.tar",
      "../models/volo/d5_512_87.07.pth.tar",
  ]

  model_path = model_paths[index]
  model_type = "volo_" + os.path.basename(model_path).split("_")[0]
  input_shape = int(os.path.basename(model_path).split("_")[1])
  keras_model_path = model_path.replace(".pth.tar", ".h5")
  print(f">>>> {model_path = }, {keras_model_path = }, {model_type = }, {input_shape = }")

  import torch
  sys.path.append('../volo')
  import models.volo as torch_volo
  from utils import load_pretrained_weights
  torch_model = getattr(torch_volo, model_type)(img_size=input_shape)
  torch_model.eval()
  load_pretrained_weights(torch_model, model_path, use_ema=False, strict=True, num_classes=1000)

  import volo
  mm = keras.models.load_model(keras_model_path)

  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{(np.abs(torch_out - keras_out) < 1e-3).sum() / keras_out.shape[-1] = }")
  print(f"{(np.abs(torch_out - keras_out) < 5e-3).sum() / keras_out.shape[-1] = }")
  print(f"{(np.abs(torch_out - keras_out) < 1e-2).sum() / keras_out.shape[-1] = }")
  print(f"{(np.abs(torch_out - keras_out) < 5e-2).sum() / keras_out.shape[-1] = }")
  ```
## Volo check
  ```py
  """ PyTorch """
  torch_aa = torch.from_numpy(np.ones([1, 3, 224, 224], dtype='float32'))
  outlooker = torch_model.network[0][0]
  oa = outlooker.attn
  torch_before_attn = outlooker.norm1(torch_model.forward_embeddings(torch_aa))
  torch_attn = (oa.attn(oa.pool(torch_before_attn.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)).reshape([1, 196, 6, 9, 9]).permute(0, 2, 1, 3, 4) * oa.scale).softmax(-1)

  torch_vv = oa.unfold(oa.v(torch_before_attn).permute(0, 3, 1, 2)).reshape([1, 6, 32, 9, 196]).permute(0, 1, 4, 3, 2)

  torch_before_fold = (torch_attn @ torch_vv).permute(0, 1, 4, 3, 2).reshape(1, 1728, 196)
  torch_fold = torch.nn.functional.fold(torch_before_fold, output_size=(28, 28), kernel_size=3, padding=1, stride=2)

  """ TF """
  tf_aa = tf.ones((1, 224, 224, 3))
  tf_vv = keras.models.Model(mm.inputs[0], mm.get_layer('tf.reshape').output)(tf_aa)
  tf_attn = keras.models.Model(mm.inputs[0], mm.get_layer('tf.nn.softmax').output)(tf_aa)
  tf_before_fold = tf.reshape(tf.matmul(tf_vv, tf_attn, transpose_b=True), [-1, 196, 1728])
  tf_fold = mm.get_layer('torch_fold')(tf_before_fold)

  print(f"{np.allclose(tf_vv, torch_vv.permute(0, 2, 1, 4, 3).detach(), atol=1e-2) = }")
  print(f"{np.allclose(tf_attn, torch_attn.permute(0, 2, 1, 3, 4).detach(), atol=1e-3) = }")
  print(f"{np.allclose(tf_before_fold, torch_before_fold.permute(0, 2, 1).detach(), atol=1e-2) = }")
  print(f"{np.allclose(tf_fold, torch_fold.permute(0, 2, 3, 1).detach(), atol=1e-2) = }")
  ```
  ```py
  torch_outlooker_out = torch_model.network[0][0](torch_model.forward_embeddings(torch_aa)).detach()
  tf_block_0_out = keras.models.Model(mm.inputs[0], mm.get_layer('add_1').output)(tf_aa)
  print(f"{np.allclose(torch_outlooker_out, tf_block_0_out, atol=5e-2) = }")

  torch_network_out = torch_model.forward_tokens(torch_model.forward_embeddings(torch_aa)).detach().numpy()
  tf_stack_out = keras.models.Model(mm.inputs[0], mm.get_layer('tf.reshape_40').output)(tf_aa).numpy()
  print(f"{np.allclose(torch_network_out, tf_stack_out, atol=4e-1) = }, {(np.abs(torch_network_out - tf_stack_out) > 5e-2).sum() = }")
  # np.allclose(torch_network_out, tf_stack_out, atol=4e-1) = True, (np.abs(torch_network_out - tf_stack_out) > 5e-2).sum() = 686

  torch_xx = torch_model.norm(torch_model.forward_cls(torch_model.forward_tokens(torch_model.forward_embeddings(torch_aa))))
  tf_xx = keras.models.Model(mm.inputs[0], mm.get_layer('pre_out_LN').output)(tf_aa).numpy()
  print(f"{np.allclose(torch_xx.detach().numpy(), tf_xx, atol=1e-2) = }")
  ```
***
