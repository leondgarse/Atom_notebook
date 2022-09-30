# ___2022 - 08 - 03 Decomposition___
***

# SVD
  - [Github IntelLabs/distiller truncated_svd.ipynb](https://github.com/IntelLabs/distiller/blob/master/jupyter/truncated_svd.ipynb)
  ```py
  # __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
  class SVDDecompositionDense(tf.keras.layers.Dense):
      def __init__(self, units, ratio=1.0, **kwargs):
          super().__init__(units=units, **kwargs)
          self.ratio = ratio

      def build(self, input_shape):
          input_channel = input_shape[-1]
          hidden_units = int((input_channel * self.units * self.ratio) / (input_channel + self.units))
          self.left_ww = self.add_weight(
              name="left_weight",
              shape=(input_channel, hidden_units),
              initializer=self.kernel_initializer,
              regularizer=self.kernel_regularizer,
              constraint=self.kernel_constraint,
              dtype=self.dtype,
              trainable=True,
          )
          self.right_ww = self.add_weight(
              name="right_weight",
              shape=(hidden_units, self.units),
              initializer=self.kernel_initializer,
              regularizer=self.kernel_regularizer,
              constraint=self.kernel_constraint,
              dtype=self.dtype,
              trainable=True,
          )

          if self.use_bias:
              self.bias = self.add_weight(
                  'bias',
                  shape=[self.units,],
                  initializer=self.bias_initializer,
                  regularizer=self.bias_regularizer,
                  constraint=self.bias_constraint,
                  dtype=self.dtype,
                  trainable=True
              )
          self.hidden_units = hidden_units
          self.built = True

      def call(self, inputs, training=None, **kwargs):
          left = tf.matmul(inputs, self.left_ww)
          outputs = tf.matmul(left, self.right_ww)

          if self.use_bias:
              outputs = tf.nn.bias_add(outputs, self.bias)

          if self.activation is not None:
              outputs = self.activation(outputs)
          return outputs

      def get_config(self):
          config = super().get_config()
          config.update({"ratio": self.ratio})
          return config

      def set_decomposed_weights(self, source_weights):
          source_ww = source_weights[0]

          ss, uu, vv = tf.linalg.svd(source_ww)
          ss, uu, vv = ss[:self.hidden_units], uu[:, :self.hidden_units], vv[:, :self.hidden_units]
          ss_vv = tf.matmul(tf.linalg.diag(ss), vv, adjoint_b=True)
          self.left_ww.assign(uu)
          self.right_ww.assign(ss_vv)
          if self.use_bias and len(source_weights) > 1:
              self.bias.assign(source_weights[1])
  ```
  ```py
  aa = keras.layers.Dense(32)
  aa.build([10, 64])
  print(aa(tf.ones([1, 64])))
  # [[-1.467143   -0.6310908  -2.2083158   0.20878458 -0.15939099  0.39235908
  #   -0.43526876 -0.17894727  0.45390975 -1.6907663  -0.48900402  1.0123206
  #    2.0042305   0.5704645   0.08754021  1.2944189   0.20613146 -0.6944201
  #    3.8866606   1.7557755   0.7844685  -1.2994835   1.2929428  -1.3100526
  #    0.802915    1.7859     -0.5365555  -0.3312707   0.62369776 -2.846467
  #   -1.4553483   0.9249731 ]], shape=(1, 32), dtype=float32)

  bb = SVDDecompositionDense(32)
  bb.build([10, 64])
  bb.set_decomposed_weights(aa.get_weights())
  print(f"{bb.hidden_units = }")
  # bb.hidden_units = 21

  print(bb(tf.ones([1, 64])))
  # [[-1.1220691  -0.84883225 -2.6551738  -0.4942645  -0.11023892  0.44805747
  #   -0.2649902  -0.41182667  0.15110652 -1.8178451   0.12451547  1.5191154
  #    1.3587747   0.08355719 -0.12822032  1.6146692   0.29505682 -0.616733
  #    3.3996115   2.3625576   0.48977804 -0.85943604  1.2775613  -1.3054411
  #    0.51417243  1.9973481  -0.53608227 -0.2776103   0.25374067 -2.462182
  #   -1.2069365   0.6259171 ]], shape=(1, 32), dtype=float32)

  bb = SVDDecompositionDense(32, ratio=0.5)
  bb.build([10, 64])
  bb.set_decomposed_weights(aa.get_weights())
  print(f"{bb.hidden_units = }")
  # bb.hidden_units = 10
  print(bb(tf.ones([1, 64])))
  # [[-0.3058235   0.6975229  -2.1448812  -0.3182568   0.28915346  0.6349824
  #    0.16195928  0.718627   -0.04990605 -0.3511393  -0.27716663  1.2024267
  #    0.80014825  0.7982341   0.54726696  1.2693114   0.02804485 -0.95943004
  #    2.6270747   2.0000093  -0.08941066  0.17918253  1.9232392  -1.0759805
  #   -0.03066717  2.5296388   0.30663323 -0.48439768  0.9957781  -1.8503966
  #   -0.5905285  -0.19627097]], shape=(1, 32), dtype=float32)

  print(f"{(32 * 32 + 32 * 64) / 32 / 64 = }")
  # (32 * 32 + 32 * 64) / 32 / 64 = 1.5
  bb = SVDDecompositionDense(32, ratio=1.5)
  bb.build([10, 64])
  bb.set_decomposed_weights(aa.get_weights())
  print(f"{bb.hidden_units = }")
  # bb.hidden_units = 32
  print(bb(tf.ones([1, 64])))
  # [[-1.4671437  -0.63109344 -2.2083173   0.20878437 -0.15939078  0.39236057
  #   -0.43527165 -0.17894748  0.45391208 -1.6907703  -0.48900807  1.0123203
  #    2.0042346   0.57046247  0.08754128  1.2944196   0.20613207 -0.694421
  #    3.8866644   1.7557802   0.78447074 -1.2994858   1.2929443  -1.3100554
  #    0.8029162   1.7859037  -0.5365571  -0.33127123  0.62369734 -2.8464704
  #   -1.4553529   0.9249739 ]], shape=(1, 32), dtype=float32)
  ```
  **分解后时间测试**
  ```py
  in_channel, out_channel, hidden_channel = 32, 64, 16
  inputs = tf.random.uniform([1, 32, 32, in_channel])
  filters = tf.random.uniform([3, 3, in_channel, out_channel])
  print(tf.nn.conv2d(inputs, filters, 2, 'SAME').shape)
  %timeit tf.nn.conv2d(inputs, filters, 2, 'SAME')
  # 110 µs ± 718 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
  %timeit tf.nn.conv2d(inputs, filters, 2, 'SAME')
  # 110 µs ± 474 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

  hh_1 = tf.random.uniform([1, 1, in_channel, hidden_channel])
  hh_2 = tf.random.uniform([3, 3, hidden_channel, hidden_channel])
  hh_3 = tf.random.uniform([1, 1, hidden_channel, out_channel])
  print(tf.nn.conv2d(tf.nn.conv2d(tf.nn.conv2d(inputs, hh_1, 1, 'VALID'), hh_2, 2, 'SAME'), hh_3, 1, "VALID").shape)
  %timeit tf.nn.conv2d(tf.nn.conv2d(tf.nn.conv2d(inputs, hh_1, 1, 'VALID'), hh_2, 2, 'SAME'), hh_3, 1, "VALID")
  # 246 µs ± 3.25 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
  %timeit tf.nn.conv2d(tf.nn.conv2d(tf.nn.conv2d(inputs, hh_1, 1, 'VALID'), hh_2, 2, 'SAME'), hh_3, 1, "VALID")
  # 249 µs ± 6.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

  print((in_channel * hidden_channel + 3 * 3 * hidden_channel * hidden_channel + hidden_channel * out_channel) / (3 * 3 * in_channel * out_channel))
  # 0.20833333333333334
  ```
  **显存占用**
  ```py
  xx = tf.random.uniform([1000, 32, 32, 3])
  yy = tf.one_hot(tf.random.uniform([1000,], minval=0, maxval=9, dtype=tf.int32), 10)

  mm = keras.models.Sequential([
      keras.layers.Input([32, 32, 3]),
      keras.layers.Dense(768),
      keras.layers.Dense(3024),
      keras.layers.Dense(768),
      keras.layers.Dense(3024),
      keras.layers.Dense(768),
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(10),
  ])

  # mm = keras.models.Sequential([
  #     keras.layers.Input([32, 32, 3]),
  #     keras.layers.Dense(768),
  #     keras.layers.Dense(160),
  #     keras.layers.Dense(3024),
  #     keras.layers.Dense(160),
  #     keras.layers.Dense(768),
  #     keras.layers.Dense(160),
  #     keras.layers.Dense(3024),
  #     keras.layers.Dense(160),
  #     keras.layers.Dense(768),
  #     keras.layers.GlobalAveragePooling2D(),
  #     keras.layers.Dense(10),
  # ])

  mm.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)
  mm.fit(xx, yy, epochs=5, batch_size=256)
  ```
