- [Github alibaba/lightweight-neural-architecture-search](https://github.com/alibaba/lightweight-neural-architecture-search)
- Weights not right, and performance too bad, not adding
  | Model      | Params | FLOPs | Top1 Acc | T4 inference |      |
  | ---------- | ------ | ----- | -------- | ------------ | ---- |
  | DeepMAD18M | 11.69  | 1.82  | 77.7     | 613.717 qps  | 77.7 |
  | DeepMAD34M | 21.80  | 3.68  | 79.7     | 398.34 qps   | 79.7 |
  | DeepMAD50M | 25.55  | 4.13  | 80.6     | 351.184 qps  | 80.6 |

- Code
  ```py
  from keras_cv_attention_models import backend
  from keras_cv_attention_models.backend import layers, models, functional, image_data_format
  from keras_cv_attention_models.models import register_model
  from keras_cv_attention_models.attention_layers import (
      activation_by_name,
      batchnorm_with_activation,
      conv2d_no_bias,
      depthwise_conv2d_no_bias,
      drop_block,
      output_block,
      add_pre_post_process,
  )
  from keras_cv_attention_models.download_and_load import reload_model_weights

  PRETRAINED_DICT = {
      "deepmad_18": {"imagenet": "a4e110248bea5073bfa28cc5c5d747c2"},
      "deepmad_34": {"imagenet": "e058aa1be7d855497c919f5016632fa3"},
      "deepmad_50": {"imagenet": "a3e86e61e9202da792f663310c434560"},
  }


  def se_module(inputs, se_ratio=0.25, divisor=8, use_bias=False, activation="swish", name=""):
      """Using batch_norm, different from traditional one"""
      channel_axis = -1 if image_data_format() == "channels_last" else 1
      h_axis, w_axis = [1, 2] if image_data_format() == "channels_last" else [2, 3]

      filters = inputs.shape[channel_axis]
      reduction = max(1, int(filters * se_ratio) // divisor) * divisor
      # print(f"{filters = }, {se_ratio = }, {divisor = }, {reduction = }")
      se = functional.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
      se = conv2d_no_bias(se, reduction, use_bias=use_bias, name=name + "1_")
      se = batchnorm_with_activation(se, activation=activation, name=name + "1_")
      se = conv2d_no_bias(se, filters, use_bias=use_bias, name=name + "2_")
      se = batchnorm_with_activation(se, activation="sigmoid", name=name + "2_")
      return inputs * se


  def inverted_residual_block(
      inputs, output_channel, hidden_channel, strides=1, proj_shortcut=False, kernel_size=5, se_ratio=0, use_dw=False, drop_rate=0, activation="relu", name="",
  ):
      if proj_shortcut:
          input_channel = inputs.shape[-1 if backend.is_channels_last() else 1]
          shortcut = layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
          shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_") if input_channel != output_channel else shortcut
          shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shortcut_")
      else:
          shortcut = inputs

      nn = conv2d_no_bias(inputs, hidden_channel, name=name + "1_")
      nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
      if use_dw:
          nn = depthwise_conv2d_no_bias(nn, kernel_size=kernel_size, strides=strides, padding="SAME", name=name + "2_")
      else:
          nn = conv2d_no_bias(nn, hidden_channel, kernel_size=kernel_size, strides=strides, padding="SAME", name=name + "2_")
      nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
      if se_ratio > 0:
          nn = se_module(nn, se_ratio, name=name + "se_")

      nn = conv2d_no_bias(nn, output_channel, name=name + "3_")
      nn = batchnorm_with_activation(nn, activation=None, name=name + "3_")

      nn = drop_block(nn, drop_rate, name=name + "drop")
      nn = layers.Add()([shortcut, nn])
      return layers.Activation(activation, name=name + "output")(nn)


  def DeepMAD(
      num_blocks=[1, 6, 9, 10, 12],
      out_channels=[96, 384, 384, 1024, 672],
      hidden_channels=[40, 40, 64, 104, 104],
      kernel_sizes=[5, 5, 5, 3, 5],
      strides=[2, 2, 2, 1, 2],
      stem_width=32,
      se_ratio=0,
      use_dw=False,
      output_conv_filter=0,
      input_shape=(224, 224, 3),
      num_classes=1000,
      activation="relu",
      drop_connect_rate=0,
      dropout=0,
      layer_scale=1e-6,
      classifier_activation="softmax",
      pretrained=None,
      model_name="deepmad",
      kwargs=None,
  ):
      # Regard input_shape as force using original shape if len(input_shape) == 4,
      # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
      input_shape = backend.align_input_shape_by_image_data_format(input_shape)
      inputs = layers.Input(input_shape)

      """ Stem """
      nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="SAME", name="stem_")
      nn = batchnorm_with_activation(nn, activation="relu", name="stem_")
      pre_out_channel = stem_width

      """ stacks """
      total_blocks = sum(num_blocks)
      global_block_id = 0
      for stack_id, (num_block, out_channel, hidden_channel, stride) in enumerate(zip(num_blocks, out_channels, hidden_channels, strides)):
          stack_name = "stack{}_".format(stack_id + 1)
          kernel_size = kernel_sizes[stack_id] if isinstance(kernel_sizes, (list, tuple)) else kernel_sizes
          for block_id in range(num_block):
              name = stack_name + "block{}_".format(block_id + 1)
              cur_stride = stride if block_id == 0 else 1
              proj_shortcut = True if out_channel != pre_out_channel or cur_stride != 1 or block_id % 4 == 0 else False
              block_drop_rate = drop_connect_rate * global_block_id / total_blocks
              nn = inverted_residual_block(
                  nn, out_channel, hidden_channel, cur_stride, proj_shortcut, kernel_size, se_ratio, use_dw, block_drop_rate, activation=activation, name=name
              )
              pre_out_channel = out_channel
              global_block_id += 1

      """  Output head """
      nn = output_block(nn, output_conv_filter, activation="relu", num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
      model = models.Model(inputs, nn, name=model_name)
      add_pre_post_process(model, rescale_mode="torch")
      reload_model_weights(model, PRETRAINED_DICT, "deepmad", pretrained)
      return model


  @register_model
  def DeepMAD18(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
      return DeepMAD(**locals(), model_name="deepmad_18", **kwargs)


  @register_model
  def DeepMAD34(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
      num_blocks = [1, 10, 13, 11, 14]
      out_channels = [128, 512, 896, 1024, 616]
      hidden_channels = [24, 48, 88, 104, 128]
      kernel_sizes = 5
      stem_width = 24
      return DeepMAD(**locals(), model_name="deepmad_34", **kwargs)


  @register_model
  def DeepMAD50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
      num_blocks = [1, 11, 12, 13, 15]
      out_channels = [128, 512, 872, 1024, 1232]
      hidden_channels = [40, 48, 88, 104, 120]
      kernel_sizes = 5
      stem_width = 32
      return DeepMAD(**locals(), model_name="deepmad_50", **kwargs)


  @register_model
  def DeepMAD29M_224(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained=None, **kwargs):
      num_blocks = [6, 7, 6, 3, 1]
      out_channels = [84, 132, 198, 297, 464]
      strides = [1, 2, 2, 2, 2]
      hidden_channels = [504, 792, 1184, 1784, 2784]
      kernel_sizes = [5, 5, 5, 5 ,3]
      stem_width = 32
      se_ratio = 0.25
      use_dw = True
      output_conv_filter = 1792
      return DeepMAD(**locals(), model_name="deepmad_29m_224", **kwargs)


  @register_model
  def DeepMAD29M_288(input_shape=(288, 288, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained=None, **kwargs):
      num_blocks = [4, 4, 5, 6, 4]
      out_channels = [61, 96, 145, 217, 328]
      strides = [2, 2, 2, 1, 2]
      hidden_channels = [368, 576, 872, 1304, 1968]
      kernel_sizes = 5
      stem_width = 40
      se_ratio = 0.25
      use_dw = True
      output_conv_filter = 1792
      return DeepMAD(**locals(), model_name="deepmad_29m_288", **kwargs)


  @register_model
  def DeepMAD50M(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained=None, **kwargs):
      num_blocks = [7, 9, 7, 3, 1]
      out_channels = [112, 168, 252, 378, 576]
      strides = [2, 2, 2, 2, 1]
      hidden_channels = [672, 1008, 1512, 2272, 3456]
      kernel_sizes = [5, 5, 5, 5 ,3]
      stem_width = 48
      se_ratio = 0.25
      use_dw = True
      output_conv_filter = 2048
      return DeepMAD(**locals(), model_name="deepmad_50m", **kwargs)


  @register_model
  def DeepMAD89M(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained=None, **kwargs):
      num_blocks = [10, 9, 7, 4, 1]
      out_channels = [136, 204, 326, 490, 744]
      strides = [2, 2, 2, 2, 1]
      hidden_channels = [816, 1224, 1952, 2944, 4464]
      kernel_sizes = [5, 5, 5, 5 ,3]
      stem_width = 56
      se_ratio = 0.25
      use_dw = True
      output_conv_filter = 2560
      return DeepMAD(**locals(), model_name="deepmad_89m", **kwargs)
  ```
