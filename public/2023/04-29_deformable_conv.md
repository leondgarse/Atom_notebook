## Links
  - [Github miemie2013/Keras-YOLOv4](https://github.com/miemie2013/Keras-YOLOv4)
  - [Github Visual-Behavior/detr-tensorflow](https://github.com/Visual-Behavior/detr-tensorflow/pull/25/files)
  - [Github lucidrains/deformable-attention](https://github.com/lucidrains/deformable-attention)
  - `pip install MultiScaleDeformableAttention`
***

## DCNv2 Deformable Convolution Network
  - [Github miemie2013/Keras-YOLOv4](https://github.com/miemie2013/Keras-YOLOv4)
  ```py
  class DCNv2(Layer):
      '''
      咩酱自实现的DCNv2，咩酱的得意之作，tensorflow的纯python接口实现，效率极高。
      '''
      def __init__(self, input_dim, filters, filter_size, stride=1, padding=0, bias_attr=False, distribution='normal', gain=1, name=''):
          super(DCNv2, self).__init__()
          assert distribution in ['uniform', 'normal']
          self.input_dim = input_dim
          self.filters = filters
          self.filter_size = filter_size
          self.stride = stride
          self.padding = padding
          self.bias_attr = bias_attr

          self.conv_offset_padding = keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))
          self.zero_padding = keras.layers.ZeroPadding2D(padding=((padding, padding+1), (padding, padding+1)))

      def build(self, input_shape):
          input_dim = self.input_dim
          filters = self.filters
          filter_size = self.filter_size
          bias_attr = self.bias_attr
          self.offset_w = self.add_weight('offset_w', shape=[filter_size, filter_size, input_dim, filter_size * filter_size * 3], initializer='zeros')
          self.offset_b = self.add_weight('offset_b', shape=[1, 1, 1, filter_size * filter_size * 3], initializer='zeros')
          self.dcn_weight = self.add_weight('dcn_weight', shape=[filters, input_dim, filter_size, filter_size], initializer='uniform')
          self.dcn_bias = None
          if bias_attr:
              self.dcn_bias = self.add_weight('dcn_bias', shape=[filters, ], initializer='zeros')

      def compute_output_shape(self, input_shape):
          filters = self.filters
          return (None, None, None, filters)

      def call(self, x):
          filter_size = self.filter_size
          stride = self.stride
          padding = self.padding
          dcn_weight = self.dcn_weight
          dcn_bias = self.dcn_bias


          # 当filter_size = 3, stride = 2, padding = 1时， 设置padding2 = 'valid'，K.conv2d层前加一个self.conv_offset_padding
          # 当filter_size = 3, stride = 1, padding = 1时， 设置padding2 = 'same'，K.conv2d层前不用加一个self.conv_offset_padding
          # 无论什么条件，self.zero_padding层都是必须要加的。
          if stride == 2:
              temp = self.conv_offset_padding(x)
          else:
              temp = x
          padding2 = None
          if stride == 2:
              padding2 = 'valid'
          else:
              padding2 = 'same'
          offset_mask = K.conv2d(temp, self.offset_w, strides=(stride, stride), padding=padding2)
          offset_mask += self.offset_b

          offset_mask = tf.transpose(offset_mask, [0, 3, 1, 2])
          offset = offset_mask[:, :filter_size ** 2 * 2, :, :]
          mask = offset_mask[:, filter_size ** 2 * 2:, :, :]
          mask = tf.nn.sigmoid(mask)


          # ===================================
          N = tf.shape(x)[0]
          H = tf.shape(x)[1]
          W = tf.shape(x)[2]
          out_C = tf.shape(dcn_weight)[0]
          in_C = tf.shape(dcn_weight)[1]
          kH = tf.shape(dcn_weight)[2]
          kW = tf.shape(dcn_weight)[3]
          W_f = tf.cast(W, tf.float32)
          H_f = tf.cast(H, tf.float32)
          kW_f = tf.cast(kW, tf.float32)
          kH_f = tf.cast(kH, tf.float32)

          out_W = (W_f + 2 * padding - (kW_f - 1)) // stride
          out_H = (H_f + 2 * padding - (kH_f - 1)) // stride
          out_W = tf.cast(out_W, tf.int32)
          out_H = tf.cast(out_H, tf.int32)
          out_W_f = tf.cast(out_W, tf.float32)
          out_H_f = tf.cast(out_H, tf.float32)

          # 1.先对图片x填充得到填充后的图片pad_x
          pad_x = self.zero_padding(x)
          pad_x = tf.transpose(pad_x, [0, 3, 1, 2])

          # 卷积核中心点在pad_x中的位置
          rows = tf.range(out_W_f, dtype=tf.float32) * stride + padding
          cols = tf.range(out_H_f, dtype=tf.float32) * stride + padding
          rows = tf.tile(rows[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis], [1, out_H, 1, 1, 1])
          cols = tf.tile(cols[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], [1, 1, out_W, 1, 1])
          start_pos_yx = tf.concat([cols, rows], axis=-1)  # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
          start_pos_yx = tf.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
          start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
          start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置

          # 卷积核内部的偏移
          half_W = (kW_f - 1) / 2
          half_H = (kH_f - 1) / 2
          rows2 = tf.range(kW_f, dtype=tf.float32) - half_W
          cols2 = tf.range(kH_f, dtype=tf.float32) - half_H
          rows2 = tf.tile(rows2[tf.newaxis, :, tf.newaxis], [kH, 1, 1])
          cols2 = tf.tile(cols2[:, tf.newaxis, tf.newaxis], [1, kW, 1])
          filter_inner_offset_yx = tf.concat([cols2, rows2], axis=-1)  # [kH, kW, 2]   卷积核内部的偏移
          filter_inner_offset_yx = tf.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
          filter_inner_offset_yx = tf.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
          filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
          filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移

          mask = tf.transpose(mask, [0, 2, 3, 1])       # [N, out_H, out_W, kH*kW*1]
          offset = tf.transpose(offset, [0, 2, 3, 1])   # [N, out_H, out_W, kH*kW*2]
          offset_yx = tf.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
          offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
          offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

          # 最终位置
          pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
          pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
          pos_y = tf.maximum(pos_y, 0.0)
          pos_y = tf.minimum(pos_y, H_f + padding * 2 - 1.0)
          pos_x = tf.maximum(pos_x, 0.0)
          pos_x = tf.minimum(pos_x, W_f + padding * 2 - 1.0)
          ytxt = tf.concat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

          pad_x = tf.transpose(pad_x, [0, 2, 3, 1])  # [N, pad_x_H, pad_x_W, C]

          mask = tf.reshape(mask, (N, out_H, out_W, kH, kW))  # [N, out_H, out_W, kH, kW]

          def _process_sample(args):
              _pad_x, _mask, _ytxt = args
              # _pad_x:    [pad_x_H, pad_x_W, in_C]
              # _mask:     [out_H, out_W, kH, kW]
              # _ytxt:     [out_H, out_W, kH*kW, 2]

              _ytxt = tf.reshape(_ytxt, (out_H * out_W * kH * kW, 2))  # [out_H*out_W*kH*kW, 2]
              _yt = _ytxt[:, :1]
              _xt = _ytxt[:, 1:]
              _y1 = tf.floor(_yt)
              _x1 = tf.floor(_xt)
              _y2 = _y1 + 1.0
              _x2 = _x1 + 1.0
              _y1x1 = tf.concat([_y1, _x1], -1)
              _y1x2 = tf.concat([_y1, _x2], -1)
              _y2x1 = tf.concat([_y2, _x1], -1)
              _y2x2 = tf.concat([_y2, _x2], -1)

              _y1x1_int = tf.cast(_y1x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
              v1 = tf.gather_nd(_pad_x, _y1x1_int)  # [out_H*out_W*kH*kW, in_C]
              _y1x2_int = tf.cast(_y1x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
              v2 = tf.gather_nd(_pad_x, _y1x2_int)  # [out_H*out_W*kH*kW, in_C]
              _y2x1_int = tf.cast(_y2x1, tf.int32)  # [out_H*out_W*kH*kW, 2]
              v3 = tf.gather_nd(_pad_x, _y2x1_int)  # [out_H*out_W*kH*kW, in_C]
              _y2x2_int = tf.cast(_y2x2, tf.int32)  # [out_H*out_W*kH*kW, 2]
              v4 = tf.gather_nd(_pad_x, _y2x2_int)  # [out_H*out_W*kH*kW, in_C]

              lh = _yt - _y1  # [out_H*out_W*kH*kW, 1]
              lw = _xt - _x1
              hh = 1 - lh
              hw = 1 - lw
              w1 = hh * hw
              w2 = hh * lw
              w3 = lh * hw
              w4 = lh * lw
              value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # [out_H*out_W*kH*kW, in_C]
              _mask = tf.reshape(_mask, (out_H * out_W * kH * kW, 1))
              value = value * _mask
              value = tf.reshape(value, (out_H, out_W, kH, kW, in_C))
              value = tf.transpose(value, [0, 1, 4, 2, 3])   # [out_H, out_W, in_C, kH, kW]
              return value

          # 旧的方案，使用逐元素相乘，慢！
          # new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
          # new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))   # [N, out_H, out_W, in_C * kH * kW]
          # new_x = tf.transpose(new_x, [0, 3, 1, 2])  # [N, in_C*kH*kW, out_H, out_W]
          # exp_new_x = tf.reshape(new_x, (N, 1, in_C*kH*kW, out_H, out_W))  # 增加1维，[N,      1, in_C*kH*kW, out_H, out_W]
          # reshape_w = tf.reshape(dcn_weight, (1, out_C, in_C * kH * kW, 1, 1))      # [1, out_C,  in_C*kH*kW,     1,     1]
          # out = exp_new_x * reshape_w                                   # 逐元素相乘，[N, out_C,  in_C*kH*kW, out_H, out_W]
          # out = tf.reduce_sum(out, axis=[2, ])                           # 第2维求和，[N, out_C, out_H, out_W]
          # out = tf.transpose(out, [0, 2, 3, 1])

          # 新的方案，用等价的1x1卷积代替逐元素相乘，快！
          new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
          new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))                # [N, out_H, out_W, in_C * kH * kW]
          tw = tf.transpose(dcn_weight, [1, 2, 3, 0])      # [out_C, in_C, kH, kW] -> [in_C, kH, kW, out_C]
          tw = tf.reshape(tw, (1, 1, in_C*kH*kW, out_C))   # [1, 1, in_C*kH*kW, out_C]  变成1x1卷积核
          out = K.conv2d(new_x, tw, strides=(1, 1), padding='valid')     # [N, out_H, out_W, out_C]
          return out
  ```
  ```py
  from tensorflow.keras import layers

  strides = 2
  kernel_size = 3
  padding = 0
  filters = 64
  inputs = tf.ones([1, 24, 24, 32])

  offset_mask = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs) if strides == 2 else inputs
  offset_mask = layers.Conv2D(kernel_size * kernel_size * 3, kernel_size, strides, kernel_initializer="zeros")(offset_mask)
  offset_y, offset_x, mask = tf.split(offset_mask, 3, axis=-1)
  mask = tf.nn.sigmoid(mask)

  padded = layers.ZeroPadding2D(((padding, padding+1), (padding, padding+1)))(inputs)

  # [TODO] grid
  pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
  pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
  ytxt = tf.concat([pos_y, pos_x], -1)  # [N, out_H, out_W, kH*kW, 2]

  # [TODO] map_fn
  new_x = tf.map_fn(_process_sample, [pad_x, mask, ytxt], dtype=tf.float32)   # [N, out_H, out_W, in_C, kH, kW]
  new_x = tf.reshape(new_x, (N, out_H, out_W, in_C * kH * kW))  # [N, out_H, out_W, in_C * kH * kW]
  out = layers.Conv2D(filters, 1, strides=1, kernel_initializer="zeros")(new_x)

  ```
## Python failback of MSDeformAttnFunction
  - [Github Visual-Behavior/detr-tensorflow](https://github.com/Visual-Behavior/detr-tensorflow/pull/25/files)
  ```py
  import tensorflow as tf

  #Python failback of MSDeformAttnFunction

  def MSDeformAttnFunction(values, sampling_locations, attention_weights):

      # for debug and test only,
      # need to use cuda version instead
      """
      :param values                        level, (N, H, W, num_heads, head_dim)
      :param sampling_locations            level, (N, Len_q, num_heads, num_sampling_points, 2)
      :param attention_weights              N, Len_q, num_heads, num_level, num_sampling_points
      """

      sampling_value_list = []
      for lid_, (value, sl) in enumerate(zip(values, sampling_locations)):
          N, h_l, w_l, num_heads, head_dim = tf.unstack(tf.shape(value))
          # N*num_heads, h, w, c
          value = tf.reshape(tf.transpose(value, [0, 3, 1, 2, 4]), [N*num_heads, h_l, w_l, head_dim])

          # N, Len_q, num_heads, num_sampling_points, 2
          sl = 2 * sl - 1 #between (-1, 1)
          N, Len_q, num_heads, num_sampling_points, _ = tf.unstack(tf.shape(sl))

          # N*num_heads, Len_q, num_sampling_points, 2
          sampling_grid_l_ = tf.reshape(tf.transpose(sl, [0, 2, 1, 3, 4]), [N*num_heads, Len_q, num_sampling_points, 2])

          #N*num_heads, Len_q, num_sampling_points, c
          if True:
              sampled_values = bilinear_sampler(value, sampling_grid_l_)
          else:
              sampled_values = nearest_sampler(value, sampling_grid_l_)

          sampling_value_list.append(sampled_values)

      # N*num_heads, Len_q, num_level, num_sampling_points, c
      sampling_value = tf.stack(sampling_value_list, axis=2)
      # N, num_heads, Len_q, num_level, num_sampling_points, c
      sampling_value = tf.reshape(sampling_value, (N, num_heads, Len_q, len(values), num_sampling_points, head_dim))
      # N, Len_q, num_heads, num_level, num_sampling_points, c
      sampling_value = tf.transpose(sampling_value, [0, 2, 1, 3, 4, 5])
      # (N, Len_q, num_heads, num_level, num_sampling_points, 1)
      attention_weights = tf.expand_dims(attention_weights, -1)
      # N, Len_q, num_heads, num_level, num_sampling_points, c
      output = attention_weights * sampling_value
      # N, Len_q, num_heads, -1, head_dim
      output = tf.reshape(output, (N, Len_q, num_heads, -1, head_dim))
      # N, Len_q, num_heads, c
      output = tf.reduce_sum(output, axis=3)

      output = tf.reshape(output, (N, Len_q, num_heads*head_dim))

      return output


  def within_bounds(x, lower, upper):
      lower_tensor = tf.greater_equal(x, lower)
      upper_tensor = tf.less_equal(x, upper)
      return tf.logical_and(lower_tensor, upper_tensor)

  def bilinear_sampler(image, coords):
      ''' Value sampler using tf.gather_nd
      Args:
        image: tensor with shape (bs, h, w, c)
        coords: coordinates tensor with shape (bs, ... , 2), xy-indexing between 0, 1

      Returns:
        sampled tensor with shape (bs, ... , c)
      '''

      #Correspond to padding="zeros" (optimistic : discard only out of bound bilinear coefficient, not the full value)

      with tf.name_scope("bilinear_sampler"):
        _, h, w, _ = tf.unstack(tf.shape(image))


        gx, gy = tf.unstack(coords, axis=-1)

        # rescale x and y to [0, W-1/H-1]
        gx = (gx+1.0)/2.0  * tf.cast(w-1, tf.float32)
        gy = (gy+1.0)/2.0  * tf.cast(h-1, tf.float32)

        gx0 = tf.floor(gx)
        gx1 = gx0 + 1.0
        gy0 = tf.floor(gy)
        gy1 = gy0 + 1.0

        mx0 = within_bounds(gx0, 0, tf.cast(w, tf.float32)-1)
        mx1 = within_bounds(gx1, 0, tf.cast(w, tf.float32)-1)
        my0 = within_bounds(gy0, 0, tf.cast(h, tf.float32)-1)
        my1 = within_bounds(gy1, 0, tf.cast(h, tf.float32)-1)

        c00 = tf.expand_dims((gy1 - gy)*(gx1 - gx), axis=-1)
        c01 = tf.expand_dims((gy1 - gy)*(gx - gx0), axis=-1)
        c10 = tf.expand_dims((gy - gy0)*(gx1 - gx), axis=-1)
        c11 = tf.expand_dims((gy - gy0)*(gx - gx0), axis=-1)

        #clip for CPU (out_of_bound-error), optionnal on GPU (as corresponding m.. while be zeroed)
        gx0 = tf.clip_by_value(gx0, 0, tf.cast(w, tf.float32)-1)
        gx1 = tf.clip_by_value(gx1, 0, tf.cast(w, tf.float32)-1)
        gy0 = tf.clip_by_value(gy0, 0, tf.cast(h, tf.float32)-1)
        gy1 = tf.clip_by_value(gy1, 0, tf.cast(h, tf.float32)-1)

        g00 = tf.stack([gy0, gx0], axis=-1)
        g01 = tf.stack([gy0, gx1], axis=-1)
        g10 = tf.stack([gy1, gx0], axis=-1)
        g11 = tf.stack([gy1, gx1], axis=-1)

        m00 = tf.cast(tf.expand_dims(tf.logical_and(my0, mx0), axis=-1), tf.float32)
        m01 = tf.cast(tf.expand_dims(tf.logical_and(my0, mx1), axis=-1), tf.float32)
        m10 = tf.cast(tf.expand_dims(tf.logical_and(my1, mx0), axis=-1), tf.float32)
        m11 = tf.cast(tf.expand_dims(tf.logical_and(my1, mx1), axis=-1), tf.float32)

        x00 = tf.gather_nd(image, tf.cast(g00, dtype=tf.int32), batch_dims=1)
        x01 = tf.gather_nd(image, tf.cast(g01, dtype=tf.int32), batch_dims=1)
        x10 = tf.gather_nd(image, tf.cast(g10, dtype=tf.int32), batch_dims=1)
        x11 = tf.gather_nd(image, tf.cast(g11, dtype=tf.int32), batch_dims=1)

        output = c00 * x00 * m00 \
               + c01 * x01 * m01 \
               + c10 * x10 * m10 \
               + c11 * x11 * m11

        return output


  def nearest_sampler(image, coords):
      with tf.name_scope("nearest_sampler"):
          _, h, w, _ = tf.unstack(tf.shape(image))

          gx, gy = tf.unstack(coords, axis=-1)

          # rescale x and y to [0, W-1/H-1]
          gx = (gx+1.0)/2.0  * tf.cast(w-1, tf.float32)
          gy = (gy+1.0)/2.0  * tf.cast(h-1, tf.float32)

          gx0 = tf.round(gx)
          gy0 = tf.round(gy)

          g00 = tf.stack([gy0, gx0], axis=-1)

          return tf.gather_nd(image, tf.cast(g00, dtype=tf.int32), batch_dims=1)



  if __name__ == "__main__":
      import torch
      import torch.nn.functional as F

      import numpy as np

      for i in range(1000):

          test_size = 100

          grid_size = test_size
          feature_len = 1
          batch_size = test_size

          grid_sampling_size = test_size

          values = np.random.rand(batch_size, grid_size, grid_size, feature_len)

          t_values = np.transpose(values, (0, 3, 1, 2) )

          coords = np.random.rand(batch_size, grid_sampling_size, grid_sampling_size, 2) * 2 - 1
          coords = coords * 1.1

          values = values.astype(np.float32)
          coords = coords.astype(np.float32)
          t_values = t_values.astype(np.float32)

          tf_result = bilinear_sampler(values, coords)
          tf_result = tf_result.numpy()

          torch_result = F.grid_sample(torch.from_numpy(t_values), torch.from_numpy(coords),
              mode='bilinear', padding_mode='zeros', align_corners=True)


          torch_result = torch_result.view(batch_size, grid_sampling_size, grid_sampling_size, feature_len).numpy()

          diff = np.abs(tf_result - torch_result)

          print("diff", np.amax(diff), np.unravel_index(diff.argmax(), diff.shape))

          if np.amax(diff) > 1e-3:
              break
  ```
## DeformableAttention2D
  - [Github lucidrains/deformable-attention](https://github.com/lucidrains/deformable-attention)
  ```py
  import torch
  import torch.nn.functional as F
  from torch import nn, einsum

  from einops import rearrange, repeat

  # helper functions

  def exists(val):
      return val is not None

  def default(val, d):
      return val if exists(val) else d

  def divisible_by(numer, denom):
      return (numer % denom) == 0

  # tensor helpers

  def create_grid_like(t, dim = 0):
      h, w, device = *t.shape[-2:], t.device

      grid = torch.stack(torch.meshgrid(
          torch.arange(h, device = device),
          torch.arange(w, device = device),
      indexing = 'ij'), dim = dim)

      grid.requires_grad = False
      grid = grid.type_as(t)
      return grid

  def normalize_grid(grid, dim = 1, out_dim = -1):
      # normalizes a grid to range from -1 to 1
      h, w = grid.shape[-2:]
      grid_h, grid_w = grid.unbind(dim = dim)

      grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
      grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

      return torch.stack((grid_h, grid_w), dim = out_dim)

  class Scale(nn.Module):
      def __init__(self, scale):
          super().__init__()
          self.scale = scale

      def forward(self, x):
          return x * self.scale

  # continuous positional bias from SwinV2

  class CPB(nn.Module):
      """ https://arxiv.org/abs/2111.09883v1 """

      def __init__(self, dim, *, heads, offset_groups, depth):
          super().__init__()
          self.heads = heads
          self.offset_groups = offset_groups

          self.mlp = nn.ModuleList([])

          self.mlp.append(nn.Sequential(
              nn.Linear(2, dim),
              nn.ReLU()
          ))

          for _ in range(depth - 1):
              self.mlp.append(nn.Sequential(
                  nn.Linear(dim, dim),
                  nn.ReLU()
              ))

          self.mlp.append(nn.Linear(dim, heads // offset_groups))

      def forward(self, grid_q, grid_kv):
          device, dtype = grid_q.device, grid_kv.dtype

          grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
          grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

          pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
          bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

          for layer in self.mlp:
              bias = layer(bias)

          bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

          return bias

  # main class

  class DeformableAttention2D(nn.Module):
      def __init__(
          self,
          *,
          dim,
          dim_head = 64,
          heads = 8,
          dropout = 0.,
          downsample_factor = 4,
          offset_scale = None,
          offset_groups = None,
          offset_kernel_size = 6,
          group_queries = True,
          group_key_values = True
      ):
          super().__init__()
          offset_scale = default(offset_scale, downsample_factor)
          assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
          assert divisible_by(offset_kernel_size - downsample_factor, 2)

          offset_groups = default(offset_groups, heads)
          assert divisible_by(heads, offset_groups)

          inner_dim = dim_head * heads
          self.scale = dim_head ** -0.5
          self.heads = heads
          self.offset_groups = offset_groups

          offset_dims = inner_dim // offset_groups

          self.downsample_factor = downsample_factor

          self.to_offsets = nn.Sequential(
              nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
              nn.GELU(),
              nn.Conv2d(offset_dims, 2, 1, bias = False),
              nn.Tanh(),
              Scale(offset_scale)
          )

          self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

          self.dropout = nn.Dropout(dropout)
          self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
          self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
          self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
          self.to_out = nn.Conv2d(inner_dim, dim, 1)

      def forward(self, x, return_vgrid = False):
          """
          b - batch
          h - heads
          x - height
          y - width
          d - dimension
          g - offset groups
          """

          heads, b, h, w, downsample_factor, device = self.heads, x.shape[0], *x.shape[-2:], self.downsample_factor, x.device

          # queries

          q = self.to_q(x)

          # calculate offsets - offset MLP shared across all groups

          group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

          grouped_queries = group(q)
          offsets = self.to_offsets(grouped_queries)

          # calculate grid + offsets

          grid =create_grid_like(offsets)
          vgrid = grid + offsets

          vgrid_scaled = normalize_grid(vgrid)

          kv_feats = F.grid_sample(
              group(x),
              vgrid_scaled,
          mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

          kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

          # derive key / values

          k, v = self.to_k(kv_feats), self.to_v(kv_feats)

          # scale queries

          q = q * self.scale

          # split out heads

          q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

          # query / key similarity

          sim = einsum('b h i d, b h j d -> b h i j', q, k)

          # relative positional bias

          grid = create_grid_like(x)
          grid_scaled = normalize_grid(grid, dim = 0)
          rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
          sim = sim + rel_pos_bias

          # numerical stability

          sim = sim - sim.amax(dim = -1, keepdim = True).detach()

          # attention

          attn = sim.softmax(dim = -1)
          attn = self.dropout(attn)

          # aggregate and combine heads

          out = einsum('b h i j, b h j d -> b h i d', attn, v)
          out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
          out = self.to_out(out)

          if return_vgrid:
              return out, vgrid

          return out
  ```
## deformable_attention_core_func
  - [Github PaddlePaddle/PaddleDetection/utils.py](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/utils.py)
  ```py
  def deformable_attention_core_func(value, value_spatial_shapes,
                                     value_level_start_index, sampling_locations,
                                     attention_weights):
      """
      Args:
          value (Tensor): [bs, value_length, n_head, c]
          value_spatial_shapes (Tensor|List): [n_levels, 2]
          value_level_start_index (Tensor|List): [n_levels]
          sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
          attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

      Returns:
          output (Tensor): [bs, Length_{query}, C]
      """
      bs, _, n_head, c = value.shape
      _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

      split_shape = [h * w for h, w in value_spatial_shapes]
      value_list = value.split(split_shape, axis=1)
      sampling_grids = 2 * sampling_locations - 1
      sampling_value_list = []
      for level, (h, w) in enumerate(value_spatial_shapes):
          # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
          value_l_ = value_list[level].flatten(2).transpose(
              [0, 2, 1]).reshape([bs * n_head, c, h, w])
          # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
          sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(
              [0, 2, 1, 3, 4]).flatten(0, 1)
          # N_*M_, D_, Lq_, P_
          sampling_value_l_ = F.grid_sample(
              value_l_,
              sampling_grid_l_,
              mode='bilinear',
              padding_mode='zeros',
              align_corners=False)
          sampling_value_list.append(sampling_value_l_)
      # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
      attention_weights = attention_weights.transpose([0, 2, 1, 3, 4]).reshape(
          [bs * n_head, 1, Len_q, n_levels * n_points])
      output = (paddle.stack(
          sampling_value_list, axis=-2).flatten(-2) *
                attention_weights).sum(-1).reshape([bs, n_head * c, Len_q])

      return output.transpose([0, 2, 1])
  ```
***
## Grid sample
```py
def bilinear_sampler(image, coords):
    ''' Value sampler using tf.gather_nd
    Args:
      image: tensor with shape (bs, h, w, c)
      coords: coordinates tensor with shape (bs, ... , 2), xy-indexing between 0, 1

    Returns:
      sampled tensor with shape (bs, ... , c)
    '''

    #Correspond to padding="zeros" (optimistic : discard only out of bound bilinear coefficient, not the full value)

    with tf.name_scope("bilinear_sampler"):
      _, h, w, _ = tf.unstack(tf.shape(image))


      gx, gy = tf.unstack(coords, axis=-1)

      # rescale x and y to [0, W-1/H-1]
      gx = (gx+1.0)/2.0  * tf.cast(w-1, tf.float32)
      gy = (gy+1.0)/2.0  * tf.cast(h-1, tf.float32)

      gx0 = tf.floor(gx)
      gx1 = gx0 + 1.0
      gy0 = tf.floor(gy)
      gy1 = gy0 + 1.0

      mx0 = within_bounds(gx0, 0, tf.cast(w, tf.float32)-1)
      mx1 = within_bounds(gx1, 0, tf.cast(w, tf.float32)-1)
      my0 = within_bounds(gy0, 0, tf.cast(h, tf.float32)-1)
      my1 = within_bounds(gy1, 0, tf.cast(h, tf.float32)-1)

      c00 = tf.expand_dims((gy1 - gy)*(gx1 - gx), axis=-1)
      c01 = tf.expand_dims((gy1 - gy)*(gx - gx0), axis=-1)
      c10 = tf.expand_dims((gy - gy0)*(gx1 - gx), axis=-1)
      c11 = tf.expand_dims((gy - gy0)*(gx - gx0), axis=-1)

      #clip for CPU (out_of_bound-error), optionnal on GPU (as corresponding m.. while be zeroed)
      gx0 = tf.clip_by_value(gx0, 0, tf.cast(w, tf.float32)-1)
      gx1 = tf.clip_by_value(gx1, 0, tf.cast(w, tf.float32)-1)
      gy0 = tf.clip_by_value(gy0, 0, tf.cast(h, tf.float32)-1)
      gy1 = tf.clip_by_value(gy1, 0, tf.cast(h, tf.float32)-1)

      g00 = tf.stack([gy0, gx0], axis=-1)
      g01 = tf.stack([gy0, gx1], axis=-1)
      g10 = tf.stack([gy1, gx0], axis=-1)
      g11 = tf.stack([gy1, gx1], axis=-1)

      m00 = tf.cast(tf.expand_dims(tf.logical_and(my0, mx0), axis=-1), tf.float32)
      m01 = tf.cast(tf.expand_dims(tf.logical_and(my0, mx1), axis=-1), tf.float32)
      m10 = tf.cast(tf.expand_dims(tf.logical_and(my1, mx0), axis=-1), tf.float32)
      m11 = tf.cast(tf.expand_dims(tf.logical_and(my1, mx1), axis=-1), tf.float32)

      x00 = tf.gather_nd(image, tf.cast(g00, dtype=tf.int32), batch_dims=1)
      x01 = tf.gather_nd(image, tf.cast(g01, dtype=tf.int32), batch_dims=1)
      x10 = tf.gather_nd(image, tf.cast(g10, dtype=tf.int32), batch_dims=1)
      x11 = tf.gather_nd(image, tf.cast(g11, dtype=tf.int32), batch_dims=1)

      output = c00 * x00 * m00 \
             + c01 * x01 * m01 \
             + c10 * x10 * m10 \
             + c11 * x11 * m11

      return output
```