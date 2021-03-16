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
# 图像超分
  - [Github idealo/image-super-resolution](https://github.com/idealo/image-super-resolution)
    ```py
    !cd image-super-resolution && pip install -q . && cd -

    import os
    import matplotlib.pyplot as plt
    from skimage.io import imread, imsave
    from ISR.models import RDN, RRDN

    # model = RRDN(weights='gans')        # 17462488
    # model = RDN(weights='noise-cancel') # 66071288
    # model = RDN(weights='psnr-small')   # 10694096
    model = RDN(weights='psnr-large')     # 66071288

    image_name = "307622911.jpg"
    imm = imread(image_name)
    sr_img = model.predict(imm, by_patch_of_size=100)
    print("Source:", imm.shape, "Target:", sr_img.shape)

    plt.imshow(sr_img)
    plt.axis("off")
    plt.tight_layout()

    output = os.path.splitext(os.path.basename(image_name))[0] + "_x2_psnr-large.png"
    print("Saving to:", output)
    imsave(output, sr_img)
    ```
  - [Github yinboc/liif](https://github.com/yinboc/liif)
    ```py
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.io import imread, imsave
    from tqdm import tqdm

    # import sys
    # sys.path.append("/content/liif")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import models

    class ImageSuperResolution:
        def __init__(self, model_path):
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
            self.device = torch.device(device_name)
            self.model = models.make(torch.load(model_path, map_location=self.device)['model'], load_sd=True).to(self.device)

        def make_coord(self, shape, ranges=None, flatten=True):
            """ Make coordinates at grid centers.0000
            """
            coord_seqs = []
            for i, n in enumerate(shape):
                if ranges is None:
                    v0, v1 = -1, 1
                else:
                    v0, v1 = ranges[i]
                r = (v1 - v0) / (2 * n)
                seq = v0 + r + (2 * r) * torch.arange(n).float()
                coord_seqs.append(seq)
            ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
            if flatten:
                ret = ret.view(-1, ret.shape[-1])
            return ret

        def batched_predict(self, inp, coord, cell, bsize):
            with torch.no_grad():
                self.model.gen_feat(inp)
                total = coord.shape[1]
                steps = int(np.ceil(total / bsize))
                preds = [self.model.query_rgb(coord[:, ii: ii + bsize, :], cell[:, ii: ii + bsize, :]) for ii in tqdm(range(0, total, bsize))]
                preds = torch.cat(preds, dim=1)
            return preds

        def image_super_resolution(self, image, resolution, bsize=30000):
            if isinstance(image, str):
                image = imread(image) / 255
            inputs = torch.from_numpy(image.transpose(2, 0, 1).astype("float32"))
            inputs = ((inputs - 0.5) / 0.5).to(self.device)

            if isinstance(resolution, str) and ',' in resolution:
                hh, ww = [int(ii.strip()) for ii in resolution.split(',')]
            else:
                img_h, img_w = inputs.shape[1:]
                hh, ww = img_h * int(resolution), img_w * int(resolution)
            coord = self.make_coord((hh, ww)).to(self.device)
            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / hh
            cell[:, 1] *= 2 / ww
            pred = self.batched_predict(inputs.unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=bsize)[0]
            pred = (pred * 0.5 + 0.5).clamp(0, 1).view(hh, ww, 3)

            return pred.cpu().numpy()

        def to_4k_resolution_2160_3840(self, src_hh, src_ww, target_hh=2160, target_ww=3840):
            hh_enlarge, ww_enlarge = target_hh / src_hh, target_ww / src_ww
            if hh_enlarge > ww_enlarge:
                target_ww = src_ww * hh_enlarge
            else:
                target_hh = src_hh * ww_enlarge
            return int(target_hh), int(target_ww)

        def to_4k_image(self, image_name, output=None, bsize=30000):
            image = imread(image_name) / 255
            src_hh, src_ww = image.shape[:2]
            hh, ww = self.to_4k_resolution_2160_3840(src_hh, src_ww)
            print(">>>> Source shape: [%d, %d], Target shape: [%d, %d]" % (src_hh, src_ww, hh, ww))
            if output is None:
                output = os.path.splitext(os.path.basename(image_name))[0] + "_4k.png"
            idd = self.image_super_resolution(image, resolution="{},{}".format(hh, ww), bsize=bsize)

            print(">>>> Saving to %s ..." % (output))
            imsave(output, (idd * 255).astype("uint8"))
            return idd
    ```
    ```py
    aa = ImageSuperResolution("pre_trained/edsr-baseline-liif.pth")
    idd = aa.image_super_resolution('demo/0829x4-crop.png', 4, 300)

    aa = ImageSuperResolution("pre_trained/rdn-liif.pth")
    idd = aa.to_4k_image("602.jpg")
    ```
  - [Github krasserm/super-resolution](https://github.com/krasserm/super-resolution)
    ```py
    from model import resolve_single
    from utils import load_image, plot_sample
    from model.srgan import generator
    model = generator()
    model.load_weights('weights/srgan/gan_generator.h5')

    image_name = "../liif/307622911/307622911.jpg"
    lr = load_image(image_name)
    sr = resolve_single(model, lr)
    output = os.path.splitext(os.path.basename(image_name))[0] + "_x4_srgan.png"
    imsave(output, sr.numpy())

    plot_sample(lr, sr)

    ''' edsr '''
    from model.edsr import edsr
    model = edsr(scale=4, num_res_blocks=16)
    model.load_weights('weights/edsr-16-x4/weights.h5')
    sr = resolve_single(model, lr)
    output = os.path.splitext(os.path.basename(image_name))[0] + "_x4_edsr.png"
    imsave(output, sr.numpy())

    ''' wdsr '''
    from model.wdsr import wdsr_b
    model = wdsr_b(scale=4, num_res_blocks=32)
    model.load_weights('weights/wdsr-b-32-x4/weights.h5')
    sr = resolve_single(model, lr)
    output = os.path.splitext(os.path.basename(image_name))[0] + "_x4_wdsr.png"
    imsave(output, sr.numpy())
    ```
  - **Comparing**
    ```py
    def plot_results_sub(image_folder, sub_left, sub_right, sub_top, sub_bottom, plot_cols):
        iaa = sort(glob(os.path.join(image_folder, '*')))
        imms = [imread(ii) for ii in iaa]
        base_height = imms[0].shape[0]
        ees = [ii.shape[0] // base_height for ii in imms]
        subs = [ii[sub_top * ee:sub_bottom * ee, sub_left * ee:sub_right * ee, :] for ii, ee in zip(imms, ees)]

        cols = plot_cols
        rows = int(np.ceil(len(iaa) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.flatten()
        name_skip_len = len(os.path.splitext(os.path.basename(iaa[0]))[0])
        for ax, name, sub in zip(axes, iaa, subs):
            ax.imshow(sub)
            title = os.path.splitext(os.path.basename(name))[0][name_skip_len:]
            title = "Orignal" if len(title) == 0 else title
            ax.set_title(title)
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    fig = plot_results_sub("Blueprint_Wallpaper", 400, 600, 600, 900, 3)
    fig = plot_results_sub("307622911", 150, 300, 100, 250, 4)
    ```
```py
from to_4k_image import ImageSuperResolution

isr = ImageSuperResolution("pre_trained/edsr-baseline-liif.pth")

image_name = '307622911/307622911.jpg'
imm = imread(image_name)
fig = plt.figure()
plt.imshow(imm)
plt.tight_layout()

imms = [imm / 255]
resulution = 4
while True:
    print("Waiting for input...")
    aa = plt.ginput(2)
    aa = np.array(aa)

    left, right = int(aa[:, 0].min()), int(aa[:, 0].max())
    top, bottom = int(aa[:, 1].min()), int(aa[:, 1].max())

    if right - left < 1 or bottom - top < 1:
        if len(imms) > 1:
            imms.pop(-1)
        else:
            break
            # pass
    else:
        sub_image = imms[-1][top:bottom, left:right, :]
        idd = isr.image_super_resolution(sub_image, resulution)
        imms.append(idd)
    plt.imshow(imms[-1])
    plt.draw()
    print("Sub image displayed")
```
