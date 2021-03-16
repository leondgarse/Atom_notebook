- [Github openvino](https://github.com/openvinotoolkit/openvino.git)
- [Github Go imaging](https://github.com/disintegration/imaging)
- [Github ResNeSt](https://github.com/zhanghang1989/ResNeSt)
- [Github bleakie/MaskInsightface](https://github.com/bleakie/MaskInsightface)
- [Group Convolution分组卷积，以及Depthwise Convolution和Global Depthwise Convolution](https://cloud.tencent.com/developer/article/1394912)
- [深度学习中的卷积方式](https://zhuanlan.zhihu.com/p/75972500)

# Tests
## Image rotate
  ```py
  @tf.function
  def image_rotation(imm, degree):
      if degree == 90:
          return np.transpose(imm[::-1, :, :], (1, 0, 2))
      if degree == 180:
          return imm[::-1, ::-1, :]
      if degree == 270:
          return np.transpose(imm[:, ::-1, :], (1, 0, 2))
      return imm

  plt.imshow(image_rotation(imm, 90))
  plt.imshow(image_rotation(imm, 180))
  plt.imshow(image_rotation(imm, 270))
  ```
  ```py
  mm3 = keras.Sequential([
      keras.layers.Input((None, None, 3)),
      keras.preprocessing.image.apply_affine_transform,
      mm,
      keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})
  ])

  converter = tf.lite.TFLiteConverter.from_keras_model(mm3)
  tflite_model = converter.convert()
  open('./norm_model_tf2.tflite', 'wb').write(tflite_model)

  inputs = keras.layers.Input([None, None, 3])
  nn = keras.preprocessing.image.apply_affine_transform(inputs)
  mm = keras.models.Model(inputs, nn)
  ```
  ```py
  @tf.function
  def images_funcs(image, trans, type):
      ret = image
      type = type[0]
      if type == 0:
          # Resize
          ret = keras.layers.experimental.preprocessing.Resizing(trans[0], trans[1])
      elif type == 1:
          # Rotate
          angle = trans[0]
          if angle == 90:
              ret = image[::-1, :, :].transpose(1, 0, 2)
          elif angle == 180:
              ret = imm[::-1, ::-1, :]
          elif angle == 270:
              ret = imm[:, ::-1, :].transpose(1, 0, 2)
      elif type == 2:
          # Affine
          ret = keras.preprocessing.image.apply_affine_transform(image, *trans)
      return ret

  image_input = keras.layers.Input([None, None, 3])
  trans_input = keras.layers.Input([6])
  type_input = keras.layers.Input(None)
  ```
  ```py
  def transform_matrix_offset_center(matrix, x, y):
      o_x = float(x) / 2 + 0.5
      o_y = float(y) / 2 + 0.5
      offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
      reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
      transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
      return transform_matrix


  theta = tform.rotation / np.pi * 180
  _tx, _ty = tform.translation
  tx = np.cos(theta) * _tx + np.sin(theta) * _ty
  ty = np.cos(theta) * _ty - np.sin(theta) * _tx
  # tx, ty = _tx, _ty
  zx, zy = tform.scale, tform.scale
  transform_matrix = None

  if theta != 0:
      theta = np.deg2rad(theta)
      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                  [np.sin(theta), np.cos(theta), 0],
                                  [0, 0, 1]])
      transform_matrix = rotation_matrix

  if tx != 0 or ty != 0:
      # np.cos(theta), -np.sin(theta), np.cos(theta) * tx - np.sin(theta) * ty
      # np.sin(theta), np.cos(theta), np.sin(theta) * tx + np.cos(theta) * ty
      # 0, 0, 1
      shift_matrix = np.array([[1, 0, tx],
                               [0, 1, ty],
                               [0, 0, 1]])
      if transform_matrix is None:
          transform_matrix = shift_matrix
      else:
          transform_matrix = np.dot(transform_matrix, shift_matrix)

  if shear != 0:
      shear = np.deg2rad(shear)
      shear_matrix = np.array([[1, -np.sin(shear), 0],
                               [0, np.cos(shear), 0],
                               [0, 0, 1]])
      if transform_matrix is None:
          transform_matrix = shear_matrix
      else:
          transform_matrix = np.dot(transform_matrix, shear_matrix)

  if zx != 1 or zy != 1:
      # np.cos(theta) * zx, -np.sin(theta) * zy, np.cos(theta) * tx - np.sin(theta) * ty
      # np.sin(theta) * zx, np.cos(theta) * zy, np.sin(theta) * tx + np.cos(theta) * ty
      # 0, 0, 1
      zoom_matrix = np.array([[zx, 0, 0],
                              [0, zy, 0],
                              [0, 0, 1]])
      if transform_matrix is None:
          transform_matrix = zoom_matrix
      else:
          transform_matrix = np.dot(transform_matrix, zoom_matrix)

  if transform_matrix is not None:
      h, w = x.shape[row_axis], x.shape[col_axis]
      transform_matrix = transform_matrix_offset_center(
          transform_matrix, h, w)
      x = np.rollaxis(x, channel_axis, 0)
      final_affine_matrix = transform_matrix[:2, :2]
      final_offset = transform_matrix[:2, 2]

      channel_images = [ndimage.interpolation.affine_transform(
          x_channel,
          final_affine_matrix,
          final_offset,
          order=order,
          mode=fill_mode,
          cval=cval) for x_channel in x]
      x = np.stack(channel_images, axis=0)
      x = np.rollaxis(x, 0, channel_axis + 1)
  return x
  ```
## 几何变换 geometric transformations
## Face align landmarks
  ```py
  from skimage.transform import SimilarityTransform
  import cv2

  def face_align_landmarks(img, landmarks, image_size=(112, 112)):
      ret = []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
          src = np.array(
              [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
              dtype=np.float32,
          )

          dst = landmark.astype(np.float32)
          tform = SimilarityTransform()
          tform.estimate(dst, src)
          M = tform.params[0:2, :]
          ret.append(cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0))

      return np.array(ret)
  ```
  ```py
  from skimage import transform
  def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method='similar', show=True):
      tform = transform.AffineTransform() if method == 'affine' else transform.SimilarityTransform()
      src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
      ret, nns = [], []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
          tform.estimate(landmark, src)
          ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
      ret = (np.array(ret) * 255).astype(np.uint8)

      return (np.array(ret) * 255).astype(np.uint8)

  def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method='similar', order=1, show=True):
      tform = transform.AffineTransform() if method == 'affine' else transform.SimilarityTransform()
      src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
      ret, nns = [], []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
          tform.estimate(src, landmark)
          ret.append(transform.warp(img, tform, output_shape=image_size, order=order))
          if show:
              nns.append(tform.inverse(landmark))

      ret = (np.array(ret) * 255).astype(np.uint8)
      if show:
          plt.figure()
          plt.imshow(np.hstack(ret))
          for id, ii in enumerate(nns):
              plt.scatter(ii[:, 0] + image_size[0] * id, ii[:, 1], c='r', s=8)
      return ret
  ```
  ```py
  mm3 = keras.Sequential([
      keras.layers.Input((None, None, 3)),
      keras.preprocessing.image.apply_affine_transform,
      mm,
      keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})
  ])

  converter = tf.lite.TFLiteConverter.from_keras_model(mm3)
  tflite_model = converter.convert()
  open('./norm_model_tf2.tflite', 'wb').write(tflite_model)

  inputs = keras.layers.Input([None, None, 3])
  nn = keras.preprocessing.image.apply_affine_transform(inputs)
  mm = keras.models.Model(inputs, nn)
  ```
## ndimage affine_transform
  - 对于 `skimage.transform` 生成的转换矩阵，`ndimage.interpolation.affine_transform` 在使用时，需要将横纵坐标上的变幻对调
  - **变换 tform.parameters**
    - `rotation` 改为反向旋转
    - `translation` 对调 `xy` 变换值
    ```py
    from scipy import ndimage

    tform = transform.SimilarityTransform()
    tform.estimate(src, pps[0])

    # tt = transform.SimilarityTransform(rotation=tform.rotation*-1, scale=tform.scale, translation=tform.translation[::-1]).params
    tt = tform.params.copy()
    tt[0, -1], tt[1, -1] = tt[1, -1], tt[0, -1]
    tt[0, 1], tt[1, 0] = -1 * tt[0, 1], -1 * tt[1, 0]
    channel_images = [ndimage.interpolation.affine_transform(
        imm[:, :, ii],
        tt,
        output_shape=(112, 112),
        order=1,
        mode='nearest',
        cval=0) for ii in range(3)]
    x = np.stack(channel_images, axis=-1)
    plt.imshow(x)
    ```
  - **变换时图像转置** 使用原值的 `tform.parameters`，转置图像的宽高
    ```py
    from scipy import ndimage

    tform = transform.SimilarityTransform()
    tform.estimate(src, pps[0])

    channel_images = [ndimage.interpolation.affine_transform(
        imm[:, :, ii].T,
        tform.params,
        output_shape=(112, 112),
        order=1,
        mode='nearest',
        cval=0) for ii in range(3)]
    x = np.stack(channel_images, axis=-1)
    x = np.transpose(x, (1, 0, 2))
    plt.imshow(x)
    ```
  - **生成转置的变换矩阵** `skimage.transform` `estimate` 的参数值对调 `xy` 坐标值
    ```py
    tform.estimate(src[:, ::-1], pps[0][:, ::-1])
    channel_axis = 2
    x = np.rollaxis(imm, channel_axis, 0)  # (976, 1920, 3) --> (3, 976, 1920)
    channel_images = [ndimage.interpolation.affine_transform(
        x_channel,
        tform.params,
        output_shape=(112, 112),
        order=1,
        mode='nearest',
        cval=0) for x_channel in x]
    x = np.stack(channel_images, axis=0)  # (3, 112, 112)
    x = np.rollaxis(x, 0, channel_axis + 1) # (112, 112, 3)
    plt.imshow(x)
    ```
## affine_transform 图像缩放
  ```py
  scale = 3
  tt = transform.SimilarityTransform(scale=scale, translation=[0, 0]).params
  channel_images = [ndimage.interpolation.affine_transform(
      imm[:, :, ii],
      tt,
      output_shape=(imm.shape[0] // scale, imm.shape[1] // scale),
      order=1,
      mode='nearest',
      cval=0) for ii in range(3)]
  x = np.stack(channel_images, axis=-1)
  plt.imshow(x)
  ```
## affine_transform 图像旋转
  ```py
  theta = 90 # [90, 180, 270]
  rotation = theta / 180 * np.pi
  # translation=[imm.shape[0] * abs(cos(rotation)), imm.shape[1] * abs(sin(rotation))]
  if theta == 90:
      translation=[imm.shape[0], 0]
      output_shape = imm.shape[:2][::-1]
  elif theta == 180:
      translation=imm.shape[:2]
      output_shape = imm.shape[:2]
  elif theta == 270:
      translation=[0, imm.shape[1]]
      output_shape = imm.shape[:2][::-1]

  tt = transform.SimilarityTransform(rotation=rotation, translation=translation).params
  channel_images = [ndimage.interpolation.affine_transform(
      imm[:, :, ii],
      tt,
      output_shape=output_shape,
      order=1,
      mode='nearest',
      cval=0) for ii in range(3)]
  x = np.stack(channel_images, axis=-1)
  plt.imshow(x)
  ```
## TF function
  ```py
  class WarpAffine(keras.layers.Layer):
      def __call__(self, imm, tformP, output_shape):
          rets = []
          for xx in imm:
              x = tf.transpose(xx, (2, 0, 1))
              channel_images = [ndimage.interpolation.affine_transform(
                  x_channel,
                  tformP,
                  output_shape=output_shape,
                  order=1,
                  mode='nearest',
                  cval=0) for x_channel in x]
              x = tf.stack(channel_images, axis=0)
              x = tf.transpose(x, (1, 2, 0))
              rets.append(x)
          return rets
  ```
## Rotation
  ```go
  func (nnInterpolator) transform_RGBA_RGBA_Src(dst *image.RGBA, dr, adr image.Rectangle, d2s *f64.Aff3, src *image.RGBA, sr image.Rectangle, bias image.Point) {
      for dy := int32(adr.Min.Y); dy < int32(adr.Max.Y); dy++ {
          dyf := float64(dr.Min.Y+int(dy)) + 0.5
          d := (dr.Min.Y+int(dy)-dst.Rect.Min.Y)*dst.Stride + (dr.Min.X+adr.Min.X-dst.Rect.Min.X)*4
          for dx := int32(adr.Min.X); dx < int32(adr.Max.X); dx, d = dx+1, d+4 {
              dxf := float64(dr.Min.X+int(dx)) + 0.5
              sx0 := int(d2s[0]*dxf+d2s[1]*dyf+d2s[2]) + bias.X
              sy0 := int(d2s[3]*dxf+d2s[4]*dyf+d2s[5]) + bias.Y
              if !(image.Point{sx0, sy0}).In(sr) {
                  continue
              }
              pi := (sy0-src.Rect.Min.Y)*src.Stride + (sx0-src.Rect.Min.X)*4
              pr := uint32(src.Pix[pi+0]) * 0x101
              pg := uint32(src.Pix[pi+1]) * 0x101
              pb := uint32(src.Pix[pi+2]) * 0x101
              pa := uint32(src.Pix[pi+3]) * 0x101
              dst.Pix[d+0] = uint8(pr >> 8)
              dst.Pix[d+1] = uint8(pg >> 8)
              dst.Pix[d+2] = uint8(pb >> 8)
              dst.Pix[d+3] = uint8(pa >> 8)
          }
      }    
  }
  ```
  ```cpp
  void RotateDrawWithClip(
      WDIBPIXEL *pDstBase, int dstW, int dstH, int dstDelta,
      WDIBPIXEL *pSrcBase, int srcW, int srcH, int srcDelta,
      float fDstCX, float fDstCY, float fSrcCX, float fSrcCY, float fAngle, float fScale) {
      if (dstW <= 0) { return; }
      if (dstH <= 0) { return; }

      srcDelta /= sizeof(WDIBPIXEL);
      dstDelta /= sizeof(WDIBPIXEL);

      float duCol = (float)sin(-fAngle) * (1.0f / fScale);
      float dvCol = (float)cos(-fAngle) * (1.0f / fScale);
      float duRow = dvCol;
      float dvRow = -duCol;

      float startingu = fSrcCX - (fDstCX * dvCol + fDstCY * duCol);
      float startingv = fSrcCY - (fDstCX * dvRow + fDstCY * duRow);

      float rowu = startingu;
      float rowv = startingv;

      for(int y = 0; y < dstH; y++) {
          float uu = rowu;
          float vv = rowv;

          WDIBPIXEL *pDst = pDstBase + (dstDelta * y);

          for(int x = 0; x < dstW ; x++) {
              int sx = (int)uu;
              int sy = (int)vv;

              // For non-negative values we have to check u and v (not sx and sy)
              // since u = -0.25 gives sx=0 after rounsing, so 1 extra pixel line will be drawn
              // (we assume that u,v >= 0 will lead to sx,sy >= 0)

              if ((uu >= 0) && (vv >= 0) && (sx < srcW) && (sy < srcH)) {
                  WDIBPIXEL *pSrc = pSrcBase + sx + (sy * srcDelta);
                  *pDst++ = *pSrc++;
              } else {
                  pDst++; // Skip
                  //*pDst++ = VOID_COLOR; // Fill void (black)
              }

              uu += duRow;
              vv += dvRow;
          }

          rowu += duCol;
          rowv += dvCol;
      }
  }
  ```
  ```py
  def image_rotate(src, dstW, dstH, tf):
      # convW = cos(rotate) / scale
      dst = np.zeros([dstW, dstH, 3], dtype=src.dtype)
      ww, hh = src.shape[:2]
      ww, hh = ww - 1, hh - 1
      for ii in range(dstW):
          tw, th = tf[0] * ii, tf[3] * ii
          for jj in range(dstH):
              sw = int(tw + tf[1] * jj + tf[2])
              sh = int(th + tf[4] * jj + tf[5])
              if sw > 0 and sw < ww and sh > 0 and sh < hh:
                  dst[ii, jj] = src[sw, sh]
              else:
                  dst[ii, jj] = [0, 0, 0]
      return dst

  angle = 60 / 180 * np.pi
  tf = [cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0]
  ```
## NV21 to RGB
  ```py
  import cv2
  def YUVtoRGB(byteArray, width, height):
      e = width * height
      Y = byteArray[0:e]
      Y = np.reshape(Y, (height, width))

      s = e
      V = byteArray[s::2]
      V = np.repeat(V, 2, 0)
      V = np.reshape(V, (height // 2, width))
      V = np.repeat(V, 2, 0)

      U = byteArray[s+1::2]
      U = np.repeat(U, 2, 0)
      U = np.reshape(U, (height // 2, width))
      U = np.repeat(U, 2, 0)

      RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
      RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
      return RGBMatrix

  with open('nv21.txt', 'r') as ff:
      aa = ff.read()
  bb = [byte(ii) for ii in aa[1:-1].split(', ')]
  with open('nv21.bin', 'wb') as ff:
      for ii in bb:
          ff.write(ii)

  with open('nv21.bin', 'rb') as ff:
      cc = ff.read()
  plt.imshow(YUVtoRGB([byte(ii) for ii in cc], 1280, 800))
  ```
## 曲线拟合
  ```py
  import json
  with open("./checkpoints/keras_resnet101_emore_hist.json", 'r') as ff:
      jj = json.load(ff)
  ss = jj['loss'][29:-5]
  ['%.4f' % ii for ii in jj['loss'][-10:]]
  # ['8.6066', '8.2645', '7.9587', '7.6866', '7.4418', '7.2208']

  zz = np.polyfit(np.arange(1, len(ss)), ss[1:], 3)
  yy = np.poly1d(zz)
  ["%.4f" % ii for ii in yy(np.arange(len(ss) - 5, len(ss) + 10))]
  # ['8.6065', '8.2710', '7.9557', '7.6401', '7.3035', '6.9252', '6.4847', '5.9613']

  ee = 0.105
  pp = ss[:len(ss) - 3].copy()
  for ii in range(len(ss) - 5, len(ss) + 10):
      pp.append(pp[ii - 1] - (pp[ii - 2] - pp[ii - 1]) * (1 - ee))
      print("%.4f" % pp[-1], end=', ')
  # 8.5960, 8.2454, 7.9316, 7.6508, 7.3994, 7.1744, 6.9731, 6.7929
  # ==> (f(x-1) - f(x)) / (f(x-2) - f(x-1)) = (1 - ee)
  #     && f(x) = aa * np.exp(-bb * x) + cc
  # ==> (np.exp(bb) - 1) / (np.exp(2 * bb) - np.exp(bb)) = (1 - ee)
  # ==> (1 - ee) * np.exp(2 * bb) - (2 - ee) * np.exp(bb) + 1 = 0

  from sympy import solve, symbols, Eq
  bb = symbols('bb')
  brr = solve(Eq(np.e ** (2 * bb) * (1 - ee) - (2 - ee) * np.e ** bb + 1, 0), bb)
  print(brr) # [0.0, 0.110931560707281]
  ff = lambda xx: np.e ** (-xx * brr[1])
  ['%.4f' % ((ff(ii - 1) - ff(ii)) / (ff(ii - 2) - ff(ii - 1))) for ii in range(10, 15)]
  # ['0.8950', '0.8950', '0.8950', '0.8950', '0.8950']

  aa, cc = symbols('aa'), symbols('cc')
  rr = solve([Eq(aa * ff(len(ss) - 3) + cc, ss[-3]), Eq(aa * ff(len(ss) - 1) + cc, ss[-1])], [aa, cc])
  func_solve = lambda xx: rr[aa] * ff(xx) + rr[cc]
  ["%.4f" % ii for ii in func_solve(np.arange(len(ss) - 5, len(ss) + 10))]
  # ['8.6061', '8.2645', '7.9587', '7.6850', '7.4401', '7.2209', '7.0247', '6.8491']

  from scipy.optimize import curve_fit

  def func_curv(x, a, b, c):
      return a * np.exp(-b * x) + c
  xx = np.arange(1, 1 + len(ss[1:]))
  popt, pcov = curve_fit(func_curv, xx, ss[1:])
  print(popt) # [6.13053796 0.1813183  6.47103657]
  ["%.4f" % ii for ii in func_curv(np.arange(len(ss) - 5, len(ss) + 10), *popt)]
  # ['8.5936', '8.2590', '7.9701', '7.7208', '7.5057', '7.3200', '7.1598', '7.0215']

  plt.plot(np.arange(len(ss) - 3, len(ss)), ss[-3:], label="Original Curve")
  xx = np.arange(len(ss) - 3, len(ss) + 3)
  plt.plot(xx, pp[-len(xx):], label="Manuel fit")
  plt.plot(xx, func_solve(xx), label="func_solve fit")
  plt.plot(xx, func_curv(xx, *popt), label="func_curv fit")
  plt.legend()
  ```
## Plot styles
  ```py
  big, baxes = plt.subplots(5, 5)
  baxes = baxes.flatten()
  styles = plt.style.available
  if 'dark_background' in styles: styles.remove('dark_background')
  for bax, style in zip(baxes, styles):
      fn = style + '.png'
      if not os.path.exists(fn):
          plt.style.use(style)
          fig, axes = plt.subplots(2, 2)
          axes[0][0].plot(np.random.randint(1, 10, 10), label='aa')
          axes[0][0].plot(np.random.randint(1, 10, 10), label='bb')
          axes[0][0].legend()
          axes[0][1].scatter(np.random.randint(1, 10, 10), np.random.randint(1, 10, 10))
          axes[1][0].hist(np.random.randint(1, 10, 10))
          rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
          axes[1][1].add_patch(rect)
          fig.suptitle(style)
          fig.savefig(fn)
          plt.close()
      bax.imshow(plt.imread(fn))
      bax.axis('off')
  big.tight_layout()
  ```
## Plot color palettes
  ```py
  import matplotlib.cm as cm
  from cycler import cycler
  import seaborn as sns

  def get_colors(max_color, palette='husl'):
      if palette == 'rainbow':
          colors = cm.rainbow(np.linspace(0, 1, max_color))
      else:
          colors = sns.color_palette(palette, n_colors=max_color)
      return colors

  ccs = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind', 'rainbow', 'husl', 'hls']
  max_color = 10
  fig, axes = plt.subplots(3, 3, figsize=(15, 12))
  axes = axes.flatten()
  for cc, ax in zip (ccs, axes):
      colors = get_colors(max_color, cc)
      ax.set_prop_cycle(cycler('color', colors))
      for ii in range(max_color):
          ax.plot(np.random.randint(1, 10, 10), label=ii)
      ax.legend(loc="upper right")
      ax.set_title(cc)
  fig.tight_layout()
  ```
***

# skimage segmentation
## Felzenszwalb Quickshift SLIC watershed
  This example compares four popular low-level image segmentation methods. As it is difficult to obtain good segmentations, and the definition of “good” often depends on the application, these methods are usually used for obtaining an oversegmentation, also known as superpixels. These superpixels then serve as a basis for more sophisticated algorithms such as conditional random fields (CRF).

  Felzenszwalb’s efficient graph based segmentation
  This fast 2D image segmentation algorithm, proposed in 1 is popular in the computer vision community. The algorithm has a single scale parameter that influences the segment size. The actual size and number of segments can vary greatly, depending on local contrast.

  1
  Efficient graph-based image segmentation, Felzenszwalb, P.F. and Huttenlocher, D.P. International Journal of Computer Vision, 2004

  Quickshift image segmentation
  Quickshift is a relatively recent 2D image segmentation algorithm, based on an approximation of kernelized mean-shift. Therefore it belongs to the family of local mode-seeking algorithms and is applied to the 5D space consisting of color information and image location 2.

  One of the benefits of quickshift is that it actually computes a hierarchical segmentation on multiple scales simultaneously.

  Quickshift has two main parameters: sigma controls the scale of the local density approximation, max_dist selects a level in the hierarchical segmentation that is produced. There is also a trade-off between distance in color-space and distance in image-space, given by ratio.

  2
  Quick shift and kernel methods for mode seeking, Vedaldi, A. and Soatto, S. European Conference on Computer Vision, 2008

  SLIC - K-Means based image segmentation
  This algorithm simply performs K-means in the 5d space of color information and image location and is therefore closely related to quickshift. As the clustering method is simpler, it is very efficient. It is essential for this algorithm to work in Lab color space to obtain good results. The algorithm quickly gained momentum and is now widely used. See 3 for details. The compactness parameter trades off color-similarity and proximity, as in the case of Quickshift, while n_segments chooses the number of centers for kmeans.

  3
  Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Suesstrunk, SLIC Superpixels Compared to State-of-the-art Superpixel Methods, TPAMI, May 2012.

  Compact watershed segmentation of gradient images
  Instead of taking a color image as input, watershed requires a grayscale gradient image, where bright pixels denote a boundary between regions. The algorithm views the image as a landscape, with bright pixels forming high peaks. This landscape is then flooded from the given markers, until separate flood basins meet at the peaks. Each distinct basin then forms a different image segment. 4

  As with SLIC, there is an additional compactness argument that makes it harder for markers to flood faraway pixels. This makes the watershed regions more regularly shaped. 5
  ```py
  from skimage.color import rgb2gray
  from skimage.filters import sobel
  from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
  from skimage.segmentation import mark_boundaries
  from skimage.util import img_as_float
  from skimage.io import imread

  img = img_as_float(imread('./000067.dcm.png'))[:, :, :3]
  segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
  segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
  segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
  gradient = sobel(rgb2gray(img))
  segments_watershed = watershed(gradient, markers=250, compactness=0.001)

  print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
  print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
  print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")

  fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

  ax[0, 0].imshow(mark_boundaries(img, segments_fz))
  ax[0, 0].set_title("Felzenszwalbs's method")
  ax[0, 1].imshow(mark_boundaries(img, segments_slic))
  ax[0, 1].set_title('SLIC')
  ax[1, 0].imshow(mark_boundaries(img, segments_quick))
  ax[1, 0].set_title('Quickshift')
  ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
  ax[1, 1].set_title('Compact watershed')

  for a in ax.ravel():
      a.set_axis_off()

  plt.tight_layout()
  plt.show()
  ```
  ![](images/skimage_seg_fsqw.png)
## Join segmentations
  When segmenting an image, you may want to combine multiple alternative segmentations. The skimage.segmentation.join_segmentations() function computes the join of two segmentations, in which a pixel is placed in the same segment if and only if it is in the same segment in both segmentations.
  ```py
  import numpy as np
  import matplotlib.pyplot as plt

  from skimage.filters import sobel
  from skimage.measure import label
  from skimage.segmentation import slic, join_segmentations
  from skimage.morphology import watershed
  from skimage.color import label2rgb, rgb2gray
  from skimage.io import imread

  img = (rgb2gray(imread('./000067.dcm.png')) * 255).astype(np.uint8)

  # Make segmentation using edge-detection and watershed.
  edges = sobel(img)

  # Identify some background and foreground pixels from the intensity values.
  # These pixels are used as seeds for watershed.
  markers = np.zeros_like(img)
  foreground, background = 1, 2
  markers[img < 20] = background
  markers[img > 30] = foreground

  ws = watershed(edges, markers)
  seg1 = label(ws == foreground)

  # Make segmentation using SLIC superpixels.
  seg2 = slic(img, n_segments=117, max_iter=160, sigma=1, compactness=0.75,
              multichannel=False)

  # Combine the two.
  segj = join_segmentations(seg1, seg2)

  # Show the segmentations.
  fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 5),
                           sharex=True, sharey=True)
  ax = axes.ravel()
  ax[0].imshow(img, cmap='gray')
  ax[0].set_title('Image')

  color1 = label2rgb(seg1, image=img, bg_label=0)
  ax[1].imshow(color1)
  ax[1].set_title('Sobel+Watershed')

  color2 = label2rgb(seg2, image=img, image_alpha=0.5)
  ax[2].imshow(color2)
  ax[2].set_title('SLIC superpixels')

  color3 = label2rgb(segj, image=img, image_alpha=0.5)
  ax[3].imshow(color3)
  ax[3].set_title('Join')

  for a in ax:
      a.axis('off')
  fig.tight_layout()
  plt.show()
  ```
  ![](images/skimage_seg_join.png)
## Morphological Snakes
  - **活动轮廓分割 snakes** 使用用户定义的轮廓或线进行初始化，然后该轮廓慢慢收缩

  Morphological Snakes 1 are a family of methods for image segmentation. Their behavior is similar to that of active contours (for example, Geodesic Active Contours 2 or Active Contours without Edges 3). However, Morphological Snakes use morphological operators (such as dilation or erosion) over a binary array instead of solving PDEs over a floating point array, which is the standard approach for active contours. This makes Morphological Snakes faster and numerically more stable than their traditional counterpart.

  There are two Morphological Snakes methods available in this implementation: Morphological Geodesic Active Contours (MorphGAC, implemented in the function morphological_geodesic_active_contour) and Morphological Active Contours without Edges (MorphACWE, implemented in the function morphological_chan_vese).

  MorphGAC is suitable for images with visible contours, even when these contours might be noisy, cluttered, or partially unclear. It requires, however, that the image is preprocessed to highlight the contours. This can be done using the function inverse_gaussian_gradient, although the user might want to define their own version. The quality of the MorphGAC segmentation depends greatly on this preprocessing step.

  On the contrary, MorphACWE works well when the pixel values of the inside and the outside regions of the object to segment have different averages. Unlike MorphGAC, MorphACWE does not require that the contours of the object are well defined, and it works over the original image without any preceding processing. This makes MorphACWE easier to use and tune than MorphGAC.
  ```py
  import numpy as np
  import matplotlib.pyplot as plt
  from skimage import data, img_as_float
  from skimage.io import imread
  from skimage.segmentation import (morphological_chan_vese,
                                    morphological_geodesic_active_contour,
                                    inverse_gaussian_gradient,
                                    checkerboard_level_set)


  def store_evolution_in(lst):
      """Returns a callback function to store the evolution of the level sets in
      the given list.
      """

      def _store(x):
          lst.append(np.copy(x))

      return _store


  # Morphological ACWE\
  image = rgb2gray(imread('./000067.dcm.png'))

  # Initial level set
  init_ls = checkerboard_level_set(image.shape, 6)
  # List with intermediate results for plotting the evolution
  evolution = []
  callback = store_evolution_in(evolution)
  ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                               iter_callback=callback)

  fig, axes = plt.subplots(2, 2, figsize=(8, 8))
  ax = axes.flatten()

  ax[0].imshow(image, cmap="gray")
  ax[0].set_axis_off()
  ax[0].contour(ls, [0.5], colors='r')
  ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

  ax[1].imshow(ls, cmap="gray")
  ax[1].set_axis_off()
  contour = ax[1].contour(evolution[2], [0.5], colors='g')
  contour.collections[0].set_label("Iteration 2")
  contour = ax[1].contour(evolution[7], [0.5], colors='y')
  contour.collections[0].set_label("Iteration 7")
  contour = ax[1].contour(evolution[-1], [0.5], colors='r')
  contour.collections[0].set_label("Iteration 35")
  ax[1].legend(loc="upper right")
  title = "Morphological ACWE evolution"
  ax[1].set_title(title, fontsize=12)


  # Morphological GAC
  gimage = inverse_gaussian_gradient(image)

  # Initial level set
  init_ls = np.zeros(image.shape, dtype=np.int8)
  init_ls[10:-10, 10:-10] = 1
  # List with intermediate results for plotting the evolution
  evolution = []
  callback = store_evolution_in(evolution)
  ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                             smoothing=1, balloon=-1,
                                             threshold=0.69,
                                             iter_callback=callback)

  ax[2].imshow(image, cmap="gray")
  ax[2].set_axis_off()
  ax[2].contour(ls, [0.5], colors='r')
  ax[2].set_title("Morphological GAC segmentation", fontsize=12)

  ax[3].imshow(ls, cmap="gray")
  ax[3].set_axis_off()
  contour = ax[3].contour(evolution[0], [0.5], colors='g')
  contour.collections[0].set_label("Iteration 0")
  contour = ax[3].contour(evolution[100], [0.5], colors='y')
  contour.collections[0].set_label("Iteration 100")
  contour = ax[3].contour(evolution[-1], [0.5], colors='r')
  contour.collections[0].set_label("Iteration 230")
  ax[3].legend(loc="upper right")
  title = "Morphological GAC evolution"
  ax[3].set_title(title, fontsize=12)

  fig.tight_layout()
  plt.show()
  ```
  ![](images/skiamge_seg_morphological_snakes.png)
***


# Auto Tuner
## Keras Tuner
  - [Keras Tuner 简介](https://www.tensorflow.org/tutorials/keras/keras_tuner)
    ```py
    import tensorflow as tf
    from tensorflow import keras

    !pip install -q -U keras-tuner
    import kerastuner as kt

    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
    # Normalize pixel values between 0 and 1
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.0

    def model_builder(hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))
        model.add(keras.layers.Dense(10))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                      loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective = 'val_accuracy',
                         max_epochs = 10,
                         factor = 3,
                         directory = 'my_dir',
                         project_name = 'intro_to_kt')

    tuner.search(img_train, label_train, epochs = 10, validation_data = (img_test, label_test), callbacks = [ClearTrainingOutput()])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    model.fit(img_train, label_train, epochs = 10, validation_data = (img_test, label_test))
    ```
  - **Tune on cifar10**
    ```py
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import kerastuner as kt
    import tensorflow_addons as tfa

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels_oh = tf.one_hot(tf.squeeze(train_labels), depth=10, dtype='uint8')
    test_labels_oh = tf.one_hot(tf.squeeze(test_labels), depth=10, dtype='uint8')
    print(train_images.shape, test_images.shape, train_labels_oh.shape, test_labels_oh.shape)

    def create_model(hp):
        hp_wd = hp.Choice("weight_decay", values=[0.0, 1e-5, 5e-5, 1e-4])
        hp_ls = hp.Choice("label_smoothing", values=[0.0, 0.1])
        hp_dropout = hp.Choice("dropout_rate", values=[0.0, 0.4])

        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(rate=hp_dropout))
        model.add(keras.layers.Dense(10))

        model.compile(
            optimizer=tfa.optimizers.AdamW(weight_decay=hp_wd),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=hp_ls),
            metrics = ['accuracy'])

        return model

    tuner = kt.Hyperband(create_model,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=6,
                         directory='my_dir',
                         project_name='intro_to_kt')

    tuner.search(train_images, train_labels_oh, epochs=50, validation_data=(test_images, test_labels_oh))

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("best parameters: weight_decay = {}, label_smoothing = {}, dropout_rate = {}".format(best_hps.get('weight_decay'), best_hps.get('label_smoothing'), best_hps.get('dropout_rate')))

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_images, train_labels_oh, epochs = 50, validation_data = (test_images, test_labels_oh))
    ```
## TensorBoard HParams
  - [Hyperparameter Tuning with the HParams Dashboard](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams)
    ```py
    # Load the TensorBoard notebook extension
    %load_ext tensorboard

    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    def train_test_model(hparams):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ])
        model.compile(
            optimizer=hparams[HP_OPTIMIZER],
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # model.fit(
        #   ...,
        #   callbacks=[
        #       tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        #       hp.KerasCallback(logdir, hparams),  # log hparams
        #   ],
        # )
        model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
        _, accuracy = model.evaluate(x_test, y_test)
        return accuracy

    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1

    %tensorboard --logdir logs/hparam_tuning
    ```
  - **Tune on cifar10**
    ```py
    %load_ext tensorboard

    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
    import tensorflow_addons as tfa

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels_oh = tf.one_hot(tf.squeeze(train_labels), depth=10, dtype='uint8')
    test_labels_oh = tf.one_hot(tf.squeeze(test_labels), depth=10, dtype='uint8')
    print(train_images.shape, test_images.shape, train_labels_oh.shape, test_labels_oh.shape)

    HP_WD = hp.HParam("weight_decay", hp.Discrete([0.0, 1e-5, 5e-5, 1e-4]))
    HP_LS = hp.HParam("label_smoothing", hp.Discrete([0.0, 0.1]))
    HP_DR = hp.HParam("dropout_rate", hp.Discrete([0.0, 0.4]))
    METRIC_ACCURACY = 'accuracy'
    METRIC_LOSS = 'loss'

    with tf.summary.create_file_writer('logs/hparam_tuning_cifar10').as_default():
        hp.hparams_config(
            hparams=[HP_WD, HP_LS, HP_DR],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'), hp.Metric(METRIC_LOSS, display_name='Loss')],
        )

    def create_model(dropout=1):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        if dropout > 0 and dropout < 1:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(10))
        return model

    def train_test_model(hparams, epochs=1):
        model = create_model(hparams[HP_DR])
        model.compile(
            optimizer=tfa.optimizers.AdamW(weight_decay=hparams[HP_WD]),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=hparams[HP_LS], from_logits=True),
            metrics=['accuracy'],
        )

        # model.fit(
        #   ...,
        #   callbacks=[
        #       tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        #       hp.KerasCallback(logdir, hparams),  # log hparams
        #   ],
        # )
        hist = model.fit(train_images, train_labels_oh, epochs=epochs, validation_data=(test_images, test_labels_oh)) # Run with 1 epoch to speed things up for demo purposes
        return max(hist.history["val_accuracy"]), min(hist.history["val_loss"])

    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            val_accuracy, val_loss = train_test_model(hparams, epochs=20)
            tf.summary.scalar(METRIC_ACCURACY, val_accuracy, step=1)
            tf.summary.scalar(METRIC_LOSS, val_loss, step=1)

    session_num = 0
    for dr in HP_DR.domain.values:
        for label_smoothing in HP_LS.domain.values:
            for wd in HP_WD.domain.values:
                hparams = {
                    HP_WD: wd,
                    HP_LS: label_smoothing,
                    HP_DR: dr,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning_cifar10/' + run_name, hparams)
                session_num += 1

    %tensorboard --logdir logs/hparam_tuning_cifar10
    ```
***

# Replace UpSampling2D with Conv2DTranspose
## Conv2DTranspose output shape
  ```py
  for strides in range(1, 4):
      for kernel_size in range(1, 4):
          aa = keras.layers.Conv2DTranspose(3, kernel_size, padding='same', strides=strides)
          aa.build([1, 3, 3, 3])
          print("[SAME] kernel_size: {}, strides: {}, shape: {}".format(kernel_size, strides, aa(tf.ones([1, 3, 3, 3], dtype='float32')).shape.as_list()))
  # [SAME] kernel_size: 1, strides: 1, shape: [1, 3, 3, 3]
  # [SAME] kernel_size: 2, strides: 1, shape: [1, 3, 3, 3]
  # [SAME] kernel_size: 3, strides: 1, shape: [1, 3, 3, 3]
  # [SAME] kernel_size: 1, strides: 2, shape: [1, 6, 6, 3]
  # [SAME] kernel_size: 2, strides: 2, shape: [1, 6, 6, 3]
  # [SAME] kernel_size: 3, strides: 2, shape: [1, 6, 6, 3]
  # [SAME] kernel_size: 1, strides: 3, shape: [1, 9, 9, 3]
  # [SAME] kernel_size: 2, strides: 3, shape: [1, 9, 9, 3]
  # [SAME] kernel_size: 3, strides: 3, shape: [1, 9, 9, 3]

  for strides in range(1, 4):
      for kernel_size in range(1, 5):
          aa = keras.layers.Conv2DTranspose(3, kernel_size, padding='valid', strides=strides)
          aa.build([1, 3, 3, 3])
          print("[VALID] kernel_size: {}, strides: {}, shape: {}".format(kernel_size, strides, aa(tf.ones([1, 3, 3, 3], dtype='float32')).shape.as_list()))
  # [VALID] kernel_size: 1, strides: 1, shape: [1, 3, 3, 3]
  # [VALID] kernel_size: 2, strides: 1, shape: [1, 4, 4, 3]
  # [VALID] kernel_size: 3, strides: 1, shape: [1, 5, 5, 3]
  # [VALID] kernel_size: 4, strides: 1, shape: [1, 6, 6, 3]
  # [VALID] kernel_size: 1, strides: 2, shape: [1, 6, 6, 3]
  # [VALID] kernel_size: 2, strides: 2, shape: [1, 6, 6, 3]
  # [VALID] kernel_size: 3, strides: 2, shape: [1, 7, 7, 3]
  # [VALID] kernel_size: 4, strides: 2, shape: [1, 8, 8, 3]
  # [VALID] kernel_size: 1, strides: 3, shape: [1, 9, 9, 3]
  # [VALID] kernel_size: 2, strides: 3, shape: [1, 9, 9, 3]
  # [VALID] kernel_size: 3, strides: 3, shape: [1, 9, 9, 3]
  # [VALID] kernel_size: 4, strides: 3, shape: [1, 10, 10, 3]
  ```
## Nearest interpolation
  - **Image methods**
    ```py
    imsize = 3
    x, y = np.ogrid[:imsize, :imsize]
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(imsize + imsize)
    plt.imshow(img, interpolation='none')

    import tensorflow.keras.backend as K
    iaa = tf.image.resize(img, (6, 6), method='nearest')
    ibb = K.resize_images(tf.expand_dims(tf.cast(img, 'float32'), 0), 2, 2, K.image_data_format(), interpolation='nearest')
    ```
  - **UpSampling2D**
    ```py
    aa = keras.layers.UpSampling2D((2, 2), interpolation='nearest')
    icc = aa(tf.expand_dims(tf.cast(img, 'float32'), 0)).numpy()[0]

    print(np.allclose(iaa, icc))
    # True
    ```
  - **tf.nn.conv2d_transpose**
    ```py
    def nearest_upsample_weights(factor, number_of_classes=3):
        filter_size = 2 * factor - factor % 2
        weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes), dtype=np.float32)
        upsample_kernel = np.zeros([filter_size, filter_size])
        upsample_kernel[1:factor + 1, 1:factor + 1] = 1

        for i in range(number_of_classes):
            weights[:, :, i, i] = upsample_kernel
        return weights

    channel, factor = 3, 2
    idd = tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor, img.shape[1] * factor, channel], strides=factor, padding='SAME')
    print(np.allclose(iaa, idd))
    # True

    # Output shape can be different values
    channel, factor = 3, 3
    print(tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor, img.shape[1] * factor, channel], strides=factor, padding='SAME').shape)
    # (1, 9, 9, 3)
    print(tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor - 1, img.shape[1] * factor - 1, channel], strides=factor, padding='SAME').shape)
    # (1, 8, 8, 3)
    print(tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor - 2, img.shape[1] * factor - 2, channel], strides=factor, padding='SAME').shape)
    # (1, 7, 7, 3)
    ```
  - **Conv2DTranspose**
    ```py
    bb = keras.layers.Conv2DTranspose(channel, 2 * factor - factor % 2, padding='same', strides=factor, use_bias=False)
    bb.build([None, None, None, channel])
    bb.set_weights([nearest_upsample_weights(factor, channel)])
    iee = bb(tf.expand_dims(img.astype('float32'), 0)).numpy()[0]
    print(np.allclose(iaa, iee))
    # True
    ```
## Bilinear
  - [pytorch_bilinear_conv_transpose.py](https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183)
  - [Upsampling and Image Segmentation with Tensorflow and TF-Slim](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)
  - **UpSampling2D**
    ```py
    imsize = 3
    x, y = np.ogrid[:imsize, :imsize]
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(imsize + imsize)
    plt.imshow(img, interpolation='none')

    channel, factor = 3, 3
    iaa = tf.image.resize(img, (img.shape[0] * factor, img.shape[1] * factor), method='bilinear')

    aa = keras.layers.UpSampling2D((factor, factor), interpolation='bilinear')
    ibb = aa(tf.expand_dims(tf.cast(img, 'float32'), 0)).numpy()[0]
    print(np.allclose(iaa, ibb))
    # True
    ```
  - **Pytorch BilinearConvTranspose2d**
    ```py
    import torch
    import torch.nn as nn

    class BilinearConvTranspose2d(nn.ConvTranspose2d):
        def __init__(self, channels, stride, groups=1):
            if isinstance(stride, int):
                stride = (stride, stride)

            kernel_size = (2 * stride[0] - stride[0] % 2, 2 * stride[1] - stride[1] % 2)
            # padding = (stride[0] - 1, stride[1] - 1)
            padding = 1
            super().__init__(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

        def reset_parameters(self):
            nn.init.constant(self.bias, 0)
            nn.init.constant(self.weight, 0)
            bilinear_kernel = self.bilinear_kernel(self.stride)
            for i in range(self.in_channels):
                j = i if self.groups == 1 else 0
                self.weight.data[i, j] = bilinear_kernel

        @staticmethod
        def bilinear_kernel(stride):
            num_dims = len(stride)

            shape = (1,) * num_dims
            bilinear_kernel = torch.ones(*shape)

            # The bilinear kernel is separable in its spatial dimensions
            # Build up the kernel channel by channel
            for channel in range(num_dims):
                channel_stride = stride[channel]
                kernel_size = 2 * channel_stride - channel_stride % 2
                # e.g. with stride = 4
                # delta = [-3, -2, -1, 0, 1, 2, 3]
                # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
                # delta = torch.arange(1 - channel_stride, channel_stride)
                delta = torch.arange(0, kernel_size)
                delta = delta - (channel_stride - 0.5) if channel_stride % 2 == 0 else delta - (channel_stride - 1)
                channel_filter = (1 - torch.abs(delta / float(channel_stride)))
                # Apply the channel filter to the current channel
                shape = [1] * num_dims
                shape[channel] = kernel_size
                bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
            return bilinear_kernel

    aa = BilinearConvTranspose2d(channel, factor)
    cc = aa(torch.from_numpy(np.expand_dims(img.transpose(2, 0, 1), 0).astype('float32')))
    icc = cc.detach().numpy()[0].transpose(1, 2, 0)
    print(np.allclose(iaa, icc))
    # False
    ```
  - **tf.nn.conv2d_transpose**
    ```py
    # This is same with pytorch bilinear kernel
    def upsample_filt(size):
        factor = (size + 1) // 2
        center = factor - 1 if size % 2 == 1 else factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    def bilinear_upsample_weights(factor, number_of_classes=3):
        filter_size = 2 * factor - factor % 2
        weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes), dtype=np.float32)
        upsample_kernel = upsample_filt(filter_size)

        for i in range(number_of_classes):
            weights[:, :, i, i] = upsample_kernel
        return weights

    idd = tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), bilinear_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor, img.shape[1] * factor, channel], strides=factor, padding='SAME')[0]
    print(np.allclose(icc, idd))
    # True
    ```
  - **Conv2DTranspose**
    ```py
    aa = keras.layers.Conv2DTranspose(channel, 2 * factor - factor % 2, padding='same', strides=factor, use_bias=False)
    aa.build([None, None, None, channel])
    aa.set_weights([bilinear_upsample_weights(factor, channel)])
    iee = aa(tf.expand_dims(tf.cast(img, 'float32'), 0)).numpy()[0]
    ```
  - **Plot**
    ```py
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    imgs = [img, iaa, ibb, icc, idd, iee]
    names = ["Orignal", "tf.image.resize", "UpSampling2D", "Pytorch ConvTranspose2d", "tf.nn.conv2d_transpose", "TF Conv2DTranspose"]
    for ax, imm, nn in zip(axes, imgs, names):
        ax.imshow(imm)
        ax.axis('off')
        ax.set_title(nn)
    plt.tight_layout()
    ```
    ```py
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1])
    ```
## Clone model
  ```py
  def convert_UpSampling2D_layer(layer):
      print(layer.name)
      if isinstance(layer, keras.layers.UpSampling2D):
          print(">>>> Convert UpSampling2D <<<<")
          channel = layer.input.shape[-1]
          factor = 2
          aa = keras.layers.Conv2DTranspose(channel, 2 * factor - factor % 2, padding='same', strides=factor, use_bias=False)
          aa.build(layer.input.shape)
          aa.set_weights([bilinear_upsample_weights(factor, number_of_classes=channel)])
          return aa
      return layer

  mm = keras.models.load_model('aa.h5', compile=False)
  mmn = keras.models.clone_model(mm, clone_function=convert_UpSampling2D_layer)
  ```
***

# Replace PReLU with DepthwiseConv2D
  ```py
  '''
  mm = keras.models.load_model('./checkpoints/keras_se_mobile_facenet_emore_VIII_basic_agedb_30_epoch_12_0.931000.h5')
  aa = mm.layers[-7]
  ii = np.arange(-1, 1, 2 / (7 * 7 * 512), dtype=np.float32)[:7 * 7 * 512].reshape([1, 7, 7, 512])
  ee = my_activate_test(ii, weights=aa.get_weights())
  np.alltrue(aa(ii) == ee)
  '''
  def my_activate_test(inputs, weights=None):
      channel_axis = 1 if K.image_data_format() == "channels_first" else -1
      pos = K.relu(inputs)
      nn = DepthwiseConv2D((1, 1), depth_multiplier=1, use_bias=False)
      if weights is not None:
          nn.build(inputs.shape)
          nn.set_weights([tf.reshape(weights[id], nn.weights[id].shape) for id, ii in enumerate(weights)])
      neg = -1 * nn(K.relu(-1 * inputs))
      return pos + neg
  ```
  ```py
  from backbones import mobile_facenet_mnn
  bb = mobile_facenet_mnn.mobile_facenet(256, (112, 112, 3), 0.4, use_se=True)
  bb.build((112, 112, 3))

  bb_id = 0
  for id, ii in enumerate(mm.layers):
      print(id, ii.name)
      if isinstance(ii, keras.layers.PReLU):
          print("PReLU")
          nn = bb.layers[bb_id + 2]
          print(bb_id, nn.name)
          nn.set_weights([tf.reshape(wii, nn.weights[wid].shape) for wid, wii in enumerate(ii.get_weights())])
          bb_id += 6
      else:
          nn = bb.layers[bb_id]
          print(bb_id, nn.name)
          nn.set_weights(ii.get_weights())
          bb_id += 1

  inputs = bb.inputs[0]
  embedding = bb.outputs[0]
  output = keras.layers.Dense(tt.classes, name=tt.softmax, activation="softmax")(embedding)
  model = keras.models.Model(inputs, output)
  model.layers[-1].set_weights(tt.model.layers[-2].get_weights())
  model_c = keras.models.Model(model.inputs[0], keras.layers.concatenate([bb.outputs[0], model.outputs[-1]]))
  model_c.compile(optimizer=tt.model.optimizer, loss=tt.model.loss, metrics=tt.model.metrics)
  model_c.optimizer.set_weights(tt.model.optimizer.get_weights())
  ```
  **keras.models.clone_model**
  ```py
  from tensorflow.keras import backend as K
  from tensorflow.keras.layers import DepthwiseConv2D

  # MUST be a customized layer
  # Using DepthwiseConv2D re-implementing PReLU, as MNN doesnt support it...
  class My_PRELU_act(keras.layers.Layer):
      def __init__(self, **kwargs):
          super(My_PRELU_act, self).__init__(**kwargs)
          # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
      def build(self, input_shape):
          self.dconv = DepthwiseConv2D((1, 1), depth_multiplier=1, use_bias=False)
      def call(self, inputs, **kwargs):
          pos = K.relu(inputs)
          neg = -1 * self.dconv(K.relu(-1 * inputs))
          return pos + neg
      def compute_output_shape(self, input_shape):
          return input_shape
      def get_config(self):
          config = super(My_PRELU_act, self).get_config()
          return config
      @classmethod
      def from_config(cls, config):
          return cls(**config)

  def convert_prelu_layer(layer):
      print(layer.name)
      if isinstance(layer, keras.layers.PReLU):
          print(">>>> Convert PReLu <<<<")
          return My_PRELU_act()
      return layer

  mm = keras.models.load_model('checkpoints/keras_se_mobile_facenet_emore_IV_basic_agedb_30_epoch_48_0.957833.h5', compile=False)
  mmn = keras.models.clone_model(mm, clone_function=convert_prelu_layer)
  ```
***

# OCR
  ```sh
  docker run -it -p 8866:8866 paddleocr:cpu bash
  git clone https://gitee.com/PaddlePaddle/PaddleOCR
  sed -i 's/ch_det_mv3_db/ch_det_r50_vd_db/' deploy/hubserving/ocr_system/params.py
  sed -i 's/ch_rec_mv3_crnn/ch_rec_r34_vd_crnn_enhance/' deploy/hubserving/ocr_system/params.py
  export PYTHONPATH=. && hub uninstall ocr_system; hub install deploy/hubserving/ocr_system/ && hub serving start -m ocr_system

  OCR_DID=`docker ps -a | sed -n '2,2p' | cut -d ' ' -f 1`
  docker cp ch_det_r50_vd_db_infer/* $OCR_DID:/PaddleOCR/inference/
  docker cp ch_rec_r34_vd_crnn_enhance_infer/* $OCR_DID:/PaddleOCR/inference/

  ```
  ```sh
  IMG_STR=`base64 -w 0 $TEST_PIC`
  echo "{\"images\": [\"`base64 -w 0 Selection_261.png`\"]}" > foo
  curl -H "Content-Type:application/json" -X POST --data "{\"images\": [\"填入图片Base64编码(需要删除'data:image/jpg;base64,'）\"]}" http://localhost:8866/predict/ocr_system
  curl -H "Content-Type:application/json" -X POST --data "{\"images\": [\"`base64 -w 0 Selection_101.png`\"]}" http://localhost:8866/predict/ocr_system
  curl -H "Content-Type:application/json" -X POST --data foo http://localhost:8866/predict/ocr_system
  echo "{\"images\": [\"`base64 -w 0 Selection_261.png`\"]}" | curl -v -X PUT -H 'Content-Type:application/json' -d @- http://localhost:8866/predict/ocr_system
  ```
  ```py
  import requests
  import base64
  import json
  from matplotlib.font_manager import FontProperties

  class PaddleOCR:
      def __init__(self, url, font='/usr/share/fonts/opentype/noto/NotoSerifCJK-Light.ttc'):
          self.url = url
          self.font = FontProperties(fname=font)
      def __call__(self, img_path, thresh=0.9, show=2):
          with open(img_path, 'rb') as ff:
              aa = ff.read()
          bb = base64.b64encode(aa).decode()
          rr = requests.post(self.url, headers={"Content-type": "application/json"}, data='{"images": ["%s"]}' % bb)
          dd = json.loads(rr.text)

          imm = imread(img_path)
          if show == 0:
              return dd
          if show == 1:
              fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
              axes = [axes, axes]
          elif show == 2:
              fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
              axes[1].invert_yaxis()
          axes[0].imshow(imm)
          for ii in dd['results'][0]:
              jj = np.array(ii['text_region'])
              kk = np.vstack([jj, jj[0]])
              axes[0].plot([ii[0] for ii in kk], [ii[1] for ii in kk], 'r')
              axes[1].text(jj[-1, 0], jj[-1, 1], ii['text'], fontproperties=font, fontsize=(jj[-1, 1]-jj[0, 1])/2)
          # plt.tight_layout()
          return dd

  pp = PaddleOCR("http://localhost:8866/predict/ocr_system")
  pp("./Selection_261.png")
  ```
  ```sh
  python3 tools/infer/predict_system.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/"  --rec_model_dir="./inference/rec_crnn/"
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/elementwise_layers.py
  def convert_reciprocal(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert element-wise division
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """     
      logger = logging.getLogger('onnx2keras:reciprocal')
      print(layers[node.input[0]])

      if len(node.input) != 1:
          assert AttributeError('Not 1 input for reciprocal layer.')

      layers[node_name] = 1 / layers[node.input[0]]
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/upsampling_layers.py
  def convert_resize(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert upsample.
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """
      logger = logging.getLogger('onnx2keras:resize')
      logger.warning('!!! EXPERIMENTAL SUPPORT (resize) !!!')
      print([layers[ii] for ii in node.input])

      if len(node.input) != 1:
          if node.input[-1] in layers and isinstance(layers[node.input[-1]], np.ndarray):
              params['scales'] = layers[node.input[-1]]
          else:
              raise AttributeError('Unsupported number of inputs')

      if params['mode'].decode('utf-8') != 'nearest':
          logger.error('Cannot convert non-nearest upsampling.')
          raise AssertionError('Cannot convert non-nearest upsampling')

      scale = np.uint8(params['scales'][-2:])

      upsampling = keras.layers.UpSampling2D(
          size=scale, name=keras_name
      )   

      layers[node_name] = upsampling(layers[node.input[0]])
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/operation_layers.py
  def convert_clip(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert clip layer
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """
      logger = logging.getLogger('onnx2keras:clip')
      if len(node.input) != 1:
          assert AttributeError('More than 1 input for clip layer.')

      input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
      print(node.input, [layers[ii] for ii in node.input], node_name, params)
      if len(node.input == 3):
          params['min'] = layers[node.input[1]]
          params['max'] = layers[node.input[2]]

      if params['min'] == 0:
          logger.debug("Using ReLU({0}) instead of clip".format(params['max']))
          layer = keras.layers.ReLU(max_value=params['max'], name=keras_name)
      else:
          def target_layer(x, vmin=params['min'], vmax=params['max']):
              import tensorflow as tf
              return tf.clip_by_value(x, vmin, vmax)
          layer = keras.layers.Lambda(target_layer, name=keras_name)
          lambda_func[keras_name] = target_layer

      layers[node_name] = layer(input_0)
  ```
  ```py
  # /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/activation_layers.py
  def convert_hard_sigmoid(node, params, layers, lambda_func, node_name, keras_name):
      """
      Convert Sigmoid activation layer
      :param node: current operation node
      :param params: operation attributes
      :param layers: available keras layers
      :param lambda_func: function for keras Lambda layer
      :param node_name: internal converter name
      :param keras_name: resulting layer name
      :return: None
      """
      if len(node.input) != 1:
          assert AttributeError('More than 1 input for an activation layer.')

      input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_name)
      hard_sigmoid = keras.layers.Activation(keras.activations.hard_sigmoid, name=keras_name)
      layers[node_name] = hard_sigmoid(input_0)
  ```
  ```py
  from .activation_layers import convert_hard_sigmoid
  from .elementwise_layers import convert_reciprocal
  from .upsampling_layers import convert_resize
  ```
***

## Multi GPU
  ```py
  tf.debugging.set_log_device_placement(True)

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
      inputs = tf.keras.layers.Input(shape=(1,))
      predictions = tf.keras.layers.Dense(1)(inputs)
      model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
      model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))

  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
  model.fit(dataset, epochs=2)
  model.evaluate(dataset)
  ```
  ```py
  mirrored_strategy = tf.distribute.MirroredStrategy()
  # Compute global batch size using number of replicas.
  BATCH_SIZE_PER_REPLICA = 5
  global_batch_size = (BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync)
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100)
  dataset = dataset.batch(global_batch_size)

  LEARNING_RATES_BY_BATCH_SIZE = {5: 0.1, 10: 0.15}
  learning_rate = LEARNING_RATES_BY_BATCH_SIZE[global_batch_size]

  with mirrored_strategy.scope():
      model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
      optimizer = tf.keras.optimizers.SGD()

  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(global_batch_size)
  dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

  @tf.function
  def train_step(dist_inputs):
      def step_fn(inputs):
          features, labels = inputs

          with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
            logits = model(features, training=True)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

          grads = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
          return cross_entropy

      per_example_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
      mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
      return mean_loss

  with mirrored_strategy.scope():
      for inputs in dist_dataset:
          print(train_step(inputs))
  ```
  ```py
  @tf.function
  def step_fn(inputs):
      return ss.experimental_assign_to_logical_device(mm.predict(inputs), 0)

  with ss.scope():
      ss.run(step_fn, args=(np.ones([2, 112, 112, 3]),))
  ```
## NPU
  ```sh
  资料：建议用谷歌浏览器或QQ浏览器浏览可以直接网页翻译！
  文档汇总： https://docs.khadas.com/vim3/index.html
  烧录教程： https://docs.khadas.com/vim3/UpgradeViaUSBCable.html
  硬件资料： https://docs.khadas.com/vim3/HardwareDocs.html
  安卓固件下载： https://docs.khadas.com/vim3/FirmwareAndroid.html
  Ubuntu固件下载： https://docs.khadas.com/vim3/FirmwareUbuntu.html
  第三方操作系统： https://docs.khadas.com/vim3/FirmwareThirdparty.html#AndroidTV

  VIM3释放 NPU资料流程: https://www.khadas.com/npu-toolkit-vim3
  ```
## tf dataset cache
  - **dataset.cache** **MUST** be placed **before** data random augment and shuffle
  ```py
  dd = np.arange(30).reshape(3, 10)

  ''' Cache before shuffle and random '''
  ds = tf.data.Dataset.from_tensor_slices(dd)
  # ds = ds.cache()
  ds = ds.shuffle(dd.shape[0])
  ds = ds.map(lambda xx: xx + tf.random.uniform((1,), 1, 10, dtype=tf.int64))

  for ii in range(3):
      print(">>>> Epoch:", ii)
      for jj in ds:
          print(jj)
  # >>>> Epoch: 0
  # tf.Tensor([ 9 10 11 12 13 14 15 16 17 18], shape=(10,), dtype=int64)
  # tf.Tensor([13 14 15 16 17 18 19 20 21 22], shape=(10,), dtype=int64)
  # tf.Tensor([23 24 25 26 27 28 29 30 31 32], shape=(10,), dtype=int64)
  # >>>> Epoch: 1
  # tf.Tensor([11 12 13 14 15 16 17 18 19 20], shape=(10,), dtype=int64)
  # tf.Tensor([21 22 23 24 25 26 27 28 29 30], shape=(10,), dtype=int64)
  # tf.Tensor([ 9 10 11 12 13 14 15 16 17 18], shape=(10,), dtype=int64)
  # >>>> Epoch: 2
  # tf.Tensor([23 24 25 26 27 28 29 30 31 32], shape=(10,), dtype=int64)
  # tf.Tensor([12 13 14 15 16 17 18 19 20 21], shape=(10,), dtype=int64)
  # tf.Tensor([ 3  4  5  6  7  8  9 10 11 12], shape=(10,), dtype=int64)

  ''' Cache before random but after shuffle '''
  ds2 = tf.data.Dataset.from_tensor_slices(dd)
  ds2 = ds2.shuffle(dd.shape[0])
  ds2 = ds2.cache()
  ds2 = ds2.map(lambda xx: xx + tf.random.uniform((1,), 1, 10, dtype=tf.int64))

  for ii in range(3):
      print(">>>> Epoch:", ii)
      for jj in ds2:
          print(jj)
  # >>>> Epoch: 0
  # tf.Tensor([26 27 28 29 30 31 32 33 34 35], shape=(10,), dtype=int64)
  # tf.Tensor([17 18 19 20 21 22 23 24 25 26], shape=(10,), dtype=int64)
  # tf.Tensor([ 6  7  8  9 10 11 12 13 14 15], shape=(10,), dtype=int64)
  # >>>> Epoch: 1
  # tf.Tensor([22 23 24 25 26 27 28 29 30 31], shape=(10,), dtype=int64)
  # tf.Tensor([17 18 19 20 21 22 23 24 25 26], shape=(10,), dtype=int64)
  # tf.Tensor([ 3  4  5  6  7  8  9 10 11 12], shape=(10,), dtype=int64)
  # >>>> Epoch: 2
  # tf.Tensor([21 22 23 24 25 26 27 28 29 30], shape=(10,), dtype=int64)
  # tf.Tensor([15 16 17 18 19 20 21 22 23 24], shape=(10,), dtype=int64)
  # tf.Tensor([ 3  4  5  6  7  8  9 10 11 12], shape=(10,), dtype=int64)

  ''' Cache after random and shuffle '''
  ds3 = tf.data.Dataset.from_tensor_slices(dd)
  ds3 = ds3.shuffle(dd.shape[0])
  ds3 = ds3.map(lambda xx: xx + tf.random.uniform((1,), 1, 10, dtype=tf.int64))
  ds3 = ds3.cache()

  for ii in range(3):
      print(">>>> Epoch:", ii)
      for jj in ds3:
          print(jj)
  # >>>> Epoch: 0
  # tf.Tensor([24 25 26 27 28 29 30 31 32 33], shape=(10,), dtype=int64)
  # tf.Tensor([14 15 16 17 18 19 20 21 22 23], shape=(10,), dtype=int64)
  # tf.Tensor([ 4  5  6  7  8  9 10 11 12 13], shape=(10,), dtype=int64)
  # >>>> Epoch: 1
  # tf.Tensor([24 25 26 27 28 29 30 31 32 33], shape=(10,), dtype=int64)
  # tf.Tensor([14 15 16 17 18 19 20 21 22 23], shape=(10,), dtype=int64)
  # tf.Tensor([ 4  5  6  7  8  9 10 11 12 13], shape=(10,), dtype=int64)
  # >>>> Epoch: 2
  # tf.Tensor([24 25 26 27 28 29 30 31 32 33], shape=(10,), dtype=int64)
  # tf.Tensor([14 15 16 17 18 19 20 21 22 23], shape=(10,), dtype=int64)
  # tf.Tensor([ 4  5  6  7  8  9 10 11 12 13], shape=(10,), dtype=int64)
  ```
## ReLU Activation
  ```py
  keras.activations.relu 将矩阵x内所有负值都设为零，其余的值不变
  `max(x, 0)`

  keras.activations.elu
  `x` if `x > 0`
  `alpha * (exp(x)-1)` if `x < 0`

  keras.layers.LeakyReLU
  `f(x) = alpha * x for x < 0`
  `f(x) = x for x >= 0`


  keras.layers.PReLU
  `f(x) = alpha * x for x < 0`
  `f(x) = x for x >= 0`
  where `alpha` is a learned array with the same shape as x
  ```
## Route trace
  traceroute www.baidu.com
  dig +trace www.baidu.com
  speedtest-cli
  mtr -r -c 30 -s 1024 www.baidu.com

  !wget http://im.tdweilai.com:38831/keras_ResNest101_emore_II_basic_agedb_30_epoch_64_0.968500.h5

  netstat -na | awk -F ' ' 'BEGIN {WAIT_C=0; EST_C=0} /:6379 /{if($NF == "TIME_WAIT"){WAIT_C=WAIT_C+1}else{EST_C=EST_C+1}} END {print "TIME_WAIT: "WAIT_C", ESTABLISHED: "EST_C}'

  watch -tn 3 'echo ">>>> php connection:"; ss -pl | grep php | wc -l; echo ">>>> 8800 Chat-Server connection:"; netstat -na | grep -i ":8800 " | wc -l; echo ">>>> 8812 Fs-Server connection:"; netstat -na | grep -i ":8812 " | wc -l; echo ">>>> 6379 redis-server connection:"; netstat -na | grep -i ":6379 " | wc -l; echo ">>>> Socket status:"; ss -s; echo ">>>> top"; top -b -n 1'

  查看占用端口的进程 ss -lntpd | grep :4444

  ss -lntpd | grep :8800 -- pid=8431
  ss -lntpd | grep :8812 -- pid=9029

  ps -lax | grep 8431 -- ppid 470
  ps -lax | grep 9029 -- ppid 470

  docker container ls

  3306 docker-containerd -- 470 docker-containerd-shim tdwl_ws -- 8431 Chat-Server:master -- 8432 Chat-Server:manager -- MULTI Chat-Server:work
                                                               -- 9029 Fs-Server:master -- 9030 Fs-Server:manager -- MULTI Fs-Server:work
                         -- 23830 docker-containerd-shim fs -- 24107 freeswitch -nonat -nc
                         -- 23847 bash -- 24153 fs_cli

  1 -- 1637 /usr/bin/dockerd -- 3306 docker-containerd

  ss -lntpd | grep -i freeswitch
  ss -lntpd | grep -i php
  ss -lntpd | grep -i redis
***

# Tensorflow Horovod and Distribute
## Install horovod
  - [NVIDIA Collective Communications Library (NCCL) Download Page](https://developer.nvidia.com/nccl/nccl-download)
  - [Horovod on GPU](https://github.com/horovod/horovod/blob/master/docs/gpus.rst)
  ```sh
  sudo apt install gcc-8 g++-8
  sudo rm /etc/alternatives/c++ && sudo ln -s /usr/bin/x86_64-linux-gnu-g++-8 /etc/alternatives/c++
  sudo apt install openmpi-bin

  sudo dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
  sudo apt update

  nvidia-smi
  sudo apt install libnccl2=2.8.3-1+cuda11.0 libnccl-dev=2.8.3-1+cuda11.0
  sudo apt-mark hold libnccl-dev libnccl2
  HOROVOD_GPU_OPERATIONS=NCCL pip install horovod
  ```
  ```sh
  git clone https://github.com/horovod/horovod.git
  cd horovod/examples/tensorflow2/

  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow2_keras_mnist.py
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow2_keras_synthetic_benchmark.py --model MobileNet
  # [1,0]<stdout>:Iter #4: 298.7 img/sec per GPU
  # [1,0]<stdout>:Total img/sec on 1 GPU(s): 291.3 +-12.8
  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow2_keras_synthetic_benchmark.py --model MobileNet --batch-size 64
  # [1,0]<stdout>:Iter #8: 284.7 img/sec per GPU
  # [1,0]<stdout>:Total img/sec on 1 GPU(s): 277.2 +-10.5
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow2_keras_synthetic_benchmark.py --model MobileNet
  # [1,0]<stdout>:Iter #6: 267.9 img/sec per GPU
  # [1,0]<stdout>:Total img/sec on 2 GPU(s): 530.5 +-10.5
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow2_keras_synthetic_benchmark.py --model MobileNet --fp16-allreduce
  # [1,0]<stdout>:Iter #5: 267.7 img/sec per GPU
  # [1,0]<stdout>:Total img/sec on 2 GPU(s): 528.7 +-8.6
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow2_keras_synthetic_benchmark.py --model MobileNet --batch-size 64
  # [1,0]<stdout>:Iter #9: 268.6 img/sec per GPU
  # [1,0]<stdout>:Total img/sec on 2 GPU(s): 526.6 +-9.7
  ```
## Tensorflow horovod
  ```py
  #!/usr/bin/env python3
  import tensorflow as tf
  import horovod.tensorflow.keras as hvd
  import argparse
  import sys
  import numpy as np

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-b", "--batch_size", type=int, help="batch_size", default=8)
  parser.add_argument("-e", "--epochs", type=int, help="epochs", default=10)
  parser.add_argument("-m", "--model_name", type=str, help="model name", default="MobileNet")
  parser.add_argument('--fp16_allreduce', action='store_true', default=False, help='fp16 compression allreduce')
  args = parser.parse_known_args(sys.argv[1:])[0]

  # Horovod: initialize Horovod.
  hvd.init()
  print(">>>> hvd.rank:", hvd.rank(), "hvd.size:", hvd.size())

  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  data = np.random.uniform(size=[1024, 224, 224, 3])
  target = np.random.uniform(size=[1024, 1], low=0, high=999).astype("int64")
  dataset = tf.data.Dataset.from_tensor_slices((data, target)).repeat().batch(args.batch_size)
  steps_per_epoch = int(np.ceil(data.shape[0] / args.batch_size))

  model = getattr(tf.keras.applications, args.model_name)(weights=None)

  opt = tf.optimizers.SGD(0.01)
  compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
  opt = hvd.DistributedOptimizer(opt, compression=compression)

  model.compile(loss=tf.losses.SparseCategoricalCrossentropy(), optimizer=opt, experimental_run_tf_function=False)
  callbacks = [
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
      hvd.callbacks.MetricAverageCallback(),
  ]
  verbose = 1 if hvd.rank() == 0 else 0
  model.fit(dataset, steps_per_epoch=steps_per_epoch // hvd.size(), callbacks=callbacks, epochs=args.epochs, verbose=verbose)
  ```
## Tensorflow distribute
  ```py
  #!/usr/bin/env python3
  import tensorflow as tf
  import argparse
  import sys
  import numpy as np

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-s", "--strategy", type=int, help="{1: OneDeviceStrategy, 2: MirroredStrategy, 3: MultiWorkerMirroredStrategy}", default=1)
  parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=8)
  parser.add_argument("-e", "--epochs", type=int, help="epochs", default=10)
  parser.add_argument("-m", "--model_name", type=str, help="model name", default="MobileNet")
  args = parser.parse_known_args(sys.argv[1:])[0]

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

  if args.strategy == 2:
      strategy = tf.distribute.MirroredStrategy()
  elif args.strategy == 3:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
  else:
      strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

  batch_size = args.batch_size * strategy.num_replicas_in_sync
  data = np.random.uniform(size=[1024, 224, 224, 3])
  target = np.random.uniform(size=[1024, 1], low=0, high=999).astype("int64")
  dataset = tf.data.Dataset.from_tensor_slices((data, target)).batch(batch_size)

  with strategy.scope():
      model = getattr(tf.keras.applications, args.model_name)(weights=None)

  # opt = tf.optimizers.Adam(0.001)
  opt = tf.optimizers.SGD(0.01)
  model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy())
  callbacks = []
  model.fit(dataset, callbacks=callbacks, epochs=args.epochs, verbose=1)
  ```
## Test results
  ```sh
  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 32 -s 1 -e 6 -m ResNet101V2
  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 64 -s 1 -e 6 -m ResNet101V2
  CUDA_VISIBLE_DEVICES='0,1' python tensorflow_distribute.py -b 32 -s 2 -e 6 -m ResNet101V2

  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 32 -s 1 -e 6 -m ResNet50V2
  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 64 -s 1 -e 6 -m ResNet50V2
  CUDA_VISIBLE_DEVICES='0,1' python tensorflow_distribute.py -b 32 -s 2 -e 6 -m ResNet50V2

  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 128 -s 1 -e 6 -m ResNet50V2
  CUDA_VISIBLE_DEVICES='0,1' python tensorflow_distribute.py -b 128 -s 2 -e 6 -m ResNet50V2

  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 32 -s 2 -e 6 -m MobileNet
  CUDA_VISIBLE_DEVICES='0,1' python tensorflow_distribute.py -b 32 -s 2 -e 6 -m MobileNet

  CUDA_VISIBLE_DEVICES='1' python tensorflow_distribute.py -b 128 -s 2 -e 6 -m MobileNet
  CUDA_VISIBLE_DEVICES='0,1' python tensorflow_distribute.py -b 128 -s 2 -e 6 -m MobileNet
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow_horovod.py -b 32 -e 6 -m ResNet101V2
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow_horovod.py -b 32 -e 6 -m ResNet101V2

  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow_horovod.py -b 32 -e 6 -m ResNet50V2
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow_horovod.py -b 32 -e 6 -m ResNet50V2

  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow_horovod.py -b 128 -e 6 -m ResNet50V2
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow_horovod.py -b 128 -e 6 -m ResNet50V2

  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow_horovod.py -b 32 -e 6 -m MobileNet
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow_horovod.py -b 32 -e 6 -m MobileNet

  CUDA_VISIBLE_DEVICES='1' horovodrun -np 1 -H localhost:1 python tensorflow_horovod.py -b 128 -e 6 -m MobileNet
  CUDA_VISIBLE_DEVICES='0,1' horovodrun -np 2 -H localhost:2 python tensorflow_horovod.py -b 128 -e 6 -m MobileNet
  ```

  | strategy          | batch size | mean time      | GPU memory   |
  | ----------------- | ---------- | -------------- | ------------ |
  | **ResNet101V2**   |            |                |              |
  | OneDeviceStrategy | 32         | 201ms/step     | 8897MiB      |
  | OneDeviceStrategy | 64         | 380ms/step     | 17089MiB     |
  | MirroredStrategy  | 32 * 2     | 246ms/step     | 8909MiB * 2  |
  | horovod, cuda 1   | 32         | 223ms/step     | 8925MiB      |
  | horovod, cuda 0,1 | 32 * 2     | **241ms/step** | 8925MiB * 2  |
  | **ResNet50V2**    |            |                |              |
  | OneDeviceStrategy | 32         | 120ms/step     | 8897MiB      |
  | OneDeviceStrategy | 64         | 224ms/step     | 8897MiB      |
  | MirroredStrategy  | 32 * 2     | **149ms/step** | 8897MiB * 2  |
  | horovod, cuda 1   | 32         | 146ms/step     | 8925MiB      |
  | horovod, cuda 0,1 | 32 * 2     | 154ms/step     | 8925MiB * 2  |
  | OneDeviceStrategy | 128        | 420ms/step     | 17089MiB     |
  | MirroredStrategy  | 128 * 2    | **360ms/step** | 17089MiB * 2 |
  | horovod, cuda 1   | 128        | 474ms/step     | 17117MiB     |
  | horovod, cuda 0,1 | 128 * 2    | 421ms/step     | 17117MiB * 2 |
  | **MobileNet**     |            |                |              |
  | OneDeviceStrategy | 32         | 105ms/step     |              |
  | MirroredStrategy  | 32 * 2     | **116ms/step** |              |
  | horovod, cuda 1   | 32         | 130ms/step     |              |
  | horovod, cuda 0,1 | 32 * 2     | 135ms/step     |              |
  | OneDeviceStrategy | 128        | 413ms/step     |              |
  | MirroredStrategy  | 128 * 2    | **333ms/step** |              |
  | horovod, cuda 1   | 128        | 450ms/step     |              |
  | horovod, cuda 0,1 | 128 * 2    | 397ms/step     |              |
***

# Weight decay
## MXNet SGD and tfa SGDW
  - [AdamW and Super-convergence is now the fastest way to train neural nets](https://www.fast.ai/2018/07/02/adam-weight-decay/)
  - The behavior of `weight_decay` in `mx.optimizer.SGD` and `tfa.optimizers.SGDW` is different.
  - **MXNet SGD** multiplies `wd` with `lr`.
    ```py
    import mxnet as mx
    help(mx.optimizer.SGD)
    # weight = weight - lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
    #        = (1 - lr * wd) * weight - lr * (rescale_grad * clip(grad, clip_gradient))
    ```
    Test with `learning_rate=0.1, weight_decay=5e-4`, weight is actually modified by `5e-5`.
    ```py
    import mxnet as mx
    mm_loss_grad = mx.nd.array([[1., 1], [1, 1]])

    mm = mx.nd.array([[1., 1], [1, 1]])
    mopt = mx.optimizer.SGD(learning_rate=0.1)
    mopt.update(0, mm, mm_loss_grad, None)
    print(mm.asnumpy())  # Basic value is `mm - lr * mm_loss = 0.9`
    # [[0.9 0.9] [0.9 0.9]]

    mm = mx.nd.array([[1., 1], [1, 1]])
    mopt = mx.optimizer.SGD(learning_rate=0.1, wd=5e-4)
    mopt.update(0, mm, mm_loss_grad, None)
    print(mm.asnumpy())  # 0.9 - 0.89995 = 5e-5
    # [[0.89995 0.89995]  [0.89995 0.89995]]
    ```
  - **tfa SGDW** behaves different, it does NOT multiply `wd` with `lr`. With `learning_rate=0.1, weight_decay=5e-4`, weight is actually modified with `5e-4`.
    ```py
    # /opt/anaconda3/lib/python3.7/site-packages/tensorflow_addons/optimizers/weight_decay_optimizers.py
    # 170     def _decay_weights_op(self, var, apply_state=None):
    # 177             return var.assign_sub(coefficients["wd_t"] * var, self._use_locking)
    ```
    ```py
    import tensorflow_addons as tfa
    ww_loss_grad = tf.convert_to_tensor([[1., 1.], [1., 1.]])
    ww = tf.Variable([[1., 1.], [1., 1.]])
    opt = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-4)
    opt.apply_gradients(zip([ww_loss_grad], [ww]))
    print(ww.numpy()) # 0.9 - 0.8995 = 5e-4
    # [[0.8995 0.8995] [0.8995 0.8995]]
    ```
    So `learning_rate=0.1, weight_decay=5e-4` in `mx.optimizer.SGD` is equal to `learning_rate=0.1, weight_decay=5e-5` in `tfa.optimizers.SGDW`.
  - **weight decay multiplier** If we set `wd_mult=10` in a MXNet layer, `wd` will mutiply by `10` in this layer. This means it will be `weight_decay == 5e-4` in a keras layer.
    ```py
    # https://github.com/apache/incubator-mxnet/blob/e6cea0d867329131fa6052e5f45dc5f626c00d72/python/mxnet/optimizer/optimizer.py#L482
    # 29  class Optimizer(object):
    # 482                lrs[i] *= self.param_dict[index].lr_mult
    ```
## L2 Regularization and Weight Decay
  - [Weight Decay == L2 Regularization?](https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd)
  - [PDF DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf)
  - **Keras l2 regularization**
    ```py
    ww = tf.convert_to_tensor([[1.0, -2.0], [-3.0, 4.0]])

    # loss = l2 * reduce_sum(square(x))
    aa = keras.regularizers.L2(0.2)
    aa(ww)  # tf.reduce_sum(ww ** 2) * 0.2
    # 6.0

    # output = sum(t ** 2) / 2
    tf.nn.l2_loss(ww)
    # 15.0
    tf.nn.l2_loss(ww) * 0.2
    # 3.0
    ```
    Total loss with l2 regularization will be
    ```py
    total_loss = Loss(w) + λ * R(w)
    ```
  - `Keras.optimizers.SGD`
    ```py
    help(keras.optimizers.SGD)
    # w = w - learning_rate * g
    #   = w - learning_rate * g - learning_rate * Grad(l2_loss)
    ```
    So with `keras.regularizers.L2(λ)`, it should be
    ```py
    wd * weight = Grad(l2_loss)
        --> wd * weight = 2 * λ * weight
        --> λ = wd / 2
    ```
    **Test**
    ```py
    ww_loss_grad = tf.convert_to_tensor([[1., 1.], [1., 1.]])
    ww = tf.Variable([[1., 1.], [1., 1.]])
    opt = keras.optimizers.SGD(0.1)
    with tf.GradientTape() as tape:
        # l2_loss = tf.nn.l2_loss(ww) * 5e-4
        l2_loss = keras.regularizers.L2(5e-4 / 2)(ww)  # `tf.nn.l2_loss` divided the loss by 2, `keras.regularizers.L2` not
    l2_grad = tape.gradient(l2_loss, ww).numpy()
    opt.apply_gradients(zip([ww_loss_grad + l2_grad], [ww]))
    print(ww.numpy()) # 0.9 - 0.89995 = 5e-5
    # [[0.89995 0.89995] [0.89995 0.89995]]
    ```
    That means the `L2_regulalizer` will modify the weights value by `l2 * lr == 5e-4 * 0.1 = 5e-5`.
  - If we want the same result as `mx.optimizer.SGD(learning_rate=0.1, wd=5e-4)` and `wd_mult=10` in a MXNet layer, which actually decay this layer's weights with `wd * wd_mult * learning_rate == 5e-4`, and other layers `wd * learning_rate == 5e-5`.
    - Firstlly, the keras optimizer is `tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-5)`.
    - Then add a `keras.regularizers.L2` with `l2 == weight_decay / learning_rate * (wd_mult - 1) / 2` to this layer.
    ```py
    ww_loss_grad = tf.convert_to_tensor([[1., 1.], [1., 1.]])
    ww = tf.Variable([[1., 1.], [1., 1.]])
    opt = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-5)
    with tf.GradientTape() as tape:
        l2_loss = keras.regularizers.L2(5e-5 / 0.1 * (10 - 1) / 2)(ww)
    l2_grad = tape.gradient(l2_loss, ww).numpy()
    opt.apply_gradients(zip([ww_loss_grad + l2_grad], [ww]))
    print(ww.numpy()) # 0.9 - 0.8995 = 5e-4
    # [[0.8995 0.8995] [0.8995 0.8995]]
    ```
## SGD with momentum
  - **MXNet**
    ```py
    # incubator-mxnet/python/mxnet/optimizer/sgd.py, incubator-mxnet/src/operator/optimizer_op.cc +109
    grad += wd * weight
    momentum_stat = momentum * momentum_stat - lr * grad
    weight += momentum_stat
    ```
  - **Keras SGDW** Using `wd == lr * wd`, `weight` will be the same with `MXNet SGD` in the first update, but `momentum_stat` will be different. Then in the second update, `weight` will also be different.
    ```py
    momentum_stat = momentum * momentum_stat - lr * grad
    weight += momentum_stat - wd * weight
    ```

  - **Keras SGD with l2 regularizer** can behave same as `MXNet SGD`
    ```py
    grad += regularizer_loss
    momentum_stat = momentum * momentum_stat - lr * grad
    weight += momentum_stat
    ```
## Keras Model test
  ```py
  import tensorflow_addons as tfa

  def test_optimizer_with_model(opt, epochs=3, l2=0):
      kernel_regularizer = None if l2 == 0 else keras.regularizers.L2(l2)
      aa = keras.layers.Dense(1, use_bias=False, kernel_initializer='ones', kernel_regularizer=kernel_regularizer)
      aa.build([1])
      mm = keras.Sequential([aa])
      loss = lambda y_true, y_pred: (y_true - y_pred) ** 2 / 2
      mm.compile(optimizer=opt, loss=loss)
      for ii in range(epochs):
          mm.fit([[1.]], [[0.]], epochs=ii+1, initial_epoch=ii, verbose=0)
          print("Epoch", ii, "- [weight]", aa.weights[0].numpy(), "- [losses]:", mm.history.history['loss'][0], end="")
          if len(opt.weights) > 1:
              print(" - [momentum]:", opt.weights[-1].numpy(), end="")
          print()
      return mm, opt

  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1), epochs=3)
  # Epoch 0 - [weight] [[0.9]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.81]] - [losses]: 0.4049999713897705
  # Epoch 2 - [weight] [[0.729]] - [losses]: 0.32804998755455017
  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1), l2=0.01, epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5099999904632568
  # Epoch 1 - [weight] [[0.806404]] - [losses]: 0.411266028881073
  # Epoch 2 - [weight] [[0.7241508]] - [losses]: 0.33164656162261963
  test_optimizer_with_model(tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=0.002), epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.806404]] - [losses]: 0.40320199728012085
  # Epoch 2 - [weight] [[0.72415084]] - [losses]: 0.3251436948776245
  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), epochs=3)
  # Epoch 0 - [weight] [[0.9]] - [losses]: 0.5 - [momentum]: [[-0.1]]
  # Epoch 1 - [weight] [[0.71999997]] - [losses]: 0.4049999713897705 - [momentum]: [[-0.17999999]]
  # Epoch 2 - [weight] [[0.486]] - [losses]: 0.25919997692108154 - [momentum]: [[-0.23399998]]
  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), l2=0.01, epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5099999904632568 - [momentum]: [[-0.102]] ==> 0.102 * 0.1
  # Epoch 1 - [weight] [[0.714604]] - [losses]: 0.411266028881073 - [momentum]: [[-0.183396]] ==> -0.102 * 0.9 - 0.898 * 1.02 * 0.1
  # Epoch 2 - [weight] [[0.47665802]] - [losses]: 0.2604360580444336 - [momentum]: [[-0.237946]]
  # ==> momentum_stat_2 == momentum_stat_1 * momentum - weight_1 * (1 + l2 * 2) * learning_rate
  test_optimizer_with_model(tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=0.002), epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5 - [momentum]: [[-0.1]]
  # Epoch 1 - [weight] [[0.71640396]] - [losses]: 0.40320199728012085 - [momentum]: [[-0.1798]]
  # Epoch 2 - [weight] [[0.48151073]] - [losses]: 0.25661730766296387 - [momentum]: [[-0.2334604]]

  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), l2=0.1, epochs=3)
  # Epoch 0 - [weight] [[0.88]] - [losses]: 0.6000000238418579 - [momentum]: [[-0.12]]
  # Epoch 1 - [weight] [[0.66639996]] - [losses]: 0.4646399915218353 - [momentum]: [[-0.21360001]]
  # Epoch 2 - [weight] [[0.39419195]] - [losses]: 0.266453355550766 - [momentum]: [[-0.272208]]
  ```
## MXNet model test
  - **wd_mult** NOT working if just added in `mx.symbol.Variable`, has to be added by `opt.set_wd_mult`.
  ```py
  import mxnet as mx
  import logging
  logging.getLogger().setLevel(logging.ERROR)

  def test_optimizer_with_mxnet_model(opt, epochs=3, wd_mult=None):
      xx, yy = np.array([[1.]]), np.array([[0.]])
      xx_input, yy_input = mx.nd.array(xx), mx.nd.array(yy)
      dataiter = mx.io.NDArrayIter(xx, yy)

      data = mx.symbol.Variable("data", shape=(1,))
      label = mx.symbol.Variable("softmax_label", shape=(1,))
      # ww = mx.symbol.Variable("ww", shape=(1, 1), wd_mult=wd_mult, init=mx.init.One())
      ww = mx.symbol.Variable("ww", shape=(1, 1), init=mx.init.One())
      nn = mx.sym.FullyConnected(data=data, weight=ww, no_bias=True, num_hidden=1)

      # loss = mx.symbol.SoftmaxOutput(data=nn, label=label, name='softmax')
      loss = mx.symbol.MakeLoss((label - nn) ** 2 / 2)
      # sss = loss.bind(mx.cpu(), {'data': xx_input, 'softmax_label': yy_input, 'ww': y_pred})
      # print(sss.forward()[0].asnumpy().tolist())
      # [[0.5]]
      if wd_mult is not None:
          opt.set_wd_mult({'ww': wd_mult})
      model = mx.mod.Module(context=mx.cpu(), symbol=loss)
      weight_value = mx.nd.ones([1, 1])
      for ii in range(epochs):
          loss_value = loss.bind(mx.cpu(), {'data': xx_input, 'softmax_label': yy_input, 'ww': weight_value}).forward()[0]
          # model.fit(train_data=dataiter, num_epoch=ii+1, begin_epoch=0, optimizer=opt, force_init=True)
          model.fit(train_data=dataiter, num_epoch=ii+1, begin_epoch=ii, optimizer=opt)
          weight_value = model.get_params()[0]['ww']
          # output = model.get_outputs()[0]
          print("Epoch", ii, "- [weight]", weight_value.asnumpy(), "- [losses]:", loss_value.asnumpy()[0, 0])
          # if len(opt.weights) > 1:
          #     print(" - [momentum]:", opt.weights[-1].numpy(), end="")
          # print()

  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, wd=0.02))
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.806404]] - [losses]: 0.403202
  # Epoch 2 - [weight] [[0.7241508]] - [losses]: 0.3251437
  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, wd=0.002))
  # Epoch 0 - [weight] [[0.8998]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.80964005]] - [losses]: 0.40482002
  # Epoch 2 - [weight] [[0.72851413]] - [losses]: 0.3277585
  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, momentum=0.9, wd=0.02))
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.714604]] - [losses]: 0.403202
  # Epoch 2 - [weight] [[0.47665802]] - [losses]: 0.25532946
  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, momentum=0.9, wd=0.02), wd_mult=10)
  # Epoch 0 - [weight] [[0.88]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.66639996]] - [losses]: 0.3872
  # Epoch 2 - [weight] [[0.39419195]] - [losses]: 0.22204445
  # ==> Equals to keras model `l2 == 0.1`
  ```
## Modify model with L2 regularizer
  ```py
  mm = keras.applications.MobileNet()

  regularizers_type = {}
  for layer in mm.layers:
      rrs = [kk for kk in layer.__dict__.keys() if 'regularizer' in kk and not kk.startswith('_')]
      if len(rrs) != 0:
          # print(layer.name, layer.__class__.__name__, rrs)
          if layer.__class__.__name__ not in regularizers_type:
              regularizers_type[layer.__class__.__name__] = rrs
  print(regularizers_type)
  # {'Conv2D': ['kernel_regularizer', 'bias_regularizer'],
  # 'BatchNormalization': ['beta_regularizer', 'gamma_regularizer'],
  # 'PReLU': ['alpha_regularizer'],
  # 'SeparableConv2D': ['kernel_regularizer', 'bias_regularizer', 'depthwise_regularizer', 'pointwise_regularizer'],
  # 'DepthwiseConv2D': ['kernel_regularizer', 'bias_regularizer', 'depthwise_regularizer'],
  # 'Dense': ['kernel_regularizer', 'bias_regularizer']}

  weight_decay = 5e-4
  for layer in mm.layers:
      if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.DepthwiseConv2D):
          print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
          layer.kernel_regularizer = keras.regularizers.L2(weight_decay / 2)
          if layer.use_bias:
              layer.bias_regularizer = keras.regularizers.L2(weight_decay / 2)
      if isinstance(layer, keras.layers.SeparableConv2D):
          print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
          layer.pointwise_regularizer = keras.regularizers.L2(weight_decay / 2)
          layer.depthwise_regularizer = keras.regularizers.L2(weight_decay / 2)
          if layer.use_bias:
              layer.bias_regularizer = keras.regularizers.L2(weight_decay / 2)
      if isinstance(layer, keras.layers.BatchNormalization):
          print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
          if layer.center:
              layer.beta_regularizer = keras.regularizers.L2(weight_decay / 2)
          if layer.scale:
              layer.gamma_regularizer = keras.regularizers.L2(weight_decay / 2)
      if isinstance(layer, keras.layers.PReLU):
          print(">>>> PReLU", layer.name)
          layer.alpha_regularizer = keras.regularizers.L2(weight_decay / 2)
  ```
## Optimizers with weight decay test
  ```py
  from tensorflow import keras
  import tensorflow_addons as tfa
  import losses, data, evals, myCallbacks, train
  # from tensorflow.keras.callbacks import LearningRateScheduler

  # Dataset
  data_path = '/datasets/faces_emore_112x112_folders'
  train_ds = data.prepare_dataset(data_path, batch_size=256, random_status=3, random_crop=(100, 100, 3))
  classes = train_ds.element_spec[-1].shape[-1]

  # Model
  basic_model = train.buildin_models("MobileNet", dropout=0, emb_shape=256)
  # model_output = keras.layers.Dense(classes, activation="softmax")(basic_model.outputs[0])
  model_output = train.NormDense(classes, name="arcface")(basic_model.outputs[0])
  model = keras.models.Model(basic_model.inputs[0], model_output)

  # Evals and basic callbacks
  save_name = 'keras_mxnet_test_sgdw'
  eval_paths = ['/datasets/faces_emore/lfw.bin', '/datasets/faces_emore/cfp_fp.bin', '/datasets/faces_emore/agedb_30.bin']
  my_evals = [evals.eval_callback(basic_model, ii, batch_size=256, eval_freq=1) for ii in eval_paths]
  my_evals[-1].save_model = save_name
  basic_callbacks = myCallbacks.basic_callbacks(checkpoint=save_name + '.h5', evals=my_evals, lr=0.001)
  basic_callbacks = basic_callbacks[:1] + basic_callbacks[2:]
  callbacks = my_evals + basic_callbacks
  # Compile and fit

  ss = myCallbacks.ConstantDecayScheduler([3, 5, 7], lr_base=0.1)
  optimizer = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-4, momentum=0.9)

  model.compile(optimizer=optimizer, loss=losses.arcface_loss, metrics=["accuracy"])
  # model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
  wd_callback = myCallbacks.OptimizerWeightDecay(optimizer.lr.numpy(), optimizer.weight_decay.numpy())
  model.fit(train_ds, epochs=15, callbacks=[ss, wd_callback, *callbacks], verbose=1)

  opt = tfa.optimizers.AdamW(weight_decay=lambda : None)
  opt.weight_decay = lambda : 5e-1 * opt.lr

  mlp.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy())
  ```
  ```py
  class Foo:
      def __init__(self, wd):
          self.wd = wd
      def __call__(self):
          return self.wd
      def set_wd(self, wd):
          self.wd = wd

  class L2_decay_wdm(keras.regularizers.L2):
      def __init__(self, wd_func=None, **kwargs):
          super(L2_decay_wdm, self).__init__(**kwargs)
          self.wd_func = wd_func

      def __call__(self, x):
          self.l2 = self.wd_func()
          # tf.print(", l2 =", self.l2, end='')
          return super(L2_decay_wdm, self).__call__(x)

      def get_config(self):
          self.l2 = 0  # Just a fake value for saving
          config = super(L2_decay_wdm, self).get_config()
          return config
  ```
***

# Distillation
## 链接
  - [知识蒸馏简述（一）](https://zhuanlan.zhihu.com/p/92166184)
## MNIST example
  - [Github keras-team/keras-io knowledge_distillation.py](https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py)
  ```py
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  import numpy as np

  # Create the teacher
  teacher = keras.Sequential(
      [
          layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
          layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
          layers.Flatten(),
          layers.Dense(10),
      ],
      name="teacher",
  )

  # Create the student
  student = keras.Sequential(
      [
          layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
          layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
          layers.Flatten(),
          layers.Dense(10),
      ],
      name="student",
  )

  # Prepare the train and test dataset.
  batch_size = 64
  # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  # x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(x_test, (-1, 28, 28, 1))
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

  # Normalize data
  x_train = x_train.astype("float32") / 255.0
  x_test = x_test.astype("float32") / 255.0

  # Train teacher as usual
  teacher.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  # Train and evaluate teacher on data.
  teacher.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
  teacher.evaluate(x_test, y_test)

  def create_distiller_model(teacher, student, clone=True):
      if clone:
          teacher_copy = keras.models.clone_model(teacher)
          student_copy = keras.models.clone_model(student)
      else:
          teacher_copy, student_copy = teacher, student

      teacher_copy.trainable = False
      student_copy.trainable = True
      inputs = teacher_copy.inputs[0]
      student_output = student_copy(inputs)
      teacher_output = teacher_copy(inputs)
      mm = keras.models.Model(inputs, keras.layers.Concatenate()([student_output, teacher_output]))
      return student_copy, mm

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

  student_copy, mm = create_distiller_model(teacher, student)
  mm.compile(optimizer=keras.optimizers.Adam(), loss=distiller_loss, metrics=[distiller_accuracy])
  mm.summary()
  mm.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

  mm.evaluate(x_test, y_test)
  student_copy.compile(metrics=["accuracy"])
  student_copy.evaluate(x_test, y_test)

  # Train student scratch
  student_scratch = keras.models.clone_model(student)
  student_scratch.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  student_scratch.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
  student_scratch.evaluate(x_test, y_test)
  ```
## Embedding
  ```py
  def tf_imread(file_path):
      img = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(img, channels=3) # [0, 255]
      img = tf.image.convert_image_dtype(img, tf.float32) # [0, 1]
      return img

  data_path = "faces_casia_112x112_folders_shuffle_label_embs.pkl"
  batch_size = 64
  aa = np.load(data_path, allow_pickle=True)
  image_names, image_classes, embeddings = aa['image_names'], aa['image_classes'], aa['embeddings']
  classes = np.max(image_classes) + 1
  print(">>>> Image length: %d, Image class length: %d, embeddings: %s" % (len(image_names), len(image_classes), np.shape(embeddings)))
  # >>>> Image length: 490623, Image class length: 490623, embeddings: (490623, 256)

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  dss = tf.data.Dataset.from_tensor_slices((image_names, image_classes, embeddings))
  ds = dss.map(lambda imm, label, emb: (tf_imread(imm), (tf.one_hot(label, depth=classes, dtype=tf.int32), emb)), num_parallel_calls=AUTOTUNE)

  ds = ds.batch(batch_size)  # Use batch --> map has slightly effect on dataset reading time, but harm the randomness
  ds = ds.map(lambda xx, yy: ((xx * 2) - 1, yy))
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
  xx.trainable = True
  inputs = keras.layers.Input(shape=(112, 112, 3))
  nn = xx(inputs)
  nn = keras.layers.GlobalAveragePooling2D()(nn)
  nn = keras.layers.BatchNormalization()(nn)
  # nn = layers.Dropout(0)(nn)
  embedding = keras.layers.Dense(256, name="embeddings")(nn)
  logits = keras.layers.Dense(classes, activation='softmax', name="logits")(embedding)

  model = keras.models.Model(inputs, [logits, embedding])

  def distiller_loss(true_emb_normed, pred_emb):
      pred_emb_normed = tf.nn.l2_normalize(pred_emb, axis=-1)
      # loss = tf.reduce_sum(tf.square(true_emb_normed - pred_emb_normed), axis=-1)
      loss = 1 - tf.reduce_sum(pred_emb_normed * true_emb_normed, axis=-1)
      return loss

  model.compile(optimizer='adam', loss=[keras.losses.categorical_crossentropy, distiller_loss], loss_weights=[1, 7])
  # model.compile(optimizer='adam', loss=[keras.losses.sparse_categorical_crossentropy, keras.losses.mse], metrics=['accuracy', 'mae'])
  model.summary()
  model.fit(ds)
  ```
***

层归一化-LN：用于计算递归神经网络沿通道的统计量；

权值归一化-WN：来参数化权值向量，用于监督图像识别、生成建模和深度强化学习；

切分归一化-DN：提出包含BN和LN层的归一化层，作为图像分类、语言建模和超分辨率的特例；

实例归一化-IN：为了进一步快速风格化，提出了IN层，主要用于图像分割迁移，其中统计量由高度和宽度维度计算得到；

组归一化-GN：对通道进行分组，统计每个分组通道的高度和宽度，增强对批量大小的稳定性；

位置归一化-PN：提出了位置归一化算法来计算生成网络沿信道维数的统计量；
***

# Attention
## Keras attention layers
  - [遍地开花的 Attention ，你真的懂吗？](https://developer.aliyun.com/article/713354)
  - [综述---图像处理中的注意力机制](https://blog.csdn.net/xys430381_1/article/details/89323444)
  - [全连接的图卷积网络(GCN)和self-attention这些机制有什么区别联系](https://www.zhihu.com/question/366088445/answer/1023290162)
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - [《Attention is All You Need》浅读（简介+代码）](https://spaces.ac.cn/archives/4765)
  - [3W字长文带你轻松入门视觉transformer](https://zhuanlan.zhihu.com/p/308301901)
  - `keras.layers.Attention` a.k.a. Luong-style attention.
  - `keras.layers.AdditiveAttention` a.k.a. Bahdanau-style attention. [Eager 执行环境与 Keras 定义 RNN 模型使用注意力机制为图片命名标题](https://github.com/leondgarse/Atom_notebook/blob/master/public/2018/09-06_tensorflow_tutotials.md#eager-%E6%89%A7%E8%A1%8C%E7%8E%AF%E5%A2%83%E4%B8%8E-keras-%E5%AE%9A%E4%B9%89-rnn-%E6%A8%A1%E5%9E%8B%E4%BD%BF%E7%94%A8%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%BA%E5%9B%BE%E7%89%87%E5%91%BD%E5%90%8D%E6%A0%87%E9%A2%98)
  - `keras.layers.MultiHeadAttention` multi-headed attention based on "Attention is all you Need"
  - [Github Keras Attention Augmented Convolutions](https://github.com/titu1994/keras-attention-augmented-convs)
***

## 图神经网络
  - [Github /dmlc/dgl](https://github.com/dmlc/dgl)
***
# datasets
```py
import time
from tqdm import tqdm
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in tqdm(dataset):
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)

import data
image_names, image_classes, embeddings, classes, _ = data.pre_process_folder('/datasets/faces_casia_112x112_folders/')
print(">>>> Image length: %d, Image class length: %d, classes: %d" % (len(image_names), len(image_classes), classes))
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_org = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
ds_org = ds_org.shuffle(buffer_size=len(image_names))

process_func = lambda imm, label: (data.tf_imread(imm), tf.one_hot(label, depth=classes, dtype=tf.int32))
ds = ds_org.map(process_func, num_parallel_calls=AUTOTUNE)
ds = ds.batch(128)  # Use batch --> map has slightly effect on dataset reading time, but harm the randomness
ds = ds.prefetch(buffer_size=AUTOTUNE)

benchmark(ds.batch(128))
# Execution time: 84.25039149800432
benchmark(ds.batch(128).prefetch(buffer_size=AUTOTUNE))
# Execution time: 84.96434965499793

benchmark(ds_org.interleave(lambda imm, label: tf.data.Dataset.from_tensors((imm, label)).map(process_func), num_parallel_calls=AUTOTUNE).batch(128))
# Execution time: 215.90566716800095

benchmark(tf.data.Dataset.range(2).interleave(lambda xx: ds, num_parallel_calls=AUTOTUNE).batch(128))
Execution time: 430.2685134439962 # 7666it

aa = ds_org.map(process_func, num_parallel_calls=AUTOTUNE).as_numpy_iterator()
benchmark(tf.data.Dataset.range(2).interleave(lambda xx: aa.next(), num_parallel_calls=AUTOTUNE).batch(128))
```
***

# Data Augmentation
```py
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
])
```
***
# Learning rate
- **keras.optimizers.schedules.LearningRateSchedule**
```py
from tensorflow.python.keras import backend as K

class lr_sch(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr=0.1):
        super(lr_sch, self).__init__()
        self.init_lr = init_lr
    def __call__(self, global_step:int):
        self.global_step = tf.cast(global_step, dtype=tf.float32)
        self.lr = self.init_lr / self.global_step
        tf.print("global_step:", self.global_step, "lr:", self.lr)
        # self.global_step = K.get_value(global_step)
        return self.lr

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=[28,28, 1]),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

optimizer = keras.optimizers.Adam(learning_rate=lr_sch(0.1))
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
# [0.0712303157694405, 0.9791]

np.argmax(model.predict(x_test[:1]))
```
```py
from myCallbacks import CosineLrScheduler
epochs = 50
first_restart_step=16
aa = CosineLrScheduler(0.1, first_restart_step=first_restart_step, lr_min=1e-5, warmup=1, m_mul=0.5)
cc = [[aa.on_epoch_begin(ii)] * 50 for ii in range(0, epochs)]
bb = CosineLrScheduler(0.1, first_restart_step=first_restart_step * 5000, lr_min=1e-5, warmup=50, m_mul=0.5)
dd = [bb.on_train_batch_begin(ii) for ii in range(0, epochs * 5000, 100)]
plt.plot(range(0, epochs * 5000, 100), np.ravel(cc))
plt.plot(range(0, epochs * 5000, 100), dd)
```
