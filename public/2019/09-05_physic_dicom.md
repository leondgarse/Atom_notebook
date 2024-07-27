# ___2019 - 09 - 05 Dicom Segmentation___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 09 - 05 Dicom Segmentation___](#2019-09-05-dicom-segmentation)
  - [目录](#目录)
  - [链接](#链接)
  - [Q and A](#q-and-a)
  - [Dicom](#dicom)
  	- [图像读取](#图像读取)
  	- [图像颜色调整](#图像颜色调整)
  	- [计算差值](#计算差值)
  	- [图像分割](#图像分割)
  	- [图像匹配](#图像匹配)
  	- [CaPTK](#captk)
  - [sunny_demmo](#sunnydemmo)
  - [DICOM 数据集](#dicom-数据集)
  	- [dcm to png](#dcm-to-png)
  	- [Read png and mask](#read-png-and-mask)
  	- [3D stack](#3d-stack)
  	- [New dicom structure](#new-dicom-structure)
  - [sitk segmentation](#sitk-segmentation)
  - [Breast segmentation](#breast-segmentation)
  	- [breast seg](#breast-seg)
  	- [breast seg](#breast-seg)
  	- [tumor seg](#tumor-seg)
  - [UNet](#unet)
  	- [模型](#模型)
  	- [测试](#测试)

  <!-- /TOC -->
***

# 链接
  - [图割-最大流最小切割的最直白解读](https://www.jianshu.com/p/beca253fdc9f)
  - [使用Python的scikit-image模块进行图像分割](http://www.zijin.net/news/tech/333026.html)
  - [skimage Module: segmentation](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#module-skimage.segmentation)
  - [skimage Module: restoration](https://scikit-image.org/docs/dev/api/skimage.restoration.html#module-skimage.restoration)
  - [skimage General examples](https://scikit-image.org/docs/stable/auto_examples/index.html)
  - [Segmentation of objects](https://scikit-image.org/docs/stable/auto_examples/index.html#segmentation-of-objects)
  - [W3cubDocs scikit_image](https://docs.w3cub.com/scikit_image/api/skimage.segmentation/#skimage.segmentation.chan_vese)
  - [pydicom](https://github.com/pydicom/pydicom)
  - [SimpleITK Notebooks](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/)
  - [Yonv1943/Unsupervised-Segmentation](https://github.com/Yonv1943/Unsupervised-Segmentation/tree/master)
  - [Automatic and fast segmentation of breast region-of-interest (ROI) and density in MRIs](https://www.sciencedirect.com/science/article/pii/S2405844018327178)
  - [研习U-Net](https://zhuanlan.zhihu.com/p/44958351)
  - [Interpolation (scipy.interpolate)](https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html)
  - [Opencv-Python学习笔记八——图像平滑（滤波）smoothing&blurring](https://www.jianshu.com/p/4ae5e8cef9ae)
  - [开源nnU-Net医学影像分割论文，可自动调参，适应所有数据集](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247490747&idx=1&sn=79be3325bc8a52c4fd5cda27881a4f7c&chksm=ec1ff142db687854f1d29b67baf68c82fac16035171223e570e1acd2b102b61a49e605604899&mpshare=1&scene=1&srcid=0924S7UCw3gx2ZW9E2n02Pro&sharer_sharetime=1569338713076&sharer_shareid=39fcc1218ba6b37488dcb9fb160affa6&pass_ticket=3Ax8ZJVGeWBc5I1gSKiPxa2vXD%2F5Sk3Io1oaZ%2BwQTJ5k1v48%2Fcacrj2chLcnHbtY#rd)
  - [3D U-Net论文解析](http://www.mamicode.com/info-detail-2399938.html)
  - [Github ozan-oktay/Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)
  - [Github wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
  - [Github xmengli999/H-DenseUNet](https://github.com/xmengli999/H-DenseUNet)
***

# Q and A
  - Q: 创建 keras 模型报错
    ```py
    AttributeError: 'Node' object has no attribute 'output_masks'
    ```
    A: 导入的层应全部使用 tensorflow.keras 或者 keras 的
  - Q: plot_model 错误
    ```py
    tf.keras.utils.vis_utils.plot_model() raises TypeError: 'InputLayer' object is not iterable
    ```
    A: 使用 `tf.keras.utils.plot_model` 替换 `tf.keras.utils.vis_utils.plot_model`，或参考 [Fix TypeError when using tf.keras.utils.plot_model](https://github.com/tensorflow/tensorflow/pull/24625)
  - Q: 模型导出报错
    ```py
    Cannot export Keras model TypeError: ('Not JSON Serializable:', b'\n...')
    ```
    A: 使用 `tf.keras.layers.Lambda` 封装模型中的函数
    ```py
    inputs_3 = tf.image.grayscale_to_rgb(inputs)
    --> inputs_3 = tf.keras.layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)

    x = tf.math.divide(x, tf.reduce_max(x))
    --> x = tf.keras.layers.Lambda(lambda x: tf.math.divide(x, tf.reduce_max(x)))(x)

    concat = tf.concat([foo, bar], axis=3)
    --> concat = tf.keras.layers.Concatenate(axis=3)([foo, bar])
    ```
***

# Dicom
## 图像读取
  - **pydicom**
    ```py
    import matplotlib.pyplot as plt
    import pydicom

    aa = pydicom.read_file('./IM287')
    img = aa.pixel_array
    plt.imshow(img, cmap='gray')
    plt.show()
    ```
    ```py
    import pydicom

    aa = pydicom.read_file('./IM287')
    bb = pydicom.read_file('./IM288')
    tt = bb.ImagePositionPatient[2] - aa.ImagePositionPatient[2]
    aa.SliceThickness = tt
    bb.SliceThickness = tt
    image = np.stack([aa.pixel_array, bb.pixel_array])
    ```
## 图像颜色调整
  - **Sigmoid 调整**
    ```py
    adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)
    ```
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization) 优化图像**
    ```py
    def limitedEqualize(img_array, limit = 4.0):
        img_array_list = []
        for img in img_array:
            clahe = cv2.createCLAHE(clipLimit = limit, tileGridSize = (8,8))
            img_array_list.append(clahe.apply(img))
        img_array_limited_equalized = np.array(img_array_list)
        return img_array_limited_equalized
    ```
## 计算差值
  ```py
  import glob2
  from skimage import exposure

  imm = glob2.glob('./*.png')
  imgs = np.array([imread(ii) for ii in imm])
  iee = np.array([exposure.adjust_sigmoid(ii) for ii in imgs])

  plt.imshow(np.vstack([np.hstack(iee[5:]) - np.hstack(iee[:5])]))
  plt.tight_layout()

  plt.imshow(np.vstack([np.hstack(imgs[5:]) - np.hstack(imgs[:5])]))
  plt.tight_layout()

  itt = glob2.glob('./*.tif')
  imt = np.array([gray2rgb(resize(imread(ii), (787, 1263))) * 255 for ii in itt]).astype(uint)
  plt.imshow(np.vstack([np.hstack(iee[:5]), np.hstack(iee[5:]), np.hstack(imt)]))
  plt.tight_layout()

  plt.imshow(np.vstack([np.hstack(iee[:5]), np.hstack(iee[5:]), np.hstack(imt), np.vstack([np.hstack(imgs[5:]) - np.hstack(imgs[:5])])]))
  plt.tight_layout()
  ```
## 图像分割
  - **形态学计算**
    ```py
    from skimage.color import rgb2gray
    from skimage.morphology import opening, binary_dilation
    from skimage.morphology.selem import square

    def pick_highlight(img, opening_square=50, dilation_square=100, dilation_thresh=0.05):
        aa = rgb2gray(img)
        bb = opening(aa, square(50))
        cc = binary_dilation(bb > 0.05, square(100))
        return aa * cc

    def pick_highlight(img, opening_square=50, dilation_square=180):
        aa = rgb2gray(img)
        bb = opening(aa, square(opening_square))
        cc = binary_dilation(bb == bb.max(), square(dilation_square))
        return aa * cc

    ipp = np.hstack([pick_highlight(ii) for ii in iee[5:]])
    ```
  - **LOG 角点检测**
    ```py
    def pick_by_blobs(img, cutoff=0.5, gain=120, min_sigma=100, max_sigma=150, num_sigma=10, threshold=.1):
        itt = exposure.adjust_sigmoid(img, cutoff=cutoff, gain=gain)
        image_gray = rgb2gray(itt)
        blobs_log = blob_log(image_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
        return blobs_log

    def show_imgs_and_blobs(imgs, blobs):
        rows = np.min([len(imgs), len(blobs)])
        fig, axes = plt.subplots(1, rows, figsize=(3 * rows, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        for id, (ii, bbs) in enumerate(zip(imgs, blobs)):
            ax[id].imshow(ii, interpolation='nearest')
            for bb in bbs:
                y, x, r = bb
                c = plt.Circle((x, y), r, color="y", linewidth=2, fill=False)
                ax[id].add_patch(c)
            ax[id].set_axis_off()

    blobs = [pick_by_blobs(ii) for ii in imgs[5:]]
    show_imgs_and_blobs(imgs, blobs)
    plt.tight_layout()
    ```
  - **kmenas 聚类**
    ```py
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3)
    label = km.fit_predict(imgs[-1].reshape(-1, 3)).reshape([787, 1263])
    plt.imshow(np.hstack([rgb2gray(imgs[-1]), label]))
    ```
  - **morphology GAC**
    ```py
    image = rgb2gray(imread('dicom_png/IM256.png'))
    gimage = inverse_gaussian_gradient(opening(image, square(3)))
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-250, 10:-10] = 1

    plt.imshow(image, cmap='gray')
    plt.contour(morphological_geodesic_active_contour(gimage, 50, init_ls, smoothing=1, balloon=-1,threshold=0.69), [0.5], colors='r')
    ```
    ```py
    image = (rgb2gray(imread('dicom_png/IM256.png')) * 255).astype(np.uint8)
    image_enhance = equalize_adapthist(opening(image, square(3)))
    image_enhance = enhance_contrast(opening(image, square(3)), disk(3))
    image_enhance = equalize_adapthist(enhance_contrast(erosion(image, square(3)), disk(3)))
    gimage = inverse_gaussian_gradient(image_enhance)
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-272, 10:-10] = 1

    plt.imshow(image, cmap='gray')
    plt.contour(morphological_geodesic_active_contour(gimage, 50, init_ls, smoothing=4, balloon=-1,threshold=0.69), [0.5], colors='y')
    ```
    ```py
    image = rgb2gray(imread('dicom_png/IM287.png'))
    gimage = inverse_gaussian_gradient(opening(image, square(3)))
    init_ls = np.zeros(image.shape, dtype=np.int8)
    init_ls[10:-270, 10:-10] = 1

    plt.imshow(image, cmap='gray')
    plt.contour(morphological_geodesic_active_contour(gimage, 50, init_ls, smoothing=1, balloon=-1,threshold=0.69), [0.5], colors='r')
    ```
  - **find low edge in middle pixels**
    ```py
    aa = image_enhance[:, 246:266].sum(1)
    (aa[::-1] > 2).argmax()
    ```
## 图像匹配
  ```py
  import numpy as np
  import matplotlib.pyplot as plt
  from skimage import data
  from skimage.feature import match_template

  def match_gray(image_array, token_temp_array):
      result = match_template(image_array, token_temp_array)
      ij = np.unravel_index(np.argmax(result), result.shape)
      x, y = ij[::-1]
      return x, y, result

  def plot_and_match(image_name, token_temp_name):
      image = rgb2gray(imread(image_name))
      token_temp = rgb2gray(imread(token_temp_name))
      x, y, result = match_gray(image, token_temp)

      fig, axes = plt.subplots(1, 3, figsize=(8, 3))
      axes[0].imshow(token_temp, cmap='gray')
      axes[0].set_axis_off()
      axes[0].set_title('template')

      axes[1].imshow(image, cmap='gray')
      axes[1].set_axis_off()
      axes[1].set_title('image')

      ht, wt = token_temp.shape[:2]
      rect = plt.Rectangle((x, y), wt, ht, edgecolor='r', facecolor='none')
      axes[1].add_patch(rect)

      axes[2].imshow(result)
      axes[2].set_axis_off()
      axes[2].set_title('`match_template`\nresult')
      # highlight matched region
      axes[2].autoscale(False)
      axes[2].plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

      plt.show()
      return x, y, result

  plot_and_match('./dicom_png/IM258.png', './Selection_025.png')
  ```
## CaPTK
  - [Github CBICA/CaPTk](https://github.com/CBICA/CaPTk)
  - [Cancer Imaging Phenomics Toolkit (CaPTk)](https://cbica.github.io/CaPTk/Getting_Started.html)
  - [CBICA Image Processing Portal](https://ipp.cbica.upenn.edu/)
  ```sh
  /cbica/home/IPP/wrappers-bin/libra --inputdir /cbica/home/IPP/IPP-users/626088490417991686/Experiments/512271977657680325/mammograms/ --outputdir /cbica/home/IPP/IPP-users/626088490417991686/Experiments/512271977657680325/Results/ --saveintermed 0
  ```
***

# sunny_demmo
  - The denoised result image obtained from Gaussian filter has blurred edges. However, the result from pixelwise adaptive **wiener** filtering technique show that sharp edges are preserved
  - We clustered an image into 10 different colors **(k = 10)** which is sufficient to observe the level of detail of landmarks and their color distribution.
  ```py
  # MATLAB
  H = fspecial('Gaussian', [r, c], sigma);

  # opencv-python
  # cv2.getGaussianKernel(r, sigma)返回一个shape为(r, 1)的np.ndarray, fspecial里核的size的参数是先行后列, 因此:
  H = np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)  # H.shape == (r, c)
  ```
  ```py
  图像滤波函数imfilter函数的应用及其扩展

  import cv2
  import numpy as np
  import matplotlib.pyplot as plt

  img = cv2.imread(‘flower.jpg‘,0) #直接读为灰度图像
  img1 = np.float32(img) #转化数值类型
  kernel = np.ones((5,5),np.float32)/25

  dst = cv2.filter2D(img1,-1,kernel)
  #cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
  #输出图像与输入图像大小相同
  #中间的数为-1，输出数值格式的相同plt.figure()
  plt.subplot(1,2,1),plt.imshow(img1,‘gray‘)#默认彩色，另一种彩色bgr
  plt.subplot(1,2,2),plt.imshow(dst,‘gray‘)
  ```
***

# DICOM 数据集
## dcm to png
  ```py
  idd = glob2.glob('./*/*.dcm')

  for ii in idd:
      if not os.path.exists(ii + '.png'):
          ipp = pydicom.dcmread(ii).pixel_array
          if ipp.shape[0] == 512 or ipp.shape[0] == 256:
              plt.imsave(ii + '.png', ipp, cmap=pylab.cm.bone)
          else:
              ## Concated DCM
              for id, ijj in enumerate(ipp):
                  plt.imsave('{}_{}.png'.format(ii, id), ijj, cmap='gray')
  ```
  ```py
  idd = glob2.glob('*/*/*/*.dcm')

  for ii in idd:
      print(ii)
      # Exclude a specific one
      if not os.path.basename(os.path.dirname(ii)).startswith('999-'):
          dest_name = os.path.join('PNG', ii) + ".png"
          if not os.path.exists(os.path.dirname(dest_name)):
              os.makedirs(os.path.dirname(dest_name), exist_ok=True)
          if not os.path.exists(dest_name):
              ipp = pydicom.dcmread(ii).pixel_array
              plt.imsave(dest_name, ipp, cmap="gray")
  ```
## Read png and mask
  ```py
  aa = glob2.glob('./dicom_png/DCE-7/*.png')
  bb = glob2.glob('./masks/7/*.tif')
  for ss, dd in zip(aa, bb):
      nn = os.path.basename(ss).split('.')[0]
      os.rename(dd, os.path.join(os.path.dirname(dd), nn + '.tif'))
  os.rename('./masks/7', 'masks/DCE-7')

  bb = glob2.glob('./known/masks/*/*.tif')
  for ii in bb:
      im = imread(ii)
      plt.imsave(ii.replace('.tif', '.png'), im, cmap='gray')

  ! rm ./known/masks/*/*.tif
  ```
  ```py
  def plot_breast_seg_multi(image_dir, rows):
      ccs = np.load(os.path.join(image_dir, "breast_border.npy"))
      images = glob2.glob(image_dir + "/*.png")
      masks = [ii.replace("dicom_png", "masks") for ii in images]
      cols = len(ccs) // rows + (len(ccs) % rows != 0)
      fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
      axes_f = axes.flatten()
      for imm, mm, cc, ax in zip(images, masks, ccs, axes_f):
          img = imread(imm)
          mask = (imread(mm) != 0)[:, :, 0]
          ax.imshow(img, cmap='gray')
          ax.contour(cc, [0.5], colors='r')
          ax.contour(mask, [0.5], colors='g')
          ax.set_axis_off()
      fig.tight_layout()
  ```
## 3D stack
  ```py
  aa = glob2.glob('../../datasets/physic_segmentation/known/dicom_png/DCE-11/*.png')
  bb = np.array([imread(ii)[:, :, 0] for ii in aa])
  dd = glob2.glob('../../datasets/physic_segmentation/known/masks/DCE-11/*.png')
  ee = np.array([imread(ii)[:, :, 0] for ii in dd])
  imsave('./goo.png', np.hstack([np.vstack(bb.transpose(2, 0, 1)), np.ones((13312, 5)) * 255, np.vstack(ee.transpose(2, 0, 1))]))
  imsave('./foo.png', np.hstack([np.vstack(bb.transpose(1, 0, 2)), np.ones((13312, 5)) * 255, np.vstack(ee.transpose(1, 0, 2))]))
  ```
## New dicom structure
  - Numeric structure
    - The pre-contrast frame (F1): 0-115
    - First post-contrast frame (F2): 116-231
    - Second post-contrast frame (F3): 232-347
    - Third post-contrast frame (F4): 348-463
    - Forth post-contrast frame (F5): 464-579
    - Fifth post-contrast frame (F6): 580-695
    ```py
    1641321 0113 - 0119, 0465, 0466
    1975297 0003 - 0013, 0066, 0112 - 0116, 0175, 0195 - 0199, 0231, 0232, 0291, 0407, 0510, 0626
    2100296 0126, 0127, 0158-0162, 0373
    2132229 0175  
    ```
  - 标注的mask是根据第三个时间点的数据剪掉第一个时间点的数据影像
    ```py
    import glob2
    from skimage.io import imread, ImageCollection, imsave

    aa = glob2.glob('./*.tif')
    bb = sorted(aa)
    tt = np.array([imread(ii) for ii in bb])
    tt_1 = tt[0:115]
    tt_3 = tt[232:347]
    dd = tt_3 - tt_1
    plt.imshow(np.vstack([np.hstack(ii) for ii in np.vsplit(dd, 5)] + [np.hstack(ii) for ii in np.vsplit(tt_1, 5)] + [np.hstack(ii) for ii in np.vsplit(tt_3, 5)]))
    plt.axis('off')
    plt.tight_layout()
    ```
***

# sitk segmentation
  ```py
  img = imread('cthead1.png')
  img = sitk.GetImageFromArray(rgb2gray(img))
  feature_img = sitk.GradientMagnitude(img)
  plt.imshow(sitk.GetArrayFromImage(feature_img), cmap='gray')

  ws_img = sitk.MorphologicalWatershed(feature_img, level=0, markWatershedLine=True, fullyConnected=False)
  plt.imshow(sitk.GetArrayFromImage(sitk.LabelToRGB(ws_img)))

  min_img = sitk.RegionalMinima(feature_img, backgroundValue=0, foregroundValue=1.0, fullyConnected=False, flatIsMinima=True)
  marker_img = sitk.ConnectedComponent(min_img)
  plt.imshow(sitk.GetArrayFromImage(sitk.LabelToRGB(marker_img)))

  ws = sitk.MorphologicalWatershedFromMarkers(feature_img, marker_img, markWatershedLine=True, fullyConnected=False)
  plt.imshow(sitk.GetArrayFromImage(sitk.LabelToRGB(ws)))

  pt = [60,60]
  idx = img.TransformPhysicalPointToIndex(pt)
  marker_img *= 0
  marker_img[0,0] = 1
  marker_img[idx] = 2
  ws = sitk.MorphologicalWatershedFromMarkers(feature_img, marker_img, markWatershedLine=True, fullyConnected=False)
  plt.imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(img, ws, opacity=.2)))
  ```
***

# Breast segmentation
## breast seg
  ```py
  import os
  import numpy as np
  import matplotlib.pyplot as plt
  from skimage.color import rgb2gray
  from skimage.exposure import equalize_adapthist, adjust_sigmoid
  from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
  from skimage.morphology import erosion, closing, opening, binary_closing, binary_erosion, binary_opening
  from skimage.morphology.selem import disk, square
  from skimage.filters import scharr_h
  from skimage.filters.rank import enhance_contrast
  from skimage.io import imread
  from scipy.ndimage import binary_fill_holes
  import pydicom

  def breast_seg(image_array):
      # Read image as uint8, value scope in [1, 256]
      if image_array.max() > 255:
          image_gray = rgb2gray(image_array) / image_array.max()
      else:
          image_gray = rgb2gray(image_array)
      image = (image_gray * 255).astype(np.uint8)
      # Enhance image by erosion, enhance contrast and equalize hist
      image_enhance = equalize_adapthist(enhance_contrast(erosion(image, disk(3)), disk(3)))
      # Detect the position of breast edge in middle area
      middle_pix = image.shape[1] // 2
      low_edge_y = ((image_enhance[:, middle_pix-20:middle_pix+20] > 0.1).sum(1)[::-1] > 10).argmax()
      print("low_edge_y = %d" % low_edge_y)

      # Inverse Gaussian gradient for morphology snakes
      imm = image_enhance.copy()
      for ii in range(2):
          for irr in np.arange(0, 512, 2):
              for icc in np.arange(0, 512):
                  if imm[irr, icc] > imm[irr+1, icc]:
                      imm[irr+1, icc] = imm[irr, icc]

          for irr in np.arange(1, 510, 2):
              for icc in np.arange(0, 512):
                  if imm[irr, icc] > imm[irr+1, icc]:
                      imm[irr+1, icc] = imm[irr, icc]
          imm = (scharr_h(imm) < 0) * imm

      for ii in range(4):
          for irr in np.arange(0, 512 - low_edge_y - 5, 2):
              for icc in np.arange(0, 512):
                  if imm[irr, icc] < imm[irr + 1, icc]:
                      imm[irr, icc] = imm[irr + 1, icc]

          for irr in np.arange(1, 512 - low_edge_y - 5, 2):
              for icc in np.arange(0, 512):
                  if imm[irr, icc] < imm[irr + 1, icc]:
                      imm[irr, icc] = imm[irr + 1, icc]

      for ii in range(2):
          for irr in np.arange(0, 512, 2):
              for icc in np.arange(0, 512):
                  if imm[irr, icc] > imm[irr+1, icc]:
                      imm[irr, icc] = imm[irr+1, icc]

          for irr in np.arange(1, 510, 2):
              for icc in np.arange(0, 512):
                  if imm[irr, icc] > imm[irr+1, icc]:
                      imm[irr, icc] = imm[irr+1, icc]
      gimage = inverse_gaussian_gradient(adjust_sigmoid(imm, cutoff=0.2), alpha=100.0, sigma=5.0)

      # Initial mask for morphology snakes of the thorax area
      init_ls = np.zeros(image.shape, dtype=np.int8)
      init_ls[10:-low_edge_y, 10:-10] = 1
      # The first part is the thorax area
      body_cc = morphological_geodesic_active_contour(gimage, 30, init_ls, smoothing=8, balloon=-1, threshold=0.89)
      body_cc = binary_erosion(body_cc, disk(5))
      # body_cc = 1 - binary_closing(1 - body_cc, disk(25))
      # return body_cc

      # Detect the position of breast edge in height
      low_breast_edge = ((image_enhance > 0.1).sum(1) > 10)[::-1].argmax() - 10
      # low_breast_edge = ((image_enhance > 0.2).sum(1) > 5)[::-1].argmax() - 5
      print("low_breast_edge = %d" % low_breast_edge)

      # Initial mask for morphology snakes of the breast area
      init_ls_2 = np.zeros(image.shape, dtype=np.int8)
      keep_body_border = (image.shape[1] - low_edge_y) // 5 * 4
      init_ls_2[keep_body_border:-low_breast_edge, 10:-10] = 1
      init_ls_2 *= (1 - body_cc)
      init_ls_2 = binary_closing(init_ls_2, disk(25))

      # Breast area
      image_enhance_2 = adjust_sigmoid(enhance_contrast(closing(opening(image, disk(3)), disk(13)), square(7)), cutoff=0.1)
      breast_cc = binary_opening(binary_fill_holes(image_enhance_2 > (image_enhance_2[-low_breast_edge+5:].max() + 12)), disk(10))
      final_cc = init_ls_2 * breast_cc

      # Move the breast border 5 pixels up to exclude the border line
      up_border = final_cc[:-low_edge_y, :]
      low_border = binary_erosion(final_cc[-low_edge_y:, :], disk(8))
      return np.vstack([up_border, low_border])
      # return np.vstack([breast_cc[:-low_edge_y-5, :], breast_cc[-low_edge_y:, :], np.zeros([5, 512])])

  def plot_breast_seg_multi(images, rows=0, save_png=None, save_result_dir=None):
      if len(images) == 0:
          print("Empty images")
          return np.array([])

      if rows == 0:
          rows = int(np.ceil(len(images) / 6))
      cols = int(np.ceil(len(images) / rows))
      print("Total images = %d, rows = %d, cols = %d" % (len(images), rows, cols))

      fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
      if len(images) != 1:
          axes_f = axes.flatten()
      else:
          axes_f = [axes]
      ccs = []
      for imm, ax in zip(images, axes_f):
          if imm.endswith('.png'):
              img = imread(imm)
          else:
              img = pydicom.read_file(imm).pixel_array
          cc = breast_seg(img)
          ax.imshow(img, cmap='gray')
          ax.contour(cc, [0.5], colors='r')
          # ax.imshow(img[:, :, 0] * cc, cmap='gray')
          ax.set_axis_off()
          ccs.append(cc)
          if save_result_dir:
              plt.imsave(os.path.join(save_result_dir, os.path.basename(imm)), img[:, :, 0] * cc, cmap='gray')
      fig.tight_layout()
      plt.show()

      if save_png:
          fig.savefig(save_png)
      return np.array(ccs)

  def plot_and_save_dir(dir):
      for ii in os.listdir(dir):
          dd = os.path.join(dir, ii)
          if os.path.isdir(dd):
              print("directory = %s" % dd)
              aa = glob2.glob(os.path.join(dd, '*'))
              save_png = os.path.join(dir, ii + '.png')
              save_result_dir = os.path.join(dir, "seg_result", ii)
              if not os.path.exists(save_result_dir):
                  os.makedirs(save_result_dir)
              plot_breast_seg_multi(aa, save_png=save_png, save_result_dir=save_result_dir)

  if __name__ == "__main__":
      import argparse
      import glob2

      parser = argparse.ArgumentParser()
      parser.add_argument('-i', '--image', type=str, required=True, help="Dicom Image path to parse")
      parser.add_argument('-r', '--rows', type=int, default=0, help="Rows of images to display")
      args = parser.parse_args()
      if os.path.isdir(args.image):
          plot_and_save_dir(args.image)
      else:
          aa = glob2.glob(args.image)
          plot_breast_seg_multi(aa, args.rows)

    import glob2
    aa = glob2.glob('./*.png')
    plot_breast_seg_multi(aa, 4)

    ccs = np.array([breast_seg(imread(ii)) for ii in aa])
    images = np.array([imread(ii) for ii in aa])
    np.savez('breast_border', breast_borders=ccs, images=images)
  ```
## breast seg
  ```py
  import os
  import numpy as np
  import matplotlib.pyplot as plt
  from skimage.color import rgb2gray
  from skimage.exposure import equalize_adapthist, adjust_sigmoid
  from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
  from skimage.morphology import erosion, closing, opening, binary_closing, binary_erosion, binary_opening
  from skimage.morphology.selem import disk, square
  from skimage.filters import scharr_h
  from skimage.filters.rank import enhance_contrast
  from skimage.io import imread
  from scipy.ndimage import binary_fill_holes
  import pydicom

  bright_up = lambda ii: np.max([ii, np.vstack([ii[1:], np.zeros_like(ii[:1])])], 0)
  dim_up = lambda ii: np.min([ii, np.vstack([ii[1:], np.zeros_like(ii[:1])])], 0)
  bright_down = lambda ii: np.max([ii, np.vstack([np.zeros_like(ii[:1]), ii[:-1]])], 0)
  dim_down = lambda ii: np.min([ii, np.vstack([np.zeros_like(ii[:1]), ii[:-1]])], 0)

  def breast_seg(image_array):
      # Read image as uint8, value scope in [1, 256]
      if image_array.max() > 255:
          image_gray = rgb2gray(image_array) / image_array.max()
      else:
          image_gray = rgb2gray(image_array)
      image = (image_gray * 255).astype(np.uint8)
      # Enhance image by erosion, enhance contrast and equalize hist
      image_enhance = equalize_adapthist(enhance_contrast(erosion(image, disk(3)), disk(3)))
      # Detect the position of breast edge in middle area
      middle_pix = image.shape[1] // 2
      low_edge_y = ((image_enhance[:, middle_pix-20:middle_pix+20] > 0.1).sum(1)[::-1] > 10).argmax()
      print("low_edge_y = %d" % low_edge_y)

      # Inverse Gaussian gradient for morphology snakes
      imm = image_enhance.copy()
      for ii in range(3):
          imm = bright_down(imm)
          imm = bright_down(imm)
          imm = (scharr_h(imm) < 0) * imm

      for ii in range(3):
          imm = np.vstack([bright_up(imm[:-low_edge_y]), imm[-low_edge_y:]])
          imm = np.vstack([bright_up(imm[:-low_edge_y]), imm[-low_edge_y:]])
          imm = np.vstack([bright_up(imm[:-low_edge_y]), imm[-low_edge_y:]])
          imm = dim_up(imm)
          imm = dim_up(imm)
      gimage = inverse_gaussian_gradient(adjust_sigmoid(imm, cutoff=0.2), alpha=100.0, sigma=5.0)

      # Initial mask for morphology snakes of the thorax area
      init_ls = np.zeros(image.shape, dtype=np.int8)
      init_ls[10:-low_edge_y, 10:-10] = 1
      # The first part is the thorax area
      body_cc = morphological_geodesic_active_contour(gimage, 30, init_ls, smoothing=8, balloon=-1, threshold=0.89)
      body_cc = binary_erosion(body_cc, disk(5))
      # body_cc = 1 - binary_closing(1 - body_cc, disk(25))
      # return body_cc

      # Detect the position of breast edge in height
      low_breast_edge = ((image_enhance > 0.1).sum(1) > 10)[::-1].argmax() - 10
      # low_breast_edge = ((image_enhance > 0.2).sum(1) > 5)[::-1].argmax() - 5
      print("low_breast_edge = %d" % low_breast_edge)

      # Initial mask for morphology snakes of the breast area
      init_ls_2 = np.zeros(image.shape, dtype=np.int8)
      keep_body_border = (image.shape[1] - low_edge_y) // 5 * 4
      init_ls_2[keep_body_border:-low_breast_edge, 10:-10] = 1
      init_ls_2 *= (1 - body_cc)
      init_ls_2 = binary_closing(init_ls_2, disk(25))

      # Breast area
      image_enhance_2 = adjust_sigmoid(enhance_contrast(closing(opening(image, disk(3)), disk(13)), square(7)), cutoff=0.1)
      breast_cc = binary_opening(binary_fill_holes(image_enhance_2 > (image_enhance_2[-low_breast_edge+5:].max() + 12)), disk(10))
      final_cc = init_ls_2 * breast_cc

      # Move the breast border 5 pixels up to exclude the border line
      up_border = final_cc[:-low_edge_y, :]
      low_border = binary_erosion(final_cc[-low_edge_y:, :], disk(8))
      return np.vstack([up_border, low_border])
      # return np.vstack([breast_cc[:-low_edge_y-5, :], breast_cc[-low_edge_y:, :], np.zeros([5, 512])])

  def plot_breast_seg_multi(images, rows=0, save_png=None, save_result_dir=None):
      if len(images) == 0:
          print("Empty images")
          return np.array([])

      if rows == 0:
          rows = int(np.ceil(len(images) / 6))
      cols = int(np.ceil(len(images) / rows))
      print("Total images = %d, rows = %d, cols = %d" % (len(images), rows, cols))

      fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
      if len(images) != 1:
          axes_f = axes.flatten()
      else:
          axes_f = [axes]
      ccs = []
      for imm, ax in zip(images, axes_f):
          if imm.endswith('.png'):
              img = imread(imm)
          else:
              img = pydicom.read_file(imm).pixel_array
          cc = breast_seg(img)
          ax.imshow(img, cmap='gray')
          ax.contour(cc, [0.5], colors='r')
          # ax.imshow(img[:, :, 0] * cc, cmap='gray')
          ax.set_axis_off()
          ccs.append(cc)
          if save_result_dir:
              plt.imsave(os.path.join(save_result_dir, os.path.basename(imm)), img[:, :, 0] * cc, cmap='gray')
      fig.tight_layout()
      plt.show()

      if save_png:
          fig.savefig(save_png)
      return np.array(ccs)

  def plot_and_save_dir(dir):
      for ii in os.listdir(dir):
          dd = os.path.join(dir, ii)
          if os.path.isdir(dd):
              print("directory = %s" % dd)
              aa = glob2.glob(os.path.join(dd, '*'))
              save_png = os.path.join(dir, ii + '.png')
              save_result_dir = os.path.join(dir, "seg_result", ii)
              if not os.path.exists(save_result_dir):
                  os.makedirs(save_result_dir)
              plot_breast_seg_multi(aa, save_png=save_png, save_result_dir=save_result_dir)

  if __name__ == "__main__":
      import argparse
      import glob2

      parser = argparse.ArgumentParser()
      parser.add_argument('-i', '--image', type=str, required=True, help="Dicom Image path to parse")
      parser.add_argument('-r', '--rows', type=int, default=0, help="Rows of images to display")
      args = parser.parse_args()
      if os.path.isdir(args.image):
          plot_and_save_dir(args.image)
      else:
          aa = glob2.glob(args.image)
          plot_breast_seg_multi(aa, args.rows)

    import glob2
    aa = glob2.glob('./*.png')
    plot_breast_seg_multi(aa, 4)

    ccs = np.array([breast_seg(imread(ii)) for ii in aa])
    images = np.array([imread(ii) for ii in aa])
    np.savez('breast_border', breast_borders=ccs, images=images)
  ```
## tumor seg
  ```py
  from skimage.morphology import binary_dilation
  from skimage.exposure import adjust_sigmoid

  def extract_mask_by_breast_border(image_array, breast_border):
      breast_pick = rgb2gray(image_array) * breast_border
      large_bright_area = binary_dilation(erosion(breast_pick, square(3)) > 0.3, square(100))
      mask = binary_dilation(adjust_sigmoid(breast_pick * large_bright_area, cutoff=0.6) > 0.1, disk(1))
      return mask

  def plot_image_and_masks(images, masks, rows, cols):
      fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
      axes_f = axes.flatten()
      for imm, mm, ax in zip(images, masks, axes_f):
          img = rgb2gray(imm)
          ax.imshow(np.hstack([img, mm]), cmap='gray')
          ax.set_axis_off()
      fig.tight_layout()

  tt = np.load('breast_border.npz')
  ccs, images = tt['breast_borders'], tt['images']
  mms = [extract_mask_by_breast_border(ii, cc) for cc, ii in zip(ccs, images)]
  plot_image_and_masks(images, mms, 4, 6)
  ```
***

# UNet
## 模型
  - [Colab physic_image_segmentation.ipynb](https://colab.research.google.com/drive/1IGims1mk8jI30N6jqj50KW8JzbE6QeA_#scrollTo=5PbSBm9uynpC&forceEdit=true&offline=true&sandboxMode=true)
  ```py
  # UNet
  Epoch 00012: val_dice_loss improved from 0.14963 to 0.14088, saving model to ./training_checkpoints/weights.hdf5
  25/25 [==============================] - 11s 451ms/step - loss: 0.1946 - dice_loss: 0.1896 - val_loss: 0.1447 - val_dice_loss: 0.1409

  # Resnet50 based UNet++ Realization 50 + 45
  Epoch 00045: val_dice_loss improved from 0.14202 to 0.13994, saving model to ./training_checkpoints/weights.hdf5
  25/25 [==============================] - 20s 817ms/step - loss: 0.1602 - dice_loss: 0.1561 - val_loss: 0.1438 - val_dice_loss: 0.1399
  ```
## 测试
  ```py
  #!/usr/bin/env python3

  import tensorflow as tf
  import os
  import glob2
  from skimage.io import imread, imsave
  import numpy as np

  def tumor_segmentation(source_image, output_path, model_path):
      graph = tf.Graph()
      with graph.as_default():
          with graph.device("cpu"):
              config = tf.ConfigProto(
                  gpu_options = tf.GPUOptions(allow_growth=True),
                  allow_soft_placement=True,
                  intra_op_parallelism_threads=4,
                  inter_op_parallelism_threads=4)
              sess = tf.Session(config=config)
              with sess.as_default():
                  meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], model_path)
                  input_x = sess.graph.get_tensor_by_name("input_3:0")
                  outputs = sess.graph.get_tensor_by_name("conv2d_68/Sigmoid:0")

      aa = glob2.glob(source_image)
      pred = np.array([sess.run(outputs, {input_x: [imread(ii)[:, :, :3] / 255]})[0, :, :, 0] for ii in aa])
      for ii, pp in zip(aa, pred):
          dest = os.path.join(output_path, os.path.basename(ii))
          if not os.path.exists(os.path.dirname(dest)):
              os.makedirs(os.path.dirname(dest))
          imsave(dest, pp)

  if __name__ == "__main__":
      import argparse
      parser = argparse.ArgumentParser()
      parser.add_argument("-s", "--source_image", type=str, required=True, help="Source image path.")
      parser.add_argument("-d", "--output_path", type=str, required=True, help="Output path to save prediction.")
      parser.add_argument("-m", "--model_path", type=str, default="./1", help="Saved model path.")
      args = parser.parse_args()
      tumor_segmentation(args.source_image, args.output_path, args.model_path)
  ```
***
