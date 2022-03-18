# ___2021 - 11 - 05 YOLOV5 Face___
***
# TOC
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2021 - 11 - 05 YOLOV5 Face___](#2021-11-05-yolov5-face)
  - [TOC](#toc)
    - [YOLOV5 Face and convert tests](#yolov5-face-and-convert-tests)
    - [Tflite Convert](#tflite-convert)
    - [Effv2B0 model to tflite](#effv2b0-model-to-tflite)
    - [NV21 to RGB](#nv21-to-rgb)
    - [Read bitmap](#read-bitmap)

  <!-- /TOC -->
***

## YOLOV5 Face and convert tests
  - [Github ultralytics/yolov5](https://github.com/ultralytics/yolov5.git)
  - [Github deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face.git)
  - **Test**
    ```py
    import torch
    from models.experimental import attempt_load
    model = attempt_load('./weights/yolov5s-face.pt', map_location=torch.device("cpu"))
    out = model(torch.ones([1, 3, 640, 640]))
    out[0].shape
    ```
  - **Convert to keras h5**
    ```sh
    CUDA_VISIBLE_DEVICES='-1' python models/tf_keras.py --weights yolov5s.pt
    CUDA_VISIBLE_DEVICES='-1' python models/tf_keras.py --weights ../yolov5-face/weights/yolov5s-face.pt --imgsz 640 384
    ```
    ```sh
    CUDA_VISIBLE_DEVICES='-1' python detect_face.py --weights weights/yolov5s-face.pt --image data/images/bus.jpg && eog result.jpg
    ```
  - **ncnn** [Github FeiGeChuanShu/ncnn_Android_face](https://github.com/FeiGeChuanShu/ncnn_Android_face)
    ```sh
    adb push ncnn-android-yolov5_face/app/src/main/assets/* /data/local/tmp/benchmark/
    adb push ncnn-android-scrfd-master/app/src/main/assets/scrfd_* /data/local/tmp/benchmark/

    adb shell 'cd /data/local/tmp/benchmark; LD_LIBRARY_PATH=.. ./benchncnn'
    #         yolov5n  min =  196.79  max =  201.00  avg =  199.21
    #     yolov5n-0.5  min =   96.08  max =   99.87  avg =   97.52
    #   scrfd_1g-opt2  min =  116.02  max =  128.08  avg =  120.76
    # scrfd_500m-opt2  min =   69.49  max =   71.69  avg =   70.34
    ```
    **rk3288, ARM32, 640x384**
    ```sh
    adb shell 'cd /data/local/tmp/benchmark; LD_LIBRARY_PATH=.. ./benchncnn'
    #         yolov5n  min = 2578.02  max = 3152.46  avg = 2871.29
    #     yolov5n-0.5  min = 1052.76  max = 1577.13  avg = 1393.49
    #   scrfd_1g-opt2  min =  985.31  max = 1576.60  avg = 1171.01
    # scrfd_500m-opt2  min =  531.48  max =  883.14  avg =  689.63

    adb shell 'cd /data/local/tmp/benchmark; LD_LIBRARY_PATH=.. ./benchncnn 4 1'
    #         yolov5n  min = 1738.41  max = 2040.99  avg = 1892.60
    #     yolov5n-0.5  min =  505.12  max =  615.21  avg =  557.84
    #   scrfd_1g-opt2  min =  756.70  max = 1089.44  avg =  891.28
    # scrfd_500m-opt2  min =  408.14  max =  524.41  avg =  440.87

    adb shell 'cd /data/local/tmp/benchmark; LD_LIBRARY_PATH=.. ./benchncnn 4 2'
    #         yolov5n  min = 1662.97  max = 2468.14  avg = 1996.67
    #     yolov5n-0.5  min =  651.80  max =  829.82  avg =  750.46
    #   scrfd_1g-opt2  min =  292.47  max =  299.20  avg =  295.30
    # scrfd_500m-opt2  min =  148.88  max =  149.20  avg =  149.04
    ```
    **rk3288, ARM32, 320x192**
    ```sh
    adb shell 'cd /data/local/tmp/benchmark; LD_LIBRARY_PATH=.. ./benchncnn 4 4'
    #         yolov5n  min =  112.15  max =  181.31  avg =  136.74
    #     yolov5n-0.5  min =   37.07  max =   62.95  avg =   43.65
    #   scrfd_1g-opt2  min =   47.94  max =   82.00  avg =   63.76
    # scrfd_500m-opt2  min =   25.27  max =   44.65  avg =   32.72
    ```
  - **tflite**
    ```sh
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/yolov5n-face.tflite --num_threads=4 --use_xnnpack=true
    # count=50 first=236744 curr=228642 min=221046 max=236985 avg=226324 std=3807
    # Inference timings in us: Init: 90791, First inference: 277454, Warmup (avg): 251886, Inference (avg): 226324
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/yolov5n-face.tflite --num_threads=8 --use_xnnpack=true
    # count=50 first=187881 curr=177890 min=172574 max=273037 avg=179291 std=14146
    # Inference timings in us: Init: 97564, First inference: 267559, Warmup (avg): 206443, Inference (avg): 179291

    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/yolov5n-0.5.tflite --num_threads=4 --use_xnnpack=true
    # count=50 first=92541 curr=91487 min=90599 max=99438 avg=93417.7 std=1839
    # Inference timings in us: Init: 38974, First inference: 127805, Warmup (avg): 99338.4, Inference (avg): 93417.7
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/yolov5n-0.5.tflite --num_threads=8 --use_xnnpack=true
    # count=50 first=79314 curr=79861 min=76824 max=85946 avg=79621.6 std=1907
    # Inference timings in us: Init: 46040, First inference: 156502, Warmup (avg): 130180, Inference (avg): 79621.6
    ```
  - **tflite TF15 (640, 384)**
    ```sh
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/yolov5n-0.5_tf15.tflite --num_threads=4 --use_xnnpack=true
    # count=50 first=86489 curr=87436 min=85356 max=92507 avg=87459.7 std=998
    # Inference timings in us: Init: 38842, First inference: 118425, Warmup (avg): 92578, Inference (avg): 87459.7
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/yolov5n-0.5_tf15.tflite --num_threads=8 --use_xnnpack=true
    # count=50 first=73698 curr=70552 min=68299 max=81571 avg=70767.2 std=2227
    # Inference timings in us: Init: 40680, First inference: 141210, Warmup (avg): 86868.7, Inference (avg): 70767.2
    ```
  - **tflite TF15 with decode**

    | model        | input    | model size (MB) | num_threads 4 (ms) | num_threads 8 (ms) |
    | ------------ | -------- | --------------- | ------------------ | ------------------ |
    | yolov5n-face | 640, 384 | 3.67216         | 265.224            | 221.162            |
    | yolov5n-face | 480, 288 | 3.61924         | 144.718            | 121.121            |
    | yolov5n-face | 320, 192 | 3.58144         | 61.383             | 54.496             |
    | yolov5n-0.5  | 640, 384 | 1.12552         | 115.085            | 104.037            |
    | yolov5n-0.5  | 480, 288 | 1.0726          | 61.272             | 55.854             |
    | yolov5n-0.5  | 320, 192 | 1.0348          | 25.319             | 24.410             |
  - **rk3288, ARM32**

  | model        | input    | model size (MB) | num_threads 4 (ms) | num_threads 8 (ms) |
  | ------------ | -------- | --------------- | ------------------ | ------------------ |
  | yolov5n-face | 640, 384 | 3.67216         |                    |                    |
  | yolov5n-face | 480, 288 | 3.61924         |                    |                    |
  | yolov5n-face | 320, 192 | 3.58144         |                    |                    |
  | yolov5n-0.5  | 640, 384 | 1.12552         |                    |                    |
  | yolov5n-0.5  | 480, 288 | 1.0726          |                    |                    |
  | yolov5n-0.5  | 320, 192 | 1.0348          | 38208.6            |                    |
## Tflite Convert
  ```sh
  # https://github.com.cnpmjs.org/ultralytics/yolov5.git
  CUDA_VISIBLE_DEVICES='-1' python models/tf_keras.py --weights ../yolov5-face/weights/yolov5n-0.5.pt --imgsz 320 192
  ```
  ```py
  tf.__version__
  # '2.2.0'
  from keras_cv_attention_models import model_surgery

  model_name = 'yolov5n-0.5'

  mm = keras.models.load_model('weights/' + model_name + '.h5')
  bb = model_surgery.convert_to_fused_conv_bn_model(mm)

  inputs = keras.layers.Input(bb.input_shape[1:], dtype=tf.uint8)
  nn = tf.cast(inputs, 'float32')
  bb = keras.models.Model(inputs, bb(nn))

  converter = tf.lite.TFLiteConverter.from_keras_model(bb)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  open(model_name + '.tflite', 'wb').write(converter.convert())

  bb.save_weights("weights_only.h5")
  json_config = bb.to_json()
  with open('model_config.json', 'w') as json_file:
      json_file.write(json_config)

  !sed -i 's/"ragged": false, //' model_config.json
  !sed -i 's/"class_name": "Functional"/"class_name": "Model"/g' model_config.json
  !sed -i 's/"groups": 1, //g' model_config.json
  ```
  ```py
  tf.__version__
  # '1.15.0'
  from tensorflow.keras.utils import CustomObjectScope
  def swish(inputs):
      return (tf.sigmoid(inputs) * inputs)
  model_name = 'yolov5n-0.5_tf15'

  with open('model_config.json') as json_file:
      json_config = json_file.read()
  with CustomObjectScope({'swish': swish}):
      new_model = keras.models.model_from_json(json_config)
  new_model.load_weights('weights_only.h5')
  # inputs = keras.layers.Input([256, 160, 3])
  # bb = keras.models.Model(inputs, new_model(inputs))
  # new_model = bb
  new_model.save(model_name + '.h5')

  ''' Convert to TFLite float16 model '''
  converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5')
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  open(model_name + '.tflite', 'wb').write(converter.convert())

  !adb push {model_name}.tflite /data/local/tmp
  !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=4 --use_xnnpack=true
  ```
  ```py
  from skimage.transform import resize
  imm = resize(imread('aa.jpg'), (320, 192))
  inputs = np.expand_dims(imm, 0).astype('float32') * 255

  interpreter = tf.lite.Interpreter(model_path='yolov5n-0.5_tf15.tflite')
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  interpreter.allocate_tensors()
  interpreter.set_tensor(input_index, inputs)
  interpreter.invoke()
  pred = interpreter.get_tensor(output_index)

  bbs, pps, ccs = yolo_nms(pred, iou_threshold=0.35, score_threshold=0.25)
  show_result(imm, bbs, ccs, pps)
  ```
  ```py
  mm = keras.models.load_model(os.path.expanduser('~/workspace/samba/yolov5-face/weights/yolov5n-0.5.h5'))
  imm = imread('TestYUV420sp2RGBA_half.jpg')
  imm_input = tf.expand_dims(tf.cast(tf.image.resize(imm, (320, 192)), 'uint8'), 0)
  pred = mm(imm_input)
  bbs, pps, ccs = yolo_nms(pred)
  print(f"{bbs.numpy() = }\n{pps.numpy() = }\n{ccs.numpy() = }")
  # bbs.numpy() = array([[123.80841, 126.14965, 187.32587, 198.51044]], dtype=float32)
  # pps.numpy() = array([[138.20782, 156.45053, 162.89815, 148.9878 , 153.21767, 166.42749,
  #         148.20929, 179.65605, 169.41959, 173.15474]], dtype=float32)
  # ccs.numpy() = array([0.8936907], dtype=float32)
  ```
  ```py
  import data
  ds, steps_per_epoch = data.prepare_dataset('/datasets/ms1m-retinaface-t1-cleaned_112x112_folders/', batch_size=1)
  def representative_data_gen():
      for ii in ds.take(1000):
          yield [ii[0]]
  ```
## Effv2B0 model to tflite
  ```py
  import models
  import keras_efficientnet_v2
  basic_model = keras_efficientnet_v2.EfficientNetV2('b0', input_shape=(112, 112, 3), num_classes=0)
  basic_model = models.buildin_models(basic_model, dropout=0, emb_shape=512, output_layer='GDC', bn_epsilon=1e-4, bn_momentum=0.9, scale=True, use_bias=False)
  basic_model.load_weights('checkpoints/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_basic_agedb_30_epoch_17_0.977333.h5')

  from keras_cv_attention_models import model_surgery
  bb = model_surgery.convert_to_fused_conv_bn_model(basic_model)
  cc = keras.models.Model(bb.inputs[0], bb.outputs[0] / tf.sqrt(tf.reduce_sum(bb.outputs[0] ** 2, axis=-1)))

  inputs = keras.layers.Input([112, 112, 3], dtype='uint8')
  nn = (tf.cast(inputs, 'float32') - 127.5) * 0.0078125
  dd = keras.models.Model(inputs, cc(nn))

  model_name = 'effv2b0_pre_uint8_input_norm_output'
  converter = tf.lite.TFLiteConverter.from_keras_model(dd)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  open(model_name + '.tflite', 'wb').write(converter.convert())
  ```
  ```py
  class Tflite_interp:
      def __init__(self, model_path):
          self.interpreter = tf.lite.Interpreter(model_path=model_path)
          self.input_index = self.interpreter.get_input_details()[0]["index"]
          self.output_index = self.interpreter.get_output_details()[0]["index"]
          self.interpreter.allocate_tensors()

      def __call__(self, inputs):
          preds = []
          for ii in inputs:
              ii = ii * 128.0 + 127.5
              ii = tf.cast(tf.expand_dims(ii, 0), "uint8")
              self.interpreter.set_tensor(self.input_index, ii)
              self.interpreter.invoke()
              preds.append(self.interpreter.get_tensor(self.output_index)[0])
          return np.array(preds)
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
## Read bitmap
  ```py
  with open('bitmap_1280_800_4.txt', 'r') as ff:
      aa = ff.read()
  bb = np.array([ubyte(ii) for ii in aa[1:-1].split(', ')])
  print(bb.min(), bb.max())
  plt.imshow(bb.reshape([1280, 800, 4]))
  ```
***
