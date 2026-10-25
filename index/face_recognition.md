# Face Recognition & Anti-Spoofing

Face detection, InsightFace training in Keras, face quality, presentation-attack detection.

Back to [all indexes](README.md).

## Recognition & training

- **[Face Recognition](../public/2018/04-16_face_recognition.md)** (2018-04-16) — Early face recognition — OpenCV video, Baidu AIP face APIs, face_recognition library, RTSP camera streams.
  - [Opencv 视频处理](../public/2018/04-16_face_recognition.md#Opencv%20视频处理) · [百度人脸识别 Aip](../public/2018/04-16_face_recognition.md#百度人脸识别%20Aip) · [Face Recognition](../public/2018/04-16_face_recognition.md#Face%20Recognition) · [百度人脸识别 Aip 检测摄像头图像](../public/2018/04-16_face_recognition.md#百度人脸识别%20Aip%20检测摄像头图像) · [OpenCV 之 网络摄像头](../public/2018/04-16_face_recognition.md#OpenCV%20之%20网络摄像头) · [活体检测 Facial Presentation Attack Detection](../public/2018/04-16_face_recognition.md#活体检测%20Facial%20Presentation%20Attack%20Detection)
- **[Keras Insightface](../public/2019/11-18_keras_insightface.md)** (2019-11-18) — InsightFace with Keras — MXNet model usage, accuracy comparisons, face recognition losses, data pipelines.
  - [Insightface MXNet 模型使用](../public/2019/11-18_keras_insightface.md#Insightface%20MXNet%20模型使用) · [Highest Accuracy](../public/2019/11-18_keras_insightface.md#Highest%20Accuracy) · [人脸识别损失函数](../public/2019/11-18_keras_insightface.md#人脸识别损失函数) · [ImageDataGenerator](../public/2019/11-18_keras_insightface.md#imagedatagenerator) · [训练模型拟合 embeddings](../public/2019/11-18_keras_insightface.md#训练模型拟合%20embeddings) · [Keras Insightface](../public/2019/11-18_keras_insightface.md#Keras%20Insightface) · [nmslib dot svm dist calculation comparing](../public/2019/11-18_keras_insightface.md#nmslib%20dot%20svm%20dist%20calculation%20comparing) · [Docker 封装](../public/2019/11-18_keras_insightface.md#Docker%20封装)
- **[Keras Insightface Training](../public/2020/12-01_Insightface_training.md)** (2020-12-01) — Keras InsightFace training experiments — test utilities, log decoding, backbones, losses, training results.
  - [Test functions](../public/2020/12-01_Insightface_training.md#Test%20functions) · [人脸识别常用评价指标](../public/2020/12-01_Insightface_training.md#人脸识别常用评价指标) · [Emore training backbones](../public/2020/12-01_Insightface_training.md#Emore%20training%20backbones) · [Mobilenet on Emore](../public/2020/12-01_Insightface_training.md#Mobilenet%20on%20Emore) · [Mobilenet on CASIA](../public/2020/12-01_Insightface_training.md#Mobilenet%20on%20CASIA) · [Distillation](../public/2020/12-01_Insightface_training.md#distillation) · [TFlite](../public/2020/12-01_Insightface_training.md#tflite)
- **[Keras Insightface Advance](../public/2021/04-07_insightface_advance.md)** (2021-04-07) — Advanced Keras InsightFace experiments — BoTNet/GhostNet/MobileNet backbones, SAM, distillation, IJB evaluation.
  - [Backbones](../public/2021/04-07_insightface_advance.md#backbones) · [EuclideanDense](../public/2021/04-07_insightface_advance.md#euclideandense) · [MXNet Pytorch Results](../public/2021/04-07_insightface_advance.md#MXNet%20Pytorch%20Results) · [cavaface.pytorch](../public/2021/04-07_insightface_advance.md#cavaface%20pytorch) · [Insightface](../public/2021/04-07_insightface_advance.md#insightface) · [IJB to bins](../public/2021/04-07_insightface_advance.md#IJB%20to%20bins) · [Distillation](../public/2021/04-07_insightface_advance.md#distillation)
- **[Face Quality](../public/2022/01-10_face_quality.md)** (2022-01-10) — Face quality and newer losses — masked-face similarity, MagFace, VPL, PartialFC, triplet mining, AdaFace.
  - [Similarity with face mask](../public/2022/01-10_face_quality.md#Similarity%20with%20face%20mask) · [MagFace Face Quality test](../public/2022/01-10_face_quality.md#MagFace%20Face%20Quality%20test) · [MagFace Training log](../public/2022/01-10_face_quality.md#MagFace%20Training%20log) · [Fine-tune](../public/2022/01-10_face_quality.md#fine-tune) · [XLA issue](../public/2022/01-10_face_quality.md#XLA%20issue) · [DepthwiseConv2D float16 issue](../public/2022/01-10_face_quality.md#DepthwiseConv2D%20float16%20issue) · [VPL](../public/2022/01-10_face_quality.md#vpl) · [PartialFC](../public/2022/01-10_face_quality.md#partialfc)

## Detection

- **[Face Detect](../public/2020/08-21_face_detect.md)** (2020-08-21) — Face detection — alignment and landmarks, MTCNN variants, Yoloface, CenterFace, RetinaFace, mask detection, Ultraface.
  - [Basic](../public/2020/08-21_face_detect.md#basic) · [MTCNN](../public/2020/08-21_face_detect.md#mtcnn) · [Yoloface](../public/2020/08-21_face_detect.md#yoloface) · [Centerface](../public/2020/08-21_face_detect.md#centerface) · [Retinaface](../public/2020/08-21_face_detect.md#retinaface) · [口罩检测](../public/2020/08-21_face_detect.md#口罩检测) · [Ultraface](../public/2020/08-21_face_detect.md#ultraface)
- **[YOLOV5 Face](../public/2021/11-05_yolov5face.md)** (2021-11-05) — YOLOv5-Face conversion and TFLite tests, NV21 to RGB, Android bitmap reading.
  - [YOLOV5 Face and convert tests](../public/2021/11-05_yolov5face.md#YOLOV5%20Face%20and%20convert%20tests) · [Tflite Convert](../public/2021/11-05_yolov5face.md#Tflite%20Convert) · [Effv2B0 model to tflite](../public/2021/11-05_yolov5face.md#Effv2B0%20model%20to%20tflite) · [NV21 to RGB](../public/2021/11-05_yolov5face.md#NV21%20to%20RGB) · [Read bitmap](../public/2021/11-05_yolov5face.md#Read%20bitmap)

## Anti-spoofing

- **[Face Anti Spoofing](../public/2019/08-20_anti_spoofing.md)** (2019-08-20) — Face anti-spoofing — liveness methods, LBP features in HSV/YCbCr, NUAA dataset, model reports, Keras models.
  - [活体检测方法](../public/2019/08-20_anti_spoofing.md#活体检测方法) · [LBP](../public/2019/08-20_anti_spoofing.md#lbp) · [Report](../public/2019/08-20_anti_spoofing.md#report) · [Keras](../public/2019/08-20_anti_spoofing.md#keras) · [其他特征](../public/2019/08-20_anti_spoofing.md#其他特征) · [数据收集与整理](../public/2019/08-20_anti_spoofing.md#数据收集与整理)
- **[SiW PAD](../public/2020/04-10_SiW_PAD.md)** (2020-04-10) — SiW presentation-attack dataset preparation, training, and Android TFLite deployment.
  - [数据整理](../public/2020/04-10_SiW_PAD.md#数据整理) · [Basic Training](../public/2020/04-10_SiW_PAD.md#Basic%20Training) · [Android TFlite](../public/2020/04-10_SiW_PAD.md#Android%20TFlite)

## See also

- [Backbones & ImageNet Training](backbones_imagenet.md)
- [Computer Vision](computer_vision.md)
