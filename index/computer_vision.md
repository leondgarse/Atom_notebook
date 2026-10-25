# Computer Vision

Image processing, detection (EfficientDet / YOLO), segmentation (UNet / SAM), CLIP and diffusion.

Back to [all indexes](README.md).

## Image processing

- **[Image Processing](../public/2018/10-11_image_processing.md)** (2018-10-11) — Image processing — pixel operations, Otsu/Sobel/Canny/watershed, skimage algorithms, video and contour handling.
  - [图片处理常用库](../public/2018/10-11_image_processing.md#图片处理常用库) · [图片像素序列处理](../public/2018/10-11_image_processing.md#图片像素序列处理) · [图像处理算法](../public/2018/10-11_image_processing.md#图像处理算法) · [skimage](../public/2018/10-11_image_processing.md#skimage) · [视频文件](../public/2018/10-11_image_processing.md#视频文件) · [图像曲线闭合](../public/2018/10-11_image_processing.md#图像曲线闭合) · [计算点集中的矩形面积](../public/2018/10-11_image_processing.md#计算点集中的矩形面积) · [skimage segmentation](../public/2018/10-11_image_processing.md#skimage%20segmentation)
- **[Computer Vision](../public/2020/05-02_computer_vision.md)** (2020-05-02) — Computer vision basics — PCA, derivatives, perspective transform, inpaint, denoising, OpenCV motion detection.
  - [Basic](../public/2020/05-02_computer_vision.md#basic) · [图像去噪](../public/2020/05-02_computer_vision.md#图像去噪) · [Opencv 移动目标检测](../public/2020/05-02_computer_vision.md#Opencv%20移动目标检测) · [Opencv 应用](../public/2020/05-02_computer_vision.md#Opencv%20应用) · [CUDA 视频解码](../public/2020/05-02_computer_vision.md#CUDA%20视频解码) · [检测上下左右区域内的直线](../public/2020/05-02_computer_vision.md#检测上下左右区域内的直线) · [RGB IR detect](../public/2020/05-02_computer_vision.md#RGB%20IR%20detect) · [FaceTrack by match template](../public/2020/05-02_computer_vision.md#FaceTrack%20by%20match%20template)
- **[defog](../public/2024/03-16_defog.md)** (2024-03-16) — Image defogging model built from kecam layers.

## Detection & datasets

- **[Car Plate Recognition](../public/2019/09-27_car_plate.md)** (2019-09-27) — Car plate localisation — corner detection from masks and UNet/Mobile UNet segmentation.
  - [PLATE MASK](../public/2019/09-27_car_plate.md#PLATE%20MASK) · [skimage 角点检测](../public/2019/09-27_car_plate.md#skimage%20角点检测) · [巻积后按区域划分后的四个最小值点](../public/2019/09-27_car_plate.md#巻积后按区域划分后的四个最小值点) · [取巻积后所有最小值点](../public/2019/09-27_car_plate.md#取巻积后所有最小值点) · [测试](../public/2019/09-27_car_plate.md#测试) · [Unet 车牌定位](../public/2019/09-27_car_plate.md#Unet%20车牌定位) · [Unet](../public/2019/09-27_car_plate.md#unet) · [Mobile Unet](../public/2019/09-27_car_plate.md#Mobile%20Unet)
- **[EfficientDet](../public/2022/01-11_efficientdet.md)** (2022-01-11) — EfficientDet in Keras — COCO dataset, anchors, augmentation, model build and weight reload, training.
  - [COCO dataset](../public/2022/01-11_efficientdet.md#COCO%20dataset) · [Init dataset](../public/2022/01-11_efficientdet.md#Init%20dataset) · [EfficientDet build](../public/2022/01-11_efficientdet.md#EfficientDet%20build) · [Convert and predict](../public/2022/01-11_efficientdet.md#Convert%20and%20predict) · [Train](../public/2022/01-11_efficientdet.md#train) · [Evaluation](../public/2022/01-11_efficientdet.md#evaluation) · [EfficientDet lite](../public/2022/01-11_efficientdet.md#EfficientDet%20lite)
- **[YOLOX](../public/2022/02-17_yolox.md)** (2022-02-17) — YOLOX and YOLOR — reloading torch weights to Keras, post-processing, eval, mosaic training, anchor-free loss.
  - [YOLOX](../public/2022/02-17_yolox.md#YOLOX) · [YOLOX training](../public/2022/02-17_yolox.md#YOLOX%20training) · [YOLOR](../public/2022/02-17_yolox.md#yolor) · [YOLOR Training](../public/2022/02-17_yolox.md#YOLOR%20Training) · [YOLOXTiny training logs](../public/2022/02-17_yolox.md#YOLOXTiny%20training%20logs) · [TOLOR_CSP training logs](../public/2022/02-17_yolox.md#TOLOR_CSP%20training%20logs)
- **[GTSRB](../public/2022/08-06_GTSRB.md)** (2022-08-06) — German traffic sign (GTSRB) dataset preparation and custom multi-label models.
  - [GTSRB](../public/2022/08-06_GTSRB.md#gtsrb) · [Custom model](../public/2022/08-06_GTSRB.md#Custom%20model) · [Multi labels](../public/2022/08-06_GTSRB.md#Multi%20labels)
- **[Custom COCO](../public/2022/10-21_custom_coco.md)** (2022-10-21) — Building custom/tiny COCO datasets, annotation formats and COCO decoding.
  - [coco tiny](../public/2022/10-21_custom_coco.md#coco%20tiny) · [coco tiny for using custom_dataset_script.py](../public/2022/10-21_custom_coco.md#coco%20tiny%20for%20using%20custom_dataset_script%20py) · [COCO eval annotations format](../public/2022/10-21_custom_coco.md#COCO%20eval%20annotations%20format) · [From custom_dataset_script.py generated json to…](../public/2022/10-21_custom_coco.md#From%20custom_dataset_script%20py%20generated%20json%20to%20annotations) · [COCO decode](../public/2022/10-21_custom_coco.md#COCO%20decode) · [To recognition](../public/2022/10-21_custom_coco.md#To%20recognition)
- **[yolov7](../public/2022/11-26_yolov7.md)** (2022-11-26) — YOLOv7 torch model test and weight conversion.
  - [Yolov7 torch model test](../public/2022/11-26_yolov7.md#Yolov7%20torch%20model%20test) · [Convert weights](../public/2022/11-26_yolov7.md#Convert%20weights)
- **[yolov8](../public/2023/04-03_yolov8.md)** (2023-04-03) — YOLOv8 classification/detection/segmentation, training, logs, and YOLO-NAS source analysis and conversion.
  - [Data](../public/2023/04-03_yolov8.md#data) · [Train](../public/2023/04-03_yolov8.md#train) · [To yolov8 detector](../public/2023/04-03_yolov8.md#To%20yolov8%20detector) · [Plot log](../public/2023/04-03_yolov8.md#Plot%20log) · [YOLO NAS](../public/2023/04-03_yolov8.md#YOLO%20NAS)

## Segmentation

- **[Dicom Segmentation](../public/2019/09-05_physic_dicom.md)** (2019-09-05) — Medical DICOM segmentation — reading and preprocessing, SimpleITK, breast segmentation with UNet/UNet++.
  - [Dicom](../public/2019/09-05_physic_dicom.md#dicom) · [sunny_demmo](../public/2019/09-05_physic_dicom.md#sunny_demmo) · [DICOM 数据集](../public/2019/09-05_physic_dicom.md#DICOM%20数据集) · [sitk segmentation](../public/2019/09-05_physic_dicom.md#sitk%20segmentation) · [Breast segmentation](../public/2019/09-05_physic_dicom.md#Breast%20segmentation) · [UNet](../public/2019/09-05_physic_dicom.md#unet)
- **[sam](../public/2023/10-16_sam.md)** (2023-10-16) — Segment Anything — keras_cv SAM, MobileSAM and EfficientViT-SAM weight reloading and prediction.
  - [MobileSAM Reload weights](../public/2023/10-16_sam.md#MobileSAM%20Reload%20weights) · [EfficientViTL0 SAM Reload weights](../public/2023/10-16_sam.md#EfficientViTL0%20SAM%20Reload%20weights) · [Rename Maks decoder](../public/2023/10-16_sam.md#Rename%20Maks%20decoder)

## Multimodal & generative

- **[Machine Learning Projects](../public/2021/02-25_ML_projects.md)** (2021-02-25) — Assorted ML projects — super-resolution, OCR, GNN, attention, BERT/word embeddings, BiT, Transformer, ViT visualising.
  - [图像超分](../public/2021/02-25_ML_projects.md#图像超分) · [OCR](../public/2021/02-25_ML_projects.md#ocr) · [Attention](../public/2021/02-25_ML_projects.md#attention) · [BERT](../public/2021/02-25_ML_projects.md#bert) · [BiT](../public/2021/02-25_ML_projects.md#bit) · [Transformer](../public/2021/02-25_ML_projects.md#transformer) · [Visualizing](../public/2021/02-25_ML_projects.md#visualizing) · [Image segmentation](../public/2021/02-25_ML_projects.md#Image%20segmentation)
- **[clip](../public/2023/05-08_clip.md)** (2023-05-08) — CLIP — OpenCLIP, TensorFlow CLIP model and training on COCO tiny captions, single-tower variants.
  - [OpenClip](../public/2023/05-08_clip.md#openclip) · [CLIP tensorflow](../public/2023/05-08_clip.md#CLIP%20tensorflow) · [COCO tiny caption](../public/2023/05-08_clip.md#COCO%20tiny%20caption) · [Single tower](../public/2023/05-08_clip.md#Single%20tower) · [TF and Torch optimizer](../public/2023/05-08_clip.md#TF%20and%20Torch%20optimizer)
- **[diffusion](../public/2023/08-27_diffusion.md)** (2023-08-27) — Diffusion models — Karlo, diffusers pipelines, DDPM, Stable Diffusion weights and components, training.
  - [Karlo](../public/2023/08-27_diffusion.md#karlo) · [Difusers](../public/2023/08-27_diffusion.md#difusers) · [DDPM](../public/2023/08-27_diffusion.md#ddpm) · [StableDiffusion weights](../public/2023/08-27_diffusion.md#StableDiffusion%20weights) · [Train](../public/2023/08-27_diffusion.md#train)

## See also

- [Face Recognition & Anti-Spoofing](face_recognition.md)
- [Backbones & ImageNet Training](backbones_imagenet.md)
