- [Github Ascend/ascend_community_projects/DetectAndAlign.cpp](https://github.com/Ascend/ascend_community_projects/blob/master/Individual_V2/src/DetectAndAlign.cpp)
- [Github /yangyucheng000/ms_cv/Resnetv2.cpp](https://github.com/yangyucheng000/ms_cv/blob/main/resnetv2/infer/resnetv2_101/mxbase/src/Resnetv2.cpp)
- [MindX SDK API](https://www.hiascend.com/document/detail/zh/mind-sdk/50rc1/vision/mxvisionug/mxvisionug_0796.html)
- [opencv C API](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html)
## CPP
  - main
    ```cpp
    #include <dirent.h>
    #include <fstream>
    #include "MxBase/Log/Log.h"
    #include "DnCNN.h"

    int main(int argc, char* argv[]) {
        InitParam initParam{};
        initParam.deviceId = 0;
        initParam.checkTensor = true;
        initParam.modelPath = "resnet50.om";

        auto model = std::make_shared<ResNet50>();
        APP_ERROR ret = model->Init(initParam);
        if (ret != APP_ERR_OK) {
            model->DeInit();
            LogError << "ResNet50 init failed, ret=" << ret << ".";
            return ret;
        }

        std::string imgPath = argv[1];
        LogInfo << "Processing " << imgPath;
        float psnr;
        ret = model->Process(imgPath, &psnr);
        if (ret != APP_ERR_OK) {
          LogError << "ResNet50 process failed, ret=" << ret << ".";
          model->DeInit();
          return ret;
        }
        LogInfo << "psnr value: " << psnr;
        model->DeInit();

        double total_time = model->GetInferCostMilliSec() / 1000;
        LogInfo << "inferance total cost time: " << total_time << ", FPS: "<< imagesPath.size() / total_time;
        return APP_ERR_OK;
    }
    ```
  - h file
    ```cpp
    #ifndef MXBASE_DnCNN_H
    #define MXBASE_DnCNN_H
    #include <memory>
    #include <string>
    #include <vector>
    #include <opencv2/opencv.hpp>
    #include "MxBase/ModelInfer/ModelInferenceProcessor.h"
    #include "MxBase/Tensor/TensorContext/TensorContext.h"
    #include "MxBase/CV/Core/DataType.h"

    struct InitParam {
        uint32_t deviceId;
        bool checkTensor;
        std::string modelPath;
    };

    class DnCNN {
    public:
        APP_ERROR Init(const InitParam &initParam);
        APP_ERROR DeInit();
        APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> *outputs);
        APP_ERROR Process(const std::string &imgPath, float *psnr);
        APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase);
        APP_ERROR TensorBaseToCVMat(const MxBase::TensorBase &tensorBase, cv::Mat *imageMat);
        // get infer time
        double GetInferCostMilliSec() const { return inferCostTimeMilliSec; }

    private:
        std::shared_ptr<MxBase::ModelInferenceProcessor> model_DnCNN;
        MxBase::ModelDesc modelDesc_;
        uint32_t deviceId_ = 0;
        // infer time
        double inferCostTimeMilliSec = 0.0;
    };

    #endif
    ```
  - c file
    ```cpp
    #include "DnCNN.h"
    #include <cstdlib>
    #include <memory>
    #include <string>
    #include <cmath>
    #include <vector>
    #include <algorithm>
    #include <queue>
    #include <utility>
    #include <fstream>
    #include <map>
    #include <iostream>
    #include "MxBase/DeviceManager/DeviceManager.h"
    #include "MxBase/Log/Log.h"

    namespace {
        const int imageSize = 481;
    }

    APP_ERROR ResNet50::Init(const InitParam &initParam) {
        deviceId_ = initParam.deviceId;
        APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
        if (ret != APP_ERR_OK) {
            LogError << "Init devices failed, ret=" << ret << ".";
            return ret;
        }
        ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
        if (ret != APP_ERR_OK) {
            LogError << "Set context failed, ret=" << ret << ".";
            return ret;
        }
        model_DnCNN = std::make_shared<MxBase::ModelInferenceProcessor>();
        ret = model_DnCNN->Init(initParam.modelPath, modelDesc_);
        if (ret != APP_ERR_OK) {
            LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
            return ret;
        }

        return APP_ERR_OK;
    }

    APP_ERROR ResNet50::DeInit() {
        model_DnCNN->DeInit();
        MxBase::DeviceManager::GetInstance()->DestroyDevices();
        return APP_ERR_OK;
    }

    APP_ERROR ResNet50::CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
        uint32_t dataSize =  imageSize * imageSize * sizeof(float);
        MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
        MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Memory malloc failed.";
            return ret;
        }
        std::vector<uint32_t> shape = {1, 1, imageSize, imageSize};
        *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
        return APP_ERR_OK;
    }

    APP_ERROR ResNet50::TensorBaseToCVMat(const MxBase::TensorBase &tensorBase, cv::Mat *imageMat) {
        uint32_t dataSize = imageSize * imageSize * sizeof(float);
        void* buffer = tensorBase.GetBuffer();
        std::vector<float> vec(imageSize * imageSize);
        memcpy(vec.data(), buffer, dataSize);
        *imageMat = cv::Mat(vec, true).reshape(0, imageSize);
        return APP_ERR_OK;
    }

    APP_ERROR ResNet50::Inference(const std::vector<MxBase::TensorBase> &inputs,
                                          std::vector<MxBase::TensorBase> *outputs) {
        auto dtypes = model_DnCNN->GetOutputDataType();
        for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
            std::vector<uint32_t> shape = {};
            for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
                shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
            }
            MxBase::TensorBase tensor(shape, dtypes[i], MxBase::MemoryData::MemoryType::MEMORY_DEVICE, deviceId_);
            APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor);
            if (ret != APP_ERR_OK) {
                LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
                return ret;
            }
            outputs->push_back(tensor);
        }
        MxBase::DynamicInfo dynamicInfo = {};
        dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
        auto startTime = std::chrono::high_resolution_clock::now();
        APP_ERROR ret = model_DnCNN->ModelInference(inputs, *outputs, dynamicInfo);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        inferCostTimeMilliSec += costMs;
        if (ret != APP_ERR_OK) {
            LogError << "ModelInference DnCNN failed, ret=" << ret << ".";
            return ret;
        }
        return APP_ERR_OK;
    }

    APP_ERROR ResNet50::Process(const std::string &imgPath, float *psnr) {
        cv::Mat img_origin = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
        cv::resize(img_origin, img_origin, cv::Size(imageSize, imageSize));

        cv::Mat img;
        img_origin.convertTo(img, CV_32F, 1.0 / 255.0);

        std::vector<MxBase::TensorBase> inputs;
        std::vector<MxBase::TensorBase> outputs;
        MxBase::TensorBase tensorBase;
        APP_ERROR ret = CVMatToTensorBase(noisy, &tensorBase);
        if (ret != APP_ERR_OK) {
            LogError << "CVMatToTensorBase failed, ret = " << ret << ".";
            return ret;
        }

        inputs.push_back(tensorBase);
        auto startTime = std::chrono::high_resolution_clock::now();
        ret = Inference(inputs, &outputs);
        auto endTime = std::chrono::high_resolution_clock::now();
        double costMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        inferCostTimeMilliSec += costMs;
        if (ret != APP_ERR_OK) {
            LogError << "Inference failed, ret=" << ret << ".";
            return ret;
        }
        if (!outputs[0].IsHost()) {
            outputs[0].ToHost();
        }

        cv::Mat residual;
        TensorBaseToCVMat(outputs[0], &residual);
        return APP_ERR_OK;
    }

    ```
## Module
  ```cpp
  APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase *tensorBase) {
      uint32_t dataSize =  imageSize * imageSize * sizeof(float);
      MxBase::MemoryData memoryDataDst(dataSize, MxBase::MemoryData::MEMORY_DEVICE, deviceId_);
      MxBase::MemoryData memoryDataSrc(imageMat.data, dataSize, MxBase::MemoryData::MEMORY_HOST_MALLOC);

      APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
      if (ret != APP_ERR_OK) {
          LogError << GetError(ret) << "Memory malloc failed.";
          return ret;
      }
      std::vector<uint32_t> shape = {1, 1, imageSize, imageSize};
      *tensorBase = MxBase::TensorBase(memoryDataDst, false, shape, MxBase::TENSOR_DTYPE_FLOAT32);
      return APP_ERR_OK;
  }
  ```
  ```cpp
  APP_ERROR TensorBaseToCVMat(const MxBase::TensorBase &tensorBase, cv::Mat *imageMat) {
      uint32_t dataSize = imageSize * imageSize * sizeof(float);
      void* buffer = tensorBase.GetBuffer();
      std::vector<float> vec(imageSize * imageSize);
      memcpy(vec.data(), buffer, dataSize);
      *imageMat = cv::Mat(vec, true).reshape(0, imageSize);
      return APP_ERR_OK;
  }
  ```
## Python
  ```py
  import numpy as np
  from mindx.sdk import base
  from mindx.sdk.base import Image, Model, ImageProcessor, Size

  # 资源初始化
  print('==========资源初始化=========')
  base.mx_init()
  # 设备ID
  device_id = 0

  # 图像解码
  # 初始化ImageProcessor对象
  imageProcessor = ImageProcessor(device_id)
  image_path = "data/test_dog.jpg"
  # 读取图片路径进行解码，解码格式为nv12（YUV_SP_420）
  decoded_image = imageProcessor.decode(image_path, base.nv12)
  print('decoded_image width:  {}'.format(decoded_image.original_width))
  print('decoded_image height:  {}'.format(decoded_image.original_height))
  print('==========图片解码完成=========')

  # 图像缩放
  # 缩放尺寸
  size_para = Size(224, 224)
  # 读取将解码后的Image类按尺寸进行缩放，缩放方式为华为自研的高阶滤波算法（huaweiu_high_order_filter）
  resized_image = imageProcessor.resize(decoded_image, size_para, base.huaweiu_high_order_filter)
  print('resized_image width:  ', resized_image.original_width)
  print('resized_image height:  ', resized_image.original_height)
  print('==========图片缩放完成=========')

  # 模型推理
  # Image类转为Tensor类并传入列表
  input_tensors = [resized_image.to_tensor()]
  # 模型路径
  model_path = "./model/resnet50_batchsize_1.om"
  # 初始化Model类
  # 也可使用model = base.model(modelPath=model_path, deviceId=device_id)
  model = Model(modelPath=model_path, deviceId=device_id)
  # 执行推理
  outputs = model.infer(input_tensors)
  print('==========模型推理完成=========')

  # 后处理
  # 获取推理结果置信度tensor
  confidence_tensor = outputs[0]
  # 将tensor数据转移到Host侧
  confidence_tensor.to_host()
  # 将Tensor类转为numpy array类型
  confidence_array = np.array(confidence_tensor)
  # 获取最大置信度序号和置信度
  max_confidence_index = confidence_array.argmax(axis=1)[0]
  max_confidence_value = confidence_array[0][max_confidence_index]
  print('max_confidence_index:', max_confidence_index)
  print('max_confidence_value:', max_confidence_value)
  # 分类标签文件路径
  label_path = "./model/resnet50_clsidx_to_labels.names"
  # 读取分类标签文件并查询分类结果
  with open(label_path, 'r', encoding='utf-8') as f:
      file = f.read()
  label_list = file.split('\n')[0:-1]
  infer_result = label_list[max_confidence_index]
  print('classification result:{}'.format(infer_result))
  ```
## test opencv
  ```cpp
  /*
  apt install libopencv-dev
  g++ -O3 cpp/yolo.cpp -o yolo_sample `pkg-config --cflags --libs opencv4`
  */
  #include <fstream>

  #include <opencv2/opencv.hpp>
  #include "MxBase/ModelInfer/ModelInferenceProcessor.h"
  #include "MxBase/Tensor/TensorContext/TensorContext.h"

  void CVMatToTensorBase(const cv::Mat &imageMat, MxBase::TensorBase &tensorBase){
      uint32_t deviceId_ = 0;
      const uint32_t dataSize = imageMat.cols * imageMat.rows * YUV444_RGB_WIDTH_NU;
      MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
      MemoryData memoryDataSrc(imageMat.data, dataSize, MemoryData::MEMORY_HOST_MALLOC);
      MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
      std::vector<uint32_t> shape = {1, 3, imageMat.rows, imageMat.cols};
      tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_FLOAT32);
  }

  APP_ERROR TensorBaseToCVMat(const MxBase::TensorBase &tensorBase, cv::Mat &imageMat) {
      void* buffer = tensorBase.GetBuffer();
      std::vector<float> vec(imageSize * imageSize);
      memcpy(vec.data(), buffer, tensorBase.GetByteSize());
      &imageMat = cv::Mat(vec, true).reshape(0, tensorBase.GetShape());
      return APP_ERR_OK;
  }

  std::vector<std::string> load_class_list() {
      std::vector<std::string> class_list;
      std::ifstream ifs("config_files/classes.txt");
      std::string line;
      while (getline(ifs, line))
      {
          class_list.push_back(line);
      }
      return class_list;
  }

  void load_net(cv::dnn::Net &net, uint32_t device_id) {
      /*
      auto result = cv::dnn::readNet("config_files/yolov5s.onnx");
      if (is_cuda)
      {
          std::cout << "Attempty to use CUDA\n";
          result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
          result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
      }
      else
      {
          std::cout << "Running on CPU\n";
          result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
          result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      }
      */
      MxBase::Model result("config_files/yolov5n.om", device_id);
      net = result;
  }

  const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.4;

  struct Detection {
      int class_id;
      float confidence;
      cv::Rect box;
  };

  cv::Mat format_yolov5(const cv::Mat &source) {
      int col = source.cols;
      int row = source.rows;
      int _max = MAX(col, row);
      cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
      source.copyTo(result(cv::Rect(0, 0, col, row)));
      return result;
  }

  void detect(cv::Mat &image, MxBase::Model &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
      cv::Mat blob;

      auto input_image = format_yolov5(image);
      cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

      /*
      net.setInput(blob);
      std::vector<cv::Mat> outputs;
      net.forward(outputs, net.getUnconnectedOutLayersNames());
      auto output = outputs[0];
      */
      std::vector<MxBase::TensorBase> inputs = {};

      TensorBase tensorBase;
      CVMatToTensorBase(blob, &tensorBase);
      mx_inputs.push_back(tensorBase);
      // model_DnCNN->ModelInference(inputs, *outputs, dynamicInfo);
      std::vector<MxBase::Tensor> mx_outputs = net.Infer(mx_inputs);
      cv::Mat output;
      TensorBaseToCVMat(tensorBase, &output);

      float x_factor = input_image.cols / INPUT_WIDTH;
      float y_factor = input_image.rows / INPUT_HEIGHT;
      float *data = (float *)output.data;

      const int dimensions = 85;
      const int rows = 25200;

      std::vector<int> class_ids;
      std::vector<float> confidences;
      std::vector<cv::Rect> boxes;

      for (int i = 0; i < rows; ++i) {

          float confidence = data[4];
          if (confidence >= CONFIDENCE_THRESHOLD) {

              float * classes_scores = data + 5;
              cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
              cv::Point class_id;
              double max_class_score;
              minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
              if (max_class_score > SCORE_THRESHOLD) {

                  confidences.push_back(confidence);

                  class_ids.push_back(class_id.x);

                  float x = data[0];
                  float y = data[1];
                  float w = data[2];
                  float h = data[3];
                  int left = int((x - 0.5 * w) * x_factor);
                  int top = int((y - 0.5 * h) * y_factor);
                  int width = int(w * x_factor);
                  int height = int(h * y_factor);
                  boxes.push_back(cv::Rect(left, top, width, height));
              }
          }
          data += 85;
      }

      std::vector<int> nms_result;
      cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
      for (int i = 0; i < nms_result.size(); i++) {
          int idx = nms_result[i];
          Detection result;
          result.class_id = class_ids[idx];
          result.confidence = confidences[idx];
          result.box = boxes[idx];
          output.push_back(result);
      }
  }

  int main(int argc, char **argv)
  {

      std::vector<std::string> class_list = load_class_list();
      cv::Mat frame = cv::imread("./test.jpg", 1);;
      if (frame.empty()) return 0;


      // bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
      // cv::dnn::Net net;
      // load_net(net, is_cuda);
      APP_ERROR ret = MxInit();
      if (ret != APP_ERR_OK) {
          LogError << "MxInit failed, ret=" << ret << ".";
          return ret;
      }
      MxBase::Model net;
      load_net(net, 0);

      auto start = std::chrono::high_resolution_clock::now();

          std::vector<Detection> output;
          detect(frame, net, output, class_list);

          int detections = output.size();

          for (int i = 0; i < detections; ++i)
          {
                  auto detection = output[i];
                  auto box = detection.box;
                  auto classId = detection.class_id;
                  const auto color = colors[classId % colors.size()];
                  cv::rectangle(frame, box, color, 3);

                  cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                  cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
          }
          auto end = std::chrono::high_resolution_clock::now();
          float fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

          std::ostringstream fps_label;
          fps_label << std::fixed << std::setprecision(2);
          fps_label << "FPS: " << fps;
          std::string fps_label_str = fps_label.str();
          cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
          cv::imwrite("output.png", frame);

      return 0;
  }
  ```
