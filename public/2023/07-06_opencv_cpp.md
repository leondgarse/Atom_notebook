- [opencv C API](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html)
## ResNet50
  ```cpp
  #include <fstream>
  #include <opencv2/opencv.hpp>

  std::vector<std::string> load_class_list() {
      std::vector<std::string> class_list;
      std::ifstream ifs("imagenet_class_labels.txt");
      std::string line;
      while (getline(ifs, line)) {
          class_list.push_back(line);
      }
      return class_list;
  }

  int main(int argc, char **argv) {
      std::vector<std::string> class_list = load_class_list();

      cv::dnn::Net net = cv::dnn::readNet("resnet50.onnx");
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

      auto start = std::chrono::high_resolution_clock::now();
      cv::Mat image = cv::imread("test.png", 1);
      if (image.empty()) return 0;

      cv::Mat resized_image, blob;
      cv::resize(image, resized_image, cv::Size(224, 224));
      cv::dnn::blobFromImage(resized_image, blob, 1.0, cv::Size(224, 224), cv::Scalar(), true, false);
      blob = (blob - cv::Scalar(123.675, 116.28, 103.53)) / cv::Scalar(58.395, 57.12, 57.375);

      auto end = std::chrono::high_resolution_clock::now();
      float time_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
      std::cout << "Image process time duration: " << time_duration << "ms" << std::endl;

      start = std::chrono::high_resolution_clock::now();
      net.setInput(blob);
      std::vector<cv::Mat> outputs;
      net.forward(outputs, net.getUnconnectedOutLayersNames());

      end = std::chrono::high_resolution_clock::now();
      time_duration= std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1.0e6;
      std::cout << "Model inference time duration: " << time_duration << "ms" << std::endl;

      int total_classes = 1000;
      int argmax = 0;
      float max_score = 0;
      float *data = (float *)outputs[0].data;
      for (int ii = 0; ii < total_classes; ii++) {
          if (data[ii] > max_score) {
              max_score = data[ii];
              argmax = ii;
          }
      }
      std::cout << "index: " << argmax << std::endl;
      std::cout << "class: " << class_list[argmax] << std::endl;
      std::cout << "score: " << max_score << std::endl;

      return 0;
  }
  ```
  **Compile**
  ```sh
  # Using pkg-config
  g++ -O3 resnet50_opencv.cpp -o resnet50_opencv `pkg-config --cflags --libs opencv4`

  # Or manually set lib
  g++ -O3 resnet50_opencv.cpp -o resnet50_opencv -I/usr/local/include/opencv4 -L/usr/local/lib/x86_64-linux-gnu -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

  # Or use custom built opencv
  g++ -O3 resnet50_opencv.cpp -o resnet50_opencv -I$HOME/local_bin/opencv4/include/opencv4 -L$HOME/local_bin/opencv4/lib -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
  ```
## YOLOV5
  ```cpp
  #include <fstream>
  #include <opencv2/opencv.hpp>

  const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

  const float INPUT_WIDTH = 640.0;
  const float INPUT_HEIGHT = 640.0;
  const float SCORE_THRESHOLD = 0.2;
  const float NMS_THRESHOLD = 0.4;
  const float CONFIDENCE_THRESHOLD = 0.3;

  struct Detection {
      int class_id;
      float confidence;
      cv::Rect box;
  };

  std::vector<std::string> load_class_list() {
      std::vector<std::string> class_list;
      std::ifstream ifs("coco_class_labels.txt");
      std::string line;
      while (getline(ifs, line)) {
          class_list.push_back(line);
      }
      return class_list;
  }

  void load_net(cv::dnn::Net &net, bool is_cuda) {
      auto result = cv::dnn::readNet("yolov5s.onnx");
      if (is_cuda) {
          std::cout << "Attempty to use CUDA\n";
          result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
          result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
      } else {
          std::cout << "Running on CPU\n";
          result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
          result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      }
      net = result;
  }

  cv::Mat format_yolov5(const cv::Mat &source) {
      int col = source.cols;
      int row = source.rows;
      int _max = MAX(col, row);
      cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
      source.copyTo(result(cv::Rect(0, 0, col, row)));
      return result;
  }

  void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
      cv::Mat blob;

      auto input_image = format_yolov5(image);
      cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
      /*
      for (int ii = 22000; ii < 22020; ii++) {
          std::cout << ((float*)blob.data)[ii] << ",";
      }
      std::cout << std::endl;
      */
      net.setInput(blob);
      std::vector<cv::Mat> outputs;
      net.forward(outputs, net.getUnconnectedOutLayersNames());

      float x_factor = input_image.cols / INPUT_WIDTH;
      float y_factor = input_image.rows / INPUT_HEIGHT;
      std::vector<int> class_ids;
      std::vector<float> confidences;
      std::vector<cv::Rect> boxes;

      const int dimensions = 85;
      const int rows = 25200;
      float *data = (float *)outputs[0].data;
      for (int i = 0; i < rows; ++i) {
          float confidence = data[4];
          if (confidence >= CONFIDENCE_THRESHOLD) {
              // std::cout << confidence << std::endl;

              float * classes_scores = data + 5;
              cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
              cv::Point class_id;
              double max_class_score;
              minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
              if (max_class_score > SCORE_THRESHOLD) {
                  confidences.push_back(confidence);
                  class_ids.push_back(class_id.x);

                  int left = int((data[0] - 0.5 * w) * x_factor);
                  int top = int((data[1] - 0.5 * h) * y_factor);
                  int width = int(data[2] * x_factor);
                  int height = int(data[3] * y_factor);
                  boxes.push_back(cv::Rect(left, top, width, height));
              }
          }
          data += dimensions;
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

  int main(int argc, char **argv) {
      std::vector<std::string> class_list = load_class_list();
      cv::Mat frame = cv::imread("test.png", 1);;
      if (frame.empty()) return 0;

      bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

      cv::dnn::Net net;
      load_net(net, is_cuda);

      auto start = std::chrono::high_resolution_clock::now();
      std::vector<Detection> output;
      detect(frame, net, output, class_list);
      int detections = output.size();

      for (int i = 0; i < detections; ++i) {
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
      std::cout << "Output filename: output.png" << std::endl;

      return 0;
  }
  ```
