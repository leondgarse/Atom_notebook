- [TFLite Performance measurement](https://www.tensorflow.org/lite/performance/measurement)
```sh
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push your_model.tflite /data/local/tmp
```
```sh
adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args \
'"--graph=/data/local/tmp/efficientformer_l1.tflite --num_threads=4"'
```
| Model             | Dense, use_xnnpack=false  | Conv, use_xnnpack=false   | Conv, use_xnnpack=true    |
| ----------------- | ------------------------- | ------------------------- | ------------------------- |
| MobileViT_S       | Inference (avg) 215371 us | Inference (avg) 163836 us | Inference (avg) 163817 us |
| EfficientFormerL1 | Inference (avg) 126829 us | Inference (avg) 107053 us | Inference (avg) 107132 us |
```sh
adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=1 --use_xnnpack=true
```
