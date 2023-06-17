## trtexec env
  - [TensorRT NGC | CATALOG](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags)
  - [colab trtexec.ipynb](https://colab.research.google.com/drive/1xLwfvbZNqadkdAZu9b0UzOrETLo657oc?usp=share_link)
  - **Get trtexec from docker**
    ```sh
    # tensorrt docker with CUDA=11.8
    docker run -v ~/Ubuntu_share:/Ubuntu_share -it nvcr.io/nvidia/tensorrt:23.04-py3
    # Cp trtexec out
    $ cp /usr/src/tensorrt/bin/trtexec /Ubuntu_share
    ```
    | docker image       | CUDA version |
    | ------------------ | ------------ |
    | tensorrt:23.05-py3 | CUDA=12.1    |
    | tensorrt:23.01-py3 | CUDA=12.0    |
    | tensorrt:22.12-py3 | CUDA=11.8    |
  - **Environment**
    ```py
    # Check os info
    !uname -a
    !cat /etc/os-release
    !ls -l /usr/local
    !nvcc --version
    !nvidia-smi

    # Install packages
    !pip install torch torchvision onnx pycuda tensorrt
    !apt install libnvinfer-dev libnvinfer-plugin-dev libnvparsers-dev libnvonnxparsers-dev

    # Get pre-build trtexec
    !wget https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/trtexec.12.1 -o trtexec
    !chmod a+x trtexec
    !ls -l
    ```
    **Test**
    ```py
    import torch
    import torchvision
    import numpy as np
    mm = torchvision.models.resnet50(pretrained=True)
    torch.onnx.export(mm, torch.ones([1, 3, 224, 224]), "aaa.onnx")
    ```
    ```py
    !./trtexec --onnx=aaa.onnx --fp16 --allowGPUFallback
    ```
## TensorRT Python timeit
  ```py
  import pycuda.driver as cuda


  def timeit(engine, loop=1000, warmup=20):
    import time

    inputs = np.random.uniform(size=engine.input_shape).astype('float32').ravel()
    batch_size = 1
    np.copyto(engine.host_input[: inputs.shape[0]], inputs)
    cuda.memcpy_htod_async(engine.cuda_input, engine.host_input[: inputs.shape[0]], engine.stream)

    total_cuda_time = 0
    for id in range(loop + warmup + warmup):
      if id == warmup:
        total_cuda_time = 0

      # Run inference asynchronously, same function in cpp is `IExecutionContext::enqueueV2`
      start = time.time()
      engine.context.execute_async_v2(bindings=engine.allocations, stream_handle=engine.stream.handle)
      engine.stream.synchronize()
      end = time.time()

      if id < loop + warmup:
        total_cuda_time += end - start
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(engine.host_output[: batch_size * engine.output_ravel_dim], engine.cuda_output, engine.stream)
    # Synchronize the stream
    engine.stream.synchronize()


    return 1 / (total_cuda_time / loop)

  timeit(engine)
  ```
## Kecam
- **export onnx using default shape**
  ```py
  import os
  os.environ["KECAM_BACKEND"] = "torch"

  import kecam
  skips = [
      "GPT2_Base", "GPT2_Medium", "GPT2_Large", "GPT2_XLarge","CoAtNet5", "CoAtNet6", "CoAtNet7",
      "EvaGiantPatch14", "ConvNeXtXXlarge", "ConvNeXtV2Huge", "DaViT_G", "DINOv2_ViT_Giant14",
  ]
  off_simplify_models = []
  start_model, started = "", True  # Start from specific model_name if interuptted
  total = len([model_name for model_name in kecam.models.__dict__.keys() if model_name[0].isupper()])
  log_path = "logs"
  failed = []
  for id, (model_name, model_class) in enumerate(kecam.models.__dict__.items()):
      if not model_name[0].isupper():
          continue
      print(f"[{id}/{total}] {model_name = }")

      if model_name == start_model:
          started = True
      if not started:
          continue
      if model_name in skips:
          continue

      save_path = model_name + ".onnx"
      if os.path.exists(os.path.join(log_path, save_path + ".log")):
          continue
      filepath = os.path.join('onnx_models', save_path)
      if os.path.exists(filepath):
          continue

      try:
          model_class(pretrained=None).export_onnx(simplify=False if model_name in off_simplify_models else True, filepath=filepath)
      except Exception as ee:
          print(ee)
          failed.append(model_name)
          continue
      print()
  ```
- **model name mapping model class**
  ```py
  from glob2 import glob

  aa = [ii for ii in glob('keras_cv_attention_models/*/*.py') if not ii.startswith("_")]

  model_name_class_dict = {}
  for ii in aa:
      with open(ii) as ff:
          bb = ff.readlines()
      for jj in bb:
          if 'def ' in jj:
              pre_def = jj.split('def ')[1].split('(')[0]
          if 'model_name=' in jj and 'local' in jj:
              cur = jj.split('model_name=')[-1]
              # print(cur)
              cur = cur.split('"), ')[0].split('", "')[1] if cur.startswith('kwargs') else cur.split('"')[1]
              model_name_class_dict[cur] = pre_def
  ```
- **Find all other shapes in `PRETRAINED_DICT` without first one**
  ```py
  import os
  os.environ["KECAM_BACKEND"] = "torch"
  import sys
  sys.path.append(os.path.expanduser("~/workspace/keras_cv_attention_models"))

  import kecam

  PRETRAINED_DICT = {}
  for name, module in kecam.__dict__.items():
      if name.startswith('_'):
          continue
      for sub_name, sub_module in module.__dict__.items():
          if sub_name.startswith('_'):
              continue
          if hasattr(sub_module, "PRETRAINED_DICT"):
              for kk, vv in getattr(sub_module, "PRETRAINED_DICT").items():
                  rr = []
                  for ii in vv.values():
                      if isinstance(ii, dict):
                          rr.extend([jj for jj in ii.keys() if jj not in rr])
                  if len(rr) > 0:
                      PRETRAINED_DICT[sub_name + "." + kk] = rr
  new_shapes = {kk: vv[1:] for kk, vv in PRETRAINED_DICT.items() if len(vv) > 1}

  log_path = "logs"
  skips = [
      # "EfficientNetV1L2", "MaxViT_XLarge", "EVA02LargePatch14",
      "EvaGiantPatch14", "CoAtNet5", "CoAtNet6", "CoAtNet7", "ConvNeXtXXlarge", "ConvNeXtV2Huge", "DaViT_G", "DINOv2_ViT_Giant14",
      "GPT2_Base", "GPT2_Medium", "GPT2_Large", "GPT2_XLarge",
  ]
  start_model, started = "", True  # Start from specific model_name if interuptted
  failed = []
  total = sum([len(ii) for ii in new_shapes.values()])
  id = 0
  gather = []
  for model_name, shapes in new_shapes.items():
      sub_class, model_name = model_name.split(".")
      model_class_name = model_name_class_dict[model_name]
      print(f"{sub_class = }, {model_name = }, {model_class_name = }")

      if model_class_name in skips:
          continue

      model_class = getattr(getattr(kecam, sub_class), model_class_name)
      gather.extend([(model_class_name, {"input_shape": input_shape}) for input_shape in shapes])

      for input_shape in shapes:
          id += 1
          print(f"[{id}/{total}] {model_name = }, {model_class_name = }, {input_shape = }")
          save_path = "{}.{}.onnx".format(model_class_name, input_shape)
          filepath = os.path.join('onnx_models', save_path)
          if os.path.exists(filepath):
              continue
          input_shape = (input_shape, input_shape, 3)

          try:
              model_class(input_shape=input_shape, pretrained=None).export_onnx(simplify=True, filepath=filepath)
          except Exception as ee:
              print(ee)
              failed.append(model_name)
              continue
          print()
  ```
- **Other specific config**
  ```py
  import os
  os.environ["KECAM_BACKEND"] = "torch"
  import sys
  sys.path.append(os.path.expanduser("~/workspace/keras_cv_attention_models"))

  import kecam

  # sub_classes = {"nat": {}, "efficientdet": {}}
  sub_classes = {"vanillanet": {"deploy": True}, "yolo_nas": {"use_reparam_conv": True}}
  skips = ["YOLOV8Backbone"]
  start_model, started = "", True  # Start from specific model_name if interuptted
  failed = []
  gather = []
  for sub_class, sub_config in sub_classes.items():
      config_names = list(sub_config.keys())
      for model_name, model_class in getattr(kecam, sub_class).__dict__.items():
          if not model_name[0].isupper() or not inspect.isfunction(model_class) or model_name.lower() == sub_class or model_name in skips:
              continue
          print(f"{model_name = }, {sub_config = }")
          gather.append((model_name, sub_config))

  for model_name, sub_config in gather:
      model_class = getattr(kecam.models, model_name)
      try:
          mm = model_class(pretrained=None, **sub_config)
          kecam.model_surgery.export_onnx(mm, batch_size=1, simplify=True, filepath=filepath)
      except Exception as ee:
          print(ee)
          failed.append(model_name)
          continue
      print()
  ```
- **Export TF onnx models**
  ```py
  import os
  import sys
  sys.path.append(os.path.expanduser("~/workspace/keras_cv_attention_models"))

  import kecam
  import inspect

  # "RegNetZC16_EVO", "RegNetZD8_EVO"

  sub_classes = ["hornet", "nfnets", "volo"]
  skips = ["NormFreeNet_Light", "NormFreeNet", "NFNetF7"]
  off_simplify_models = []
  start_model, started = "", True  # Start from specific model_name if interuptted
  failed = []
  for sub_class in sub_classes:
      all_other_shapes = {}
      for kk, vv in getattr(getattr(getattr(kecam, sub_class), sub_class), "PRETRAINED_DICT").items():
          rr = []
          for ii in vv.values():
              if isinstance(ii, dict):
                  rr.extend([jj for jj in ii.keys() if jj not in rr])
          if len(rr) > 1:
              all_other_shapes[kk] = rr[1:]

      for model_name, model_class in getattr(kecam, sub_class).__dict__.items():
          if not model_name[0].isupper() or not inspect.isfunction(model_class) or model_name.lower() == sub_class:
              continue
          print(f"{model_name = }")

          if model_name == start_model:
              started = True
          if not started:
              continue
          if model_name in skips:
              continue

          filepath = os.path.join('onnx_models', model_name + ".onnx")
          if os.path.exists(filepath):
              continue

          try:
              mm = model_class(pretrained=None)
              kecam.model_surgery.export_onnx(mm, batch_size=1, simplify=False if model_name in off_simplify_models else True, filepath=filepath)
          except Exception as ee:
              print(ee)
              failed.append(model_name)
              continue

          if mm.name in all_other_shapes:
              for ii in all_other_shapes[mm.name]:
                  print(f"{mm.name = }, {model_name = }, {ii = }")
                  filepath = os.path.join('onnx_models', "{}.{}.onnx".format(model_name, ii))
                  if os.path.exists(filepath):
                      continue
                  mm = model_class(input_shape=(ii, ii, 3), pretrained=None)
                  kecam.model_surgery.export_onnx(mm, batch_size=1, simplify=False if model_name in off_simplify_models else True, filepath=filepath)
          print()
  ```
- **Run `trtexec_test.sh`**
  ```sh
  #!/bin/bash
  waitGPU 0 50

  ALL_ONNX=( `ls -1 onnx_models` )
  total=${#ALL_ONNX[@]}
  for ((id=0; $id<${#ALL_ONNX[@]}; id=$id+1)); do
      waitGPU 0 50

      CUR=${ALL_ONNX[id]}
      echo "[$(( id + 1 ))/$total] [trtexec] Current onnx model: $CUR"
      if [ -e logs/${CUR}.log ]; then
          continue
      fi

      CUR_FILE=onnx_models/${CUR}
      # echo "trtexec --onnx=${CUR_FILE} --fp16 --allowGPUFallback --useSpinWait > logs/${CUR}.log 2>&1 && rm $CUR_FILE -f"
      trtexec --onnx=${CUR_FILE} --fp16 --allowGPUFallback --useSpinWait --useCudaGraph > logs/${CUR}.log 2>&1 # && rm $CUR_FILE -f
      grep Throughput logs/${CUR}.log
      echo ""
  done
  ```
- **perform output**
  ```py

  ```
***

## Train test
  ```py
  import os
  os.environ['KECAM_BACKEND'] = 'torch'

  import torch, torch_npu
  from kecam import fasternet
  mm = fasternet.FasterNetT2(input_shape=(32, 32, 3), num_classes=10, pretrained=None)
  mm = mm.npu()
  _ = mm.train()

  def fake_data_gen(input_shape=(3, 32, 32), num_classes=10, batch_size=16):
      while True:
          yield torch.randn([batch_size, *input_shape]), torch.randint(0, num_classes, [batch_size])

  input_shape, num_classes = mm.input_shape[1:], mm.output_shape[-1]
  optimizer = torch.optim.SGD(mm.parameters(), lr=0.1)
  dd = fake_data_gen(input_shape=input_shape, num_classes=num_classes)

  for ii in range(100):
      xx, yy = next(dd)
      out = mm(xx.npu())
      yy_one_hot = torch.functional.F.one_hot(yy, num_classes=num_classes).float().npu()
      loss = torch.functional.F.cross_entropy(out, yy_one_hot)
      loss.backward()
      optimizer.step()
      print(">>>> loss: {:.4f}".format(loss.item()))
  ```
