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

# Readme with Download
  ```py
  with open('trtexec.log') as ff:
      bb = ff.readlines()
  trtexec_qps = {ii.split('.onnx.log')[0]: ii.split('Throughput: ')[-1].split(' qps')[0] for ii in bb}

  with open("README.md") as ff:
      aa = ff.readlines()

  input_shapes = ["224", "256", "336", "384", "448", "512"]
  tt = []
  pre_name = ""
  start_mark, started = "# Recognition Models", False
  for line in aa:
      if line.startswith(start_mark):
          started = True
      if not started:
          tt.append(line)
          continue

      if "|"  in line and not ("FLOPs" in line or "------" in line):
          ss = line.split(" | ")
          orign_len = len(ss[1])

          model_name = ss[1].strip()
          prefix, model_name = ("- ", model_name[2:]) if model_name.startswith("- ") else ("", model_name)
          if "](" in line:
              surfix = " " * (orign_len - len(prefix) - len(model_name))
              model_url = prefix + "[" + model_name + "](" + ss[-1].split('](')[-1][:-3].strip() + surfix
          else:
              model_url = ss[1]
          line = " | ".join([ss[0], model_url, *ss[2:-1]]) + " |"

          if len(prefix) == 0:
              pre_name = model_name = model_name.split(",")[0]
          elif "=" in model_name:
              pre_name = "{}.{}".format(pre_name, model_name.split("=")[0])

          surfix = ""
          for ii in input_shapes:
              if ii in model_name:
                  surfix = ".{}".format(ii)
                  break
          model_name = pre_name + surfix
          cur_qps = (trtexec_qps.get(model_name)[:8] + " qps") if model_name in trtexec_qps else ""
          line += " " + cur_qps + " " * (12 - len(cur_qps)) + " |\n"
          print(model_name, cur_qps)
      line = line.replace("Download |\n", "T4 Inference |\n")
      line = line.replace("-------- |\n", "------------ |\n")
      tt.append(line)
  print("".join(tt))
  ```
# Readme with T4 Inference
  ```sh
  grep -i Throughput * > ../trtexec.log
  ```
  ```py
  with open('trtexec.log') as ff:
      bb = ff.readlines()
  trtexec_qps = {ii.split('.log')[0].replace(".onnx", ""): ii.split('Throughput: ')[-1].split(' qps')[0] for ii in bb}

  with open("README.md") as ff:
      aa = ff.readlines()

  input_shapes = ["160", "224", "256", "288", "336", "384", "448", "512"]
  tt = []
  pre_name = ""
  start_mark, started = "# Recognition Models", False
  for line in aa:
      if line.startswith(start_mark):
          started = True
      if not started:
          tt.append(line)
          continue

      if "|"  in line and not ("FLOPs" in line or "------" in line):
          ss = line.split(" | ")
          orign_len = len(ss[1])

          model_name = ss[1]
          prefix, model_name = ("- ", model_name[2:]) if model_name.startswith("- ") else ("", model_name)
          model_name = model_name.strip().split('](')[0].split('[')[-1]
          # line = " | ".join([ss[0], model_url, *ss[2:-1]]) + " |"

          surfix = ""
          check_input_shapes = ''.join(model_name.split(",")[1:]) if len(prefix) == 0 else model_name
          for ii in input_shapes:
              if ii in check_input_shapes:
                  surfix = ".{}".format(ii)
                  break

          if len(prefix) == 0:
              pre_name = model_name = model_name.split(",")[0]
          elif "=" in model_name:
              pre_name = "{}.{}".format(pre_name, model_name.split("=")[0])

          model_name = pre_name + surfix
          # if model_name in trtexec_qps:
          #     cur_qps = (trtexec_qps.get(model_name)[:8] + " qps")
          #     line = ' | '.join(ss[:-1]) + " | " + cur_qps + " " * (12 - len(cur_qps)) + " |\n"
          cur_qps = (trtexec_qps.get(model_name)[:8] + " qps") if model_name in trtexec_qps else ""
          line = ' | '.join(ss[:-1]) + " | " + cur_qps + " " * (12 - len(cur_qps)) + " |\n"
          print(model_name, cur_qps)
      tt.append(line)

  print("".join(tt))
  with open("foo.md", "w") as ff:
      ff.write("".join(tt))
  ```
## CSV result
  ```py
with open("README.md") as ff:
    aa = ff.readlines()

tt = []
start_mark, started, end_mark = "# Recognition Models", False, "# Stable Diffusion"
extra_input_resolutions = ["160", "224", "256", "288", "336", "384", "448", "512"]
category = "Recognition"
model_series = ""
for line in aa:
    if line.startswith(end_mark):
        break
    if line.startswith(start_mark):
        started = True
    if not started:
        continue

    if line.startswith("# "):
        category = line.split(" ")[1].strip()
    if line.startswith("## "):
        model_series = line.split(" ")[1].strip()
    if "|" not in line or "T4 Inference |" in line or "------------ |" in line:
        continue

    if "](" in line:
        line = line.split('](')[0].replace('[', '') + line.split(')')[-1]

    ss = [ii.strip() for ii in line.split("|")]
    ss = [*ss[1:-3], ss[-2]] if category == "Detection" else ss[1:-1]
    model_name, params, flops, input_shape, top1_acc, inference = ss[:6]
    # print(model, params, flops, input_shape, top1_acc, inference)

    # model_series_tail = ""
    if model_name.startswith("- "):
        extra, model_name = model_name[2:].split(","), tt[-1]["model"]
        if "=" in extra[0]:
            kk, vv = [ii.strip() for ii in extra[0].split("=")[:2]]
            model_name += "_" + kk + ("" if vv.lower() == "true" else ("_" + vv))
            # model_series_tail = "_deploy" if kk == "deploy" else model_series_tail
            extra = extra[1:]
    elif "," in model_name:
        ss = [ii.strip() for ii in model_name.split(",")]
        model_name, extra = ss[0].strip(), ss[1:]
    else:
        extra = []

    extra = [ii.strip() for ii in extra]
    extra = [ii for ii in extra if len(ii) > 0 and not ii.startswith("(") and ii not in extra_input_resolutions]

    cur = {}
    cur["model"] = model_name
    cur["params"] = float(params[:-1]) if params[-1] == "M" else (float(params[:-1]) * 1000)
    cur["flops"] = float(flops[:-1]) if flops[-1] == "G" else (float(flops[:-1]) / 1000)
    cur["input"] = int(input_shape)
    cur["acc_metrics"] = None if len(top1_acc) == 0 else float(top1_acc.replace("?", ""))
    cur["inference_qps"] = None if len(inference) == 0 else float(inference[:-4])

    cur["category"] = category
    cur["series"] = "ConvFormer" if model_name.startswith("ConvFormer") else model_series
    cur["extra"] = None if len(extra) == 0 else " ".join(extra)

    tt.append(cur)
dd = pd.DataFrame(tt)

ee = dd[dd.category == "Recognition"]
ee = ee[ee['acc_metrics'].notnull()]
ee = ee[ee['inference_qps'].notnull()]
ee = ee[ee['extra'].isnull()]
# plt.scatter(ee['T4 Inference (qps)'].values, ee['Top1 Acc'].values)

plot_series = ["EfficientViT_B", "EfficientViT_M", "EfficientNet", "EfficientNetV2"]
x_label = 'inference_qps'
y_label = 'acc_metrics'
for name, group in ee.groupby(ee['series']):
    if name not in plot_series:
        continue
    xx = group[x_label].values
    yy = group[y_label].values
    plt.scatter(xx, yy, label=name)
    plt.plot(xx, yy)
    for ii, jj, kk in zip(xx, yy, group['model'].values):
        plt.text(ii, jj, kk[len(name):])
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend()
plt.grid(True)
plt.tight_layout()

dd.to_csv('foo.csv', index=False)
  ```
  ```py
  cc = pd.read_csv('model_summary.csv')
  # cc.index = ['{}.{}'.format(ii, jj) for ii, jj in zip(cc.model.values, cc.input.values)]
  # dd.index = ['{}.{}'.format(ii, jj) for ii, jj in zip(dd.model.values, dd.input.values)]
  # dd.join(cc[['inference_qps']], rsuffix='_original', how='left')
  dd['original_inference_qps'] = cc.inference_qps
  dd.to_csv('goo.csv', index=False)
  ```
