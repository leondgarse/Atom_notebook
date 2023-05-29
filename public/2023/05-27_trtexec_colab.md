- [TensorRT NGC | CATALOG](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags)
- [colab tensorrt.ipynb](https://colab.research.google.com/drive/1xLwfvbZNqadkdAZu9b0UzOrETLo657oc?usp=share_link)
```sh
# tensorrt docker with CUDA=11.8
docker run -v ~/Ubuntu_share:/Ubuntu_share -it nvcr.io/nvidia/tensorrt:23.04-py3
# Cp trtexec out
$ cp /usr/src/tensorrt/bin/trtexec /Ubuntu_share
```
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
!wget https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/trtexec.12.1 -o trtexec
!chmod a+x trtexec
!ls -l
```
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
