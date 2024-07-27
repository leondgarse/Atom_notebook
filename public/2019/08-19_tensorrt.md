# ___2019 - 08 - 19 TensorRT___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 08 - 19 TensorRT___](#2019-08-19-tensorrt)
  - [目录](#目录)
  - [Install](#install)
  	- [链接](#链接)
  	- [TensorRT](#tensorrt)
  	- [ONNX tensorrt](#onnx-tensorrt)
  - [Inference](#inference)
  	- [UFF MNIST](#uff-mnist)
  	- [UFF MNIST Official Sample Definition](#uff-mnist-official-sample-definition)
  	- [ONNX Engine](#onnx-engine)
  	- [ONNX Engine With Optimization Profile](#onnx-engine-with-optimization-profile)
  	- [ONNX Engine With Multi Optimization Profile](#onnx-engine-with-multi-optimization-profile)
  	- [ONNX Engine with INT8 Calibrator](#onnx-engine-with-int8-calibrator)
  - [ONNX tensorrt](#onnx-tensorrt)
  - [CPP Inference](#cpp-inference)

  <!-- /TOC -->
***

# Install
## 链接
  - [tensorrt-developer-guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
  - [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html)
## TensorRT
  - `nvcc -V` 查看 CUDA 版本
  - **TensorRT** 只是 **推理优化器**，是对训练好的模型进行优化，当网络训练完之后，可以将训练模型文件直接丢进 tensorRT 中，而不再需要依赖深度学习框架 Caffe / TensorFlow
    - 可以认为 **tensorRT** 是一个 **只有前向传播的深度学习框架**
    - 可以将 Caffe / TensorFlow 的网络模型 **解析**，然后与 tensorRT 中对应的层进行一一 **映射**，把其他框架的模型统一转换到 tensorRT 中
    - 然后在 tensorRT 中可以针对 NVIDIA 自家 GPU 实施优化策略，并进行 **部署加速**
  - [Install](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)
    - Download and install `TensorRT deb` file, for `cuda==10.1 --> TensorRT==6.xxx`, for `cuda==10.2 --> TensorRT==7.xxx`
    ```sh
    sudo dpkg -i nv-tensorrt-repo-*1-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-repo-*/7fa2af80.pub

    sudo apt-get update
    sudo apt-get install tensorrt
    ```
    - Download and extract `tar` file for python installation
    ```sh
    export TRT_RELEASE=$HOME/local_bin/TensorRT-7.0.0.11
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local_bin/TensorRT-7.0.0.11/lib

    sudo apt install python3-libnvinfer-dev uff-converter-tf

    # Extract tar file
    cd TensorRT-7.0.0.11/python
    pip install tensorrt-7.0.0.11-cp37-none-linux_x86_64.whl
    cd ../uff/
    pip install uff-0.6.5-py2.py3-none-any.whl
    cd ../graphsurgeon/
    pip install graphsurgeon-0.4.1-py2.py3-none-any.whl
    ```
  - Compile `trtexec`
    ```sh
    cd $TRT_RELEASE/samples/trtexec/
    make
    ```
    编译完的文件位于 `$TRT_RELEASE/targets/x86_64-linux-gnu/bin/trtexec`
    ```sh
    export PATH=$PATH:$TRT_RELEASE/targets/x86_64-linux-gnu/bin
    ```
## ONNX tensorrt
  - [Github onnx-tensorrt](https://github.com/onnx/onnx-tensorrt.git)
  - **pybind11**
    ```sh
    # Needs pybind11 >= 2.2
    sudo apt install pybind11-dev # For 18.04, install version is 2.0.1
    git clone https://github.com/pybind/pybind11.git  # Install newest version pybind11 from git
    cd pybind11 && mkdir build && cd build && cmake .. && make && sudo make install
    ```
  - **cmake** [Download | CMake](https://cmake.org/download/) and extract tar file
    ```sh
    # Needs cmake >= 3.3.2
    sudo cp cmake-*-Linux-x86_64//* /usr/ -r
    cmake --version
    # cmake version 3.17.0
    ```
  - **Build**
    ```sh
    locate libnvonnxparser.so
    # /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
    # /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.7
    # /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.7.0.0

    git clone https://github.com/onnx/onnx-tensorrt.git
    cd onnx-tensorrt
    git submodule update --init --recursive
    mkdir build
    cd build

    # cmake .. -DCUDA_INCLUDE_DIRS=/usr/local/cuda/include -DTENSORRT_ROOT=/usr/src/tensorrt -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    cmake .. -DBUILD_ONNX_PYTHON=ON
    # --   BUILD_ONNX_PYTHON     : ON
    # --     Python version      :
    # --     Python executable   : /opt/anaconda3/bin/python3.7
    # --     Python includes     : /opt/anaconda3/include/python3.7m

    make && sudo make install
    ```
  - **Setup python library**
    ```sh
    # Run setup.py
    python setup.py install

    ''' Q:
    NvOnnxParser.h:213: Error: Syntax error in input(1).
    error: command 'swig' failed with exit status 1
    '''
    ''' A:
    $ vi NvOnnxParser.h
    33 + #define TENSORRTAPI
    '''
    ```
***

# Inference
## UFF MNIST
  - [Keras MNIST](09-06_tensorflow_tutotials.md#keras-mnist) Train a basic MNIST model
    ```py
    # Save trained model to h5
    model.save('aaa.h5')
    ```
  - [Keras h5 to pb](07-26_model_conversion.md#keras-h5-to-pb) save a frozen PB file
    ```py
    # Reload in tf 1.15.0
    model = keras.models.load_model('aaa.h5')
    save_frozen(model, 'aaa.pb')
    ```
  - Convert to `uff`
    ```sh
    convert-to-uff aaa.pb
    # UFF Output written to aaa.uff
    ```
  - **Build engine**
    ```py
    import tensorrt as trt
    import pycuda.driver as cuda
    # Or will throw LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
    import pycuda.autoinit

    model_file = './aaa.uff'
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        parser.register_input("flatten_input", (28, 28, 1), trt.UffInputOrder.NHWC)
        parser.register_output("dense_1/Softmax")
        parser.parse(model_file, network)
        # builder.int8_mode = True
        engine = builder.build_cuda_engine(network)
    ```
  - **Serialize**
    ```py
    # Serialize the model to a modelstream
    serialized_engine = engine.serialize()

    # Deserialize modelstream to perform inference. Deserializing requires creation of a runtime object
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)

    # Serialize the engine and write to a file
    with open("sample.engine", "wb") as f:
        f.write(engine.serialize())

    # Read the engine from the file and deserialize
    with open("sample.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    ```
  - **Inference**
    ```py
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test / 255.0, -1)

    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    context = engine.create_execution_context()
    # Transfer input data to the GPU.
    np.copyto(h_input, x_test[0].ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

    print("Prediction:", np.argmax(h_output), ", true label:", y_test[0])
    # Prediction: 7 , true label: 7
    ```
## UFF MNIST Official Sample Definition
  ```py
  import tensorrt as trt
  import pycuda.driver as cuda
  # Or will throw LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
  import pycuda.autoinit

  model_file = './aaa.uff'
  TRT_LOGGER = trt.Logger(trt.Logger.INFO)

  def build_engine(model_file):
      GiB = lambda n: n * 1 << 30
      with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
          builder.max_workspace_size = GiB(1)
          # Parse the Uff Network
          parser.register_input("flatten_input", (28, 28, 1), trt.UffInputOrder.NHWC)
          parser.register_output("dense_1/Softmax")
          parser.parse(model_file, network)
          # Build and return an engine.
          return builder.build_cuda_engine(network)

  # Simple helper data class that's a little nicer to use than a 2-tuple.
  class HostDeviceMem(object):
      def __init__(self, host_mem, device_mem):
          self.host = host_mem
          self.device = device_mem

      def __str__(self):
          return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

      def __repr__(self):
          return self.__str__()

  # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
  def allocate_buffers(engine):
      inputs = []
      outputs = []
      bindings = []
      stream = cuda.Stream()
      for binding in engine:
          size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
          dtype = trt.nptype(engine.get_binding_dtype(binding))
          # Allocate host and device buffers
          host_mem = cuda.pagelocked_empty(size, dtype)
          device_mem = cuda.mem_alloc(host_mem.nbytes)
          # Append the device buffer to device bindings.
          bindings.append(int(device_mem))
          # Append to the appropriate list.
          if engine.binding_is_input(binding):
              inputs.append(HostDeviceMem(host_mem, device_mem))
          else:
              outputs.append(HostDeviceMem(host_mem, device_mem))
      return inputs, outputs, bindings, stream

  # inputs and outputs are expected to be lists of HostDeviceMem objects.
  def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
      # Transfer input data to the GPU.
      [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
      # Run inference.
      context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
      # Transfer predictions back from the GPU.
      [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
      # Synchronize the stream
      stream.synchronize()
      # Return only the host outputs.
      return [out.host for out in outputs]

  import tensorflow as tf
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_test = np.expand_dims(x_test / 255.0, -1)

  engine = build_engine(model_file)
  # Build an engine, allocate buffers and create a stream.
  # For more information on buffer allocation, refer to the introductory samples.
  inputs, outputs, bindings, stream = allocate_buffers(engine)

  with engine.create_execution_context() as context:
      # case_num = load_normalized_test_case(data_paths, pagelocked_buffer=inputs[0].host)
      np.copyto(inputs[0].host, x_test[0].ravel())
      # For more information on performing inference, refer to the introductory samples.
      # The common.do_inference function will return a list of outputs - we only have one in this case.
      [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
      print("Prediction:", np.argmax(output), ", true label:", y_test[0])
      # Prediction: 7 , true label: 7
  ```
## ONNX Engine
  - In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the `explicitBatch` flag set
  - Currently most frameworks support `tf1.x` only, so better convert it under `tf1.x` environment
    ```py
    tf.__version__
    # '1.15.0'

    # Convert to saved model first
    model = tf.keras.models.load_model('aaa.h5', compile=False)
    tf.keras.experimental.export_saved_model(model, './aaa')
    ```
    `tf2onnx` convert `saved model` to `tflite`, also `tf1.15.0`
    ```sh
    pip install -U tf2onnx
    python -m tf2onnx.convert --saved-model ./aaa --output aaa.onnx
    ```
  - **Build engine**
    ```py
    import tensorrt as trt
    import pycuda.driver as cuda
    # Or will throw LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
    import pycuda.autoinit
    import os

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    def build_onnx_engine(model_file, max_workspace_size=-1):
        GiB = lambda n: n * 1 << 30
        engine_backup = os.path.splitext(model_file)[0] + '.engine'
        if os.path.exists(engine_backup):
            with open(engine_backup, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # Explicit batch size only
        # builder = trt.Builder(TRT_LOGGER)
        # network = builder.create_network(EXPLICIT_BATCH)
        # parser = trt.OnnxParser(network, TRT_LOGGER)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            if max_workspace_size != -1:
                builder.max_workspace_size = GiB(max_workspace_size)
            with open(model_file, 'rb') as model:
                parser.parse(model.read())
            if parser.num_errors != 0:
                print("Error:", parser.get_error(0))
                return
            input_shape = network.get_input(0).shape
            network.get_input(0).shape = [1] + list(input_shape[1:])  # Change dynamic batch_size to 1
            engine = builder.build_cuda_engine(network)

        # Serialize the engine and write to a file
        if engine is not None:
            with open(engine_backup, "wb") as f:
                f.write(engine.serialize())
        return engine

    engine = build_onnx_engine(model_file="./aaa.onnx")
    ```
  - **Inference**
    ```py
    class EngineInference:
        def __init__(self, engine):
            self.h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)

            self.h_outputs, self.d_outputs, self.output_shapes = [], [], []
            for ii in range(1, engine.num_bindings):
                h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(ii)), dtype=np.float32)
                self.h_outputs.append(h_output)
                self.d_outputs.append(cuda.mem_alloc(h_output.nbytes))
                self.output_shapes.append(engine.get_binding_shape(ii))
            self.output_num = len(self.h_outputs)
            self.stream = cuda.Stream()
            self.context = engine.create_execution_context()

        def __call__(self, img):
            np.copyto(self.h_input, img.ravel())
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            # Run inference.
            self.context.execute_async_v2(bindings=[self.d_input, *self.d_outputs], stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            # cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            rets = []
            for ii in range(self.output_num):
                cuda.memcpy_dtoh_async(self.h_outputs[ii], self.d_outputs[ii], self.stream)
                rets.append(self.h_outputs[ii].reshape(self.output_shapes[ii]))
            # Synchronize the stream
            self.stream.synchronize()
            return rets

    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test / 255.0, -1)

    aa = EngineInference(engine)
    output = aa(x_test[0])
    print("Prediction:", np.argmax(output), ", true label:", y_test[0])
    # Prediction: 7 , true label: 7
    ```
  - **Time compare**
    ```py
    model = keras.models.load_model('aaa.h5')
    model(x_test[:1])
    %timeit model(x_test[:1])
    # 806 µs ± 11.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    %timeit model(x_test[:10])
    # 840 µs ± 14.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    ```
    ```py
    %timeit aa(x_test[0])
    # 51.1 µs ± 616 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    %timeit [aa(ii) for ii in x_test[:10]]
    # 495 µs ± 1.22 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    ```
    ```py
    model = keras.models.load_model('resnet101.h5')
    imm = np.random.uniform(-1, 1, 112 * 112 * 3).reshape(1, 112, 112, 3)
    model(imm)
    %timeit model(imm)
    # 38.1 ms ± 625 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    engine = build_onnx_engine(model_file="./resnet101.onnx")
    aa = EngineInference(engine)
    imm = np.random.uniform(-1, 1, 112 * 112 * 3).reshape(112, 112, 3)
    aa(imm)
    %timeit aa(imm)
    # 9.46 ms ± 1.63 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    ```
## ONNX Engine With Optimization Profile
  - **Build engine**
    ```py
    import tensorrt as trt
    import pycuda.driver as cuda
    # Or will throw LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
    import pycuda.autoinit
    import os

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    def build_onnx_engine(model_file, max_batch_size=4, int8_mode=False, calib=None, max_workspace_size=-1):
        GiB = lambda n: n * 1 << 30
        engine_backup = os.path.splitext(model_file)[0] + '_max_batch_size_' + str(max_batch_size) +'.engine'
        if os.path.exists(engine_backup):
            with open(engine_backup, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # Explicit batch size only
        # builder = trt.Builder(TRT_LOGGER)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
            if max_workspace_size != -1:
                builder.max_workspace_size = GiB(max_workspace_size)
            builder.max_batch_size = max_batch_size
            with open(model_file, 'rb') as model:
                parser.parse(model.read())
            if parser.num_errors != 0:
                print("Error:", parser.get_error(0))
                return
            if int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib
            profile = builder.create_optimization_profile()
            input_shape = network.get_input(0).shape[1:]
            profile.set_shape(network.get_input(0).name, (1, *input_shape), (1, *input_shape), (builder.max_batch_size, *input_shape))
            config.add_optimization_profile(profile)

            # profile = builder.create_optimization_profile()
            # output_shape = network.get_output(0).shape[1:]
            # profile.set_shape(network.get_output(0).name, (1, *output_shape), (1, *output_shape), (builder.max_batch_size, *output_shape))
            # config.add_optimization_profile(profile)

            engine = builder.build_engine(network, config)
        # Serialize the engine and write to a file
        if engine is not None:
            with open(engine_backup, "wb") as f:
                f.write(engine.serialize())
        return engine
    ```
  - **Inference**
    ```py
    class EngineInference:
        def __init__(self, engine):
            max_inputs = [engine.max_batch_size, *engine.get_binding_shape(0)[1:]]
            dtype_input = trt.nptype(engine.get_binding_dtype(0))
            max_outputs = [engine.max_batch_size, *engine.get_binding_shape(1)[1:]]
            dtype_output = trt.nptype(engine.get_binding_dtype(1))

            self.h_input = cuda.pagelocked_empty(trt.volume(max_inputs), dtype=dtype_input)
            self.h_output = cuda.pagelocked_empty(trt.volume(max_outputs), dtype=dtype_output)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            self.stream = cuda.Stream()
            self.context = engine.create_execution_context()

            self.max_batch_size = engine.max_batch_size
            self.output_dim = max_outputs[1:]
            self.output_ravel_dim= self.h_output.shape[0] // engine.max_batch_size

        def __call__(self, imgs):
            batch_size = imgs.shape[0]
            if batch_size > self.max_batch_size:
                batch_size = self.max_batch_size
                imgs = imgs[:self.max_batch_size]
            inputs = imgs.ravel()
            self.context.set_binding_shape(0, imgs.shape)
            np.copyto(self.h_input[:inputs.shape[0]], inputs)
            cuda.memcpy_htod_async(self.d_input, self.h_input[:inputs.shape[0]], self.stream)
            # Run inference asynchronously, same function in cpp is `IExecutionContext::enqueueV2`
            self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(self.h_output[:batch_size * self.output_ravel_dim], self.d_output, self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            return self.h_output[:batch_size * self.output_ravel_dim].reshape([batch_size, *self.output_dim]).copy()

    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test / 255.0, -1)

    engine = build_onnx_engine(model_file="./aaa.onnx", max_batch_size=4)
    aa = EngineInference(engine)
    output = aa(x_test[:4])
    print("Prediction:", np.argmax(output, 1), ", true label:", y_test[:4])
    # Prediction: [7 2 1 0] , true label: [7 2 1 0]
    ```
  - **Time**
    ```py
    engine = build_onnx_engine(model_file="./aaa.onnx", max_batch_size=10)
    aa = EngineInference(engine)
    %timeit aa(x_test[:1])
    # 60.5 µs ± 692 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    %timeit aa(x_test[:10])
    # 87.1 µs ± 245 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    ```
    ```py
    engine = build_onnx_engine(model_file="./resnet101.onnx", max_batch_size=10)
    aa = EngineInference(engine)
    iaa = np.random.uniform(-1, 1, 112 * 112 * 3).reshape(1, 112, 112, 3)
    ibb = np.random.uniform(-1, 1, 10 * 112 * 112 * 3).reshape(10, 112, 112, 3)
    %timeit aa(iaa)
    # 19.2 ms ± 177 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    %timeit aa(ibb)
    # 45.3 ms ± 30.3 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    ```
## ONNX Engine With Multi Optimization Profile
  - **NOT working [ !!! ]**
  - **Build engine**
    ```py
    import tensorrt as trt
    import pycuda.driver as cuda
    # Or will throw LogicError: explicit_context_dependent failed: invalid device context - no currently active context?
    import pycuda.autoinit
    import os

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    def build_onnx_engine(model_file, max_batch_size=4, int8_mode=False, calib=None):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # Explicit batch size only
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = max_batch_size
            with open(model_file, 'rb') as model:
                parser.parse(model.read())
            assert parser.num_errors == 0
            if int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calib

            # for ii in range(max_batch_size, max_batch_size + 1):
            for ii in range(1, max_batch_size + 1):
                profile = builder.create_optimization_profile()
                input_shape = network.get_input(0).shape[1:]
                profile.set_shape(network.get_input(0).name, (ii, *input_shape), (ii, *input_shape), (ii, *input_shape))
                config.add_optimization_profile(profile)
                print((ii, *input_shape))

            return builder.build_engine(network, config)
    ```
  - **Inference**
    ```py
    [TensorRT] ERROR: ../rtSafe/cuda/genericReformat.cu (1262) - Cuda Error in executeMemcpy: 1 (invalid argument)
    [TensorRT] ERROR: FAILED_EXECUTION: std::exception
    ```
    ```py
    class EngineInference:
        def __init__(self, engine):
            basic_inputs = [1, *engine.get_binding_shape(0)[1:]]
            dtype_input = trt.nptype(engine.get_binding_dtype(0))
            basic_outputs = [1, *engine.get_binding_shape(1)[1:]]
            dtype_output = trt.nptype(engine.get_binding_dtype(1))
            max_batch_size = engine.max_batch_size

            self.h_input = cuda.pagelocked_empty(trt.volume(basic_inputs) * max_batch_size, dtype=dtype_input)
            self.h_output = cuda.pagelocked_empty(trt.volume(basic_outputs) * max_batch_size, dtype=dtype_output)
            self.d_input = cuda.mem_alloc(self.h_input.nbytes * 4)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes * 4)
            self.stream = cuda.Stream()
            # self.context = engine.create_execution_context()

            # self.max_batch_size = engine.max_batch_size
            self.max_batch_size = max_batch_size
            self.output_dim = basic_outputs[1:]
            self.output_ravel_shape = self.h_output.shape[0] // max_batch_size

        def __call__(self, imgs):
            batch_size = imgs.shape[0]
            if batch_size > self.max_batch_size:
                batch_size = self.max_batch_size
                imgs = imgs[:self.max_batch_size]
            inputs = imgs.ravel()
            self.context = engine.create_execution_context()
            self.context.active_optimization_profile = batch_size - 1
            print("batch_size = %d, active_optimization_profile = %d" % (batch_size, self.context.active_optimization_profile))
            self.context.set_binding_shape(2 * (batch_size - 1), imgs.shape)
            # self.context.set_binding_shape(0, imgs.shape)
            np.copyto(self.h_input[:inputs.shape[0]], inputs)
            cuda.memcpy_htod_async(self.d_input, self.h_input[:inputs.shape[0]], self.stream)
            # Run inference.
            self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(self.h_output[:batch_size * self.output_ravel_shape], self.d_output, self.stream)
            # Synchronize the stream
            self.stream.synchronize()
            return self.h_output[:batch_size * self.output_ravel_shape].reshape([batch_size, *self.output_dim])

    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test / 255.0, -1)

    engine = build_onnx_engine(model_file="./aaa.onnx", max_batch_size=4)
    for binding in engine:
        print(binding, ": ", engine.get_binding_shape(binding))

    aa = EngineInference(engine)
    output = aa(x_test[:4])
    print("Prediction:", np.argmax(output, 1), ", true label:", y_test[:4])
    # Prediction: [7 2 1 0] , true label: [7 2 1 0]
    ```
## ONNX Engine with INT8 Calibrator
  ```py
  import tensorrt as trt
  import pycuda.driver as cuda
  import pycuda.autoinit
  import os
  TRT_LOGGER = trt.Logger(trt.Logger.INFO)

  class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
      def __init__(self, data, cache_file, batch_size=32):
          # Whenever you specify a custom constructor for a TensorRT class,
          # you MUST call the constructor of the parent explicitly.
          trt.IInt8EntropyCalibrator2.__init__(self)

          self.cache_file = cache_file

          # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
          self.data = data
          self.batch_size = batch_size
          self.current_index = 0

          # Allocate enough memory for a whole batch.
          self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

      def get_batch_size(self):
          return self.batch_size

      # TensorRT passes along the names of the engine bindings to the get_batch function.
      # You don't necessarily have to use them, but they can be useful to understand the order of
      # the inputs. The bindings list is expected to have the same ordering as 'names'.
      def get_batch(self, names):
          if self.current_index + self.batch_size > self.data.shape[0]:
              return None

          current_batch = int(self.current_index / self.batch_size)
          if current_batch % 10 == 0:
              print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

          batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
          cuda.memcpy_htod(self.device_input, batch)
          self.current_index += self.batch_size
          return [self.device_input]

      def read_calibration_cache(self):
          # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
          if os.path.exists(self.cache_file):
              with open(self.cache_file, "rb") as f:
                  return f.read()

      def write_calibration_cache(self, cache):
          with open(self.cache_file, "wb") as f:
              f.write(cache)

  def build_int8_engine_fix(model_file, calib, batch_size=32):
      EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
      with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
          # We set the builder batch size to be the same as the calibrator's, as we use the same batches
          # during inference. Note that this is not required in general, and inference batch size is
          # independent of calibration batch size.
          # builder.max_batch_size = batch_size        
          builder.int8_mode = True
          builder.int8_calibrator = calib
          with open(model_file, 'rb') as model:
              parser.parse(model.read())
          assert parser.num_errors == 0
          input_shape = network.get_input(0).shape
          network.get_input(0).shape = [batch_size] + list(input_shape[1:])
          # Build engine and do int8 calibration.
          return builder.build_cuda_engine(network)

  '''
  Build engine with fixed batch_size first to generate `calibration_cache` file.
  This will throw 'WARNING: Explicit batch network detected and batch size specified, use execute without batch size instead.'
  Reason maybe using `context.execute_async` instead of `context.execute_async_v2` inside.
  '''
  import tensorflow as tf
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_test = np.expand_dims(x_test / 255.0, -1)
  calibration_cache = "mnist_calibration.cache"
  calib = MNISTEntropyCalibrator(x_test, cache_file=calibration_cache, batch_size=32)
  engine = build_int8_engine_fix(model_file="./aaa.onnx", calib=calib, batch_size=32)

  '''
  Then again, build engine with dynamic batch_size
  '''
  engine = build_onnx_engine(model_file="./aaa.onnx", max_batch_size=32, int8_mode=True, calib=calib)

  '''
  But time is not improved...
  '''
  %timeit aa(x_test[:1])
  # 57.1 µs ± 164 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
  %timeit aa(x_test[:10])
  # 86.6 µs ± 203 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
  ```
***

# ONNX tensorrt
  ```sh
  onnx2trt my_model.onnx -O "pass_1;pass_2;pass_3" -m my_model_optimized.onnx
  onnx2trt my_model.onnx -o my_engine.trt

  trtexec --explicitBatch --onnx=aaa.onnx --minShapes=input:1x28x28x3 --optShapes=input:1x28x28x3 --maxShapes=input:1x28x28x3 --saveEngine=aaa.engine

  trtexec --explicitBatch --onnx=onnx/centerface.onnx --minShapes=input:30x10x3x32x32 --optShapes=input:30x10x3x32x32 --maxShapes=input:30x10x3x32x32 --saveEngine=aaa.engine
  trtexec --explicitBatch --onnx=onnx/centerface.onnx --minShapes=input:30x10x3x32x32 --optShapes=input:30x10x3x32x32 --maxShapes=input:300x10x3x32x32 --saveEngine=bbb.engine
  trtexec --explicitBatch --onnx=onnx/centerface.onnx --saveEngine=aaa.engine --maxBatch=30
  ```
  ```py
  import onnx
  import onnx_tensorrt.backend as backend
  import numpy as np

  model = onnx.load("aaa.onnx")
  engine = backend.prepare(model, device='CUDA:0')
  input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
  output_data = engine.run(input_data)[0]
  print(output_data)
  print(output_data.shape)
  ```
***

# CPP Inference
  - **engineRT.cpp**
    ```java
    #include <fstream>
    #include <iostream>
    #include <stdio.h>
    #include <string.h>
    #include <vector>
    #include <cassert>
    #include "NvInfer.h"
    #include "cuda_runtime_api.h"
    #include "engineRT.h"

    #define CHECK(status) \
        do\
        {\
            auto ret = (status);\
            if (ret != 0)\
            {\
                std::cerr << "Cuda failure: " << ret << std::endl;\
                abort();\
            }\
        } while (0)

    #define DEVICE 0  // GPU id

    using namespace nvinfer1;

    using Severity = nvinfer1::ILogger::Severity;
    class Logger : public nvinfer1::ILogger {
    public:
        // Logger(Severity severity = Severity::kWARNING) : mReportableSeverity(severity) {}
        void log(Severity severity, const char* msg) override {
            // LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
            // suppress info-level messages
            if (severity != Severity::kINFO)
                std::cout << msg << std::endl;
        }
    } gLogger;

    class EngineInference {
    private:
        std::string engine_file;
        IRuntime* runtime;
        ICudaEngine* engine;
        IExecutionContext* context;
        cudaStream_t stream;

        void* d_input;
        void* d_output;
        void* bindings[2];

    public:
        int max_batch_size;
        int input_ravel_dim;
        int output_ravel_dim;
        int input_h;
        int input_w;

    public:
        EngineInference(const char * engine_file);
        ~EngineInference();
        void doInference(float* input, float* output, int batchSize);
        const char * getDataInfo();
    };

    EngineInference::EngineInference(const char * engine_file_c) {
        cudaSetDevice(DEVICE);
        // create a model using the API directly and serialize it to a stream
        char * trtModelStream{nullptr};
        size_t size{0};
        engine_file = std::string(engine_file_c);

        std::ifstream file(engine_file, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        assert(engine != nullptr);
        assert(engine->getNbBindings() > 1);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        delete[] trtModelStream;

        int inputIndex = 0;
        // auto dataType = engine->getBindingDataType(0);
        auto input_dims = engine->getBindingDimensions(inputIndex);
        input_ravel_dim = 1;
        for (int ii = 1; ii < input_dims.nbDims; ii++) {
            input_ravel_dim *= input_dims.d[ii];
        }
        if (input_dims.nbDims == 4) {
            input_h = input_dims.d[1];
            input_w = input_dims.d[2];
        }

        int output_num = engine->getNbBindings() - 1;
        int outputIndexes[output_num]:
        for (int ii = 0; ii < output_num; ii++) {
            outputIndexes[ii] = ii + 1;

        }


        auto output_dims = engine->getBindingDimensions(1);
        output_ravel_dim = 1;
        for (int ii = 1; ii < output_dims.nbDims; ii++) {
            output_ravel_dim *= output_dims.d[ii];
        }

        // Create GPU buffers on device
        max_batch_size = engine->getMaxBatchSize();
        CHECK(cudaMalloc(&d_input, max_batch_size * input_ravel_dim * sizeof(float)));
        CHECK(cudaMalloc(&d_output, max_batch_size * output_ravel_dim * sizeof(float)));
        bindings[0] = d_input;
        bindings[1] = d_output;

        CHECK(cudaStreamCreate(&stream));
    }

    EngineInference::~EngineInference() {
        cudaStreamDestroy(stream);
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        context->destroy();
        engine->destroy();
        runtime->destroy();
    }

    void EngineInference::doInference(float* input, float* output, int batchSize) {
        nvinfer1::Dims dynamicDims = context->getBindingDimensions(0);

        int temp_batchSize = 0;
        for (int ii = 0; ii < batchSize; ii += max_batch_size, input += max_batch_size * input_ravel_dim, output += max_batch_size * output_ravel_dim) {
            temp_batchSize = batchSize > ii + max_batch_size ? max_batch_size : batchSize - ii;
            // std::cout << temp_batchSize << std::endl;

            dynamicDims.d[0] = temp_batchSize;
            context->setBindingDimensions(0, dynamicDims);
            // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
            CHECK(cudaMemcpyAsync(d_input, input, temp_batchSize * input_ravel_dim * sizeof(float), cudaMemcpyHostToDevice, stream));
            context->enqueueV2(bindings, stream, nullptr);
            CHECK(cudaMemcpyAsync(output, d_output, temp_batchSize * output_ravel_dim * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }
    }

    // Pure c inference used for cgo or JNI
    EngineWrapper initEngine(const char * engine_file_c) {
        EngineInference *engine = new EngineInference(engine_file_c);
        return (void *) engine;
    }

    void releaseEngine(EngineWrapper engine_w) {
        EngineInference * engine = (EngineInference *) engine_w;
        delete engine;
    }

    void getFeature(EngineWrapper engine_w, float *data, int batch_size, float *feature) {
        EngineInference * engine = (EngineInference *) engine_w;
        engine->doInference(data, feature, batch_size);
    }

    int getDataInfo_int(EngineWrapper engine_w, const char * key) {
        EngineInference * engine = (EngineInference *) engine_w;
        if (strcmp("input_h", key) == 0) return engine->input_h;
        else if (strcmp("input_w", key) == 0) return engine->input_w;
        else if (strcmp("input_ravel_dim", key) == 0) return engine->input_ravel_dim;
        else if (strcmp("output_ravel_dim", key) == 0) return engine->output_ravel_dim;
        else if (strcmp("max_batch_size", key) == 0) return engine->max_batch_size;
        else return -1;
    }
    ```
  - **engineRT.h**
    ```java
    #ifndef _ENGINE_RT_H_
    #define _ENGINE_RT_H_

    #ifdef __cplusplus
    extern "C" {
    #endif
        typedef void* EngineWrapper;
        EngineWrapper initEngine(const char * engine_file_c);
        void releaseEngine(EngineWrapper engine_w);
        void getFeature(EngineWrapper engine_w, float *data, int batch_size, float *feature);
        int getDataInfo_int(EngineWrapper engine_w, const char * key);
    #ifdef __cplusplus
    }
    #endif
    #endif // _ENGINE_RT_H_
    ```
  - **test_engineRT.cpp**
    ```java
    #include <iostream>
    #include "engineRT.h"

    int main(int argc, char** argv) {
        EngineWrapper engine = initEngine("model.engine");

        auto input_h = getDataInfo_int(engine, "input_h");
        auto input_w = getDataInfo_int(engine, "input_w");
        auto output_size = getDataInfo_int(engine, "output_ravel_dim");
        auto max_batch_size = getDataInfo_int(engine, "max_batch_size");
        std::cout << "max_batch_size: " << max_batch_size << ", input_h: " << input_h << ", input_w: " << input_w << ", output_ravel_dim: " << output_size << std::endl;

        // prepare input data using true image
        int BATCH_SIZE = 1;
        float data[BATCH_SIZE * 3 * input_h * input_w];
        // do inference
        float feature[BATCH_SIZE * output_size];
        getFeature(engine, data, BATCH_SIZE, feature);

        // print the output, set to output_size to print all values
        for (int ii = 0; ii < BATCH_SIZE; ii++) {
            std::cout << ">>>> " << std::endl;
            for (int jj = 0; jj < output_size; jj++) {
                std::cout << feature[ii * output_size + jj] << ", ";
            }
            std::cout << std::endl;
        }

        // release
        releaseEngine(engine);
        return 0;
    }
    ```
  - **Compile and run**
    ```sh
    g++ test_engineRT.cpp engineRT.cpp -I/usr/local/cuda-10.2/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -lcudart -lnvinfer
    CUDA_VISIBLE_DEVICES='0' ./a.out
    ```
***
