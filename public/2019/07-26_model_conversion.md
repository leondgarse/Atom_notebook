# ___2019 - 07 - 26 Model Conversion___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 07 - 26 Model Conversion___](#2019-07-26-model-conversion)
  - [目录](#目录)
  - [MMDNN](#mmdnn)
  	- [安装](#安装)
  	- [基本命令](#基本命令)
  	- [mxnet 模型转化为 tensorflow](#mxnet-模型转化为-tensorflow)
  	- [加载模型用于迁移学习](#加载模型用于迁移学习)
  	- [MMdnn IR 层的表示方式](#mmdnn-ir-层的表示方式)
  - [ONNX](#onnx)
  	- [链接](#链接)
  	- [Tensorflow to ONNX](#tensorflow-to-onnx)
  	- [MXNet to ONNX](#mxnet-to-onnx)
  	- [PyTorch to ONNX](#pytorch-to-onnx)
  	- [ONNX to Caffe2](#onnx-to-caffe2)
  	- [ONNX to Tensorflow](#onnx-to-tensorflow)
  	- [PyTorch to TensorFlow](#pytorch-to-tensorflow)
  	- [TensorFlow to MXNet](#tensorflow-to-mxnet)
  	- [Caffe ONNX convert](#caffe-onnx-convert)
  	- [mxnet-model-server and ArcFace-ResNet100 (from ONNX model zoo)](#mxnet-model-server-and-arcface-resnet100-from-onnx-model-zoo)
  - [Tensorflow SavedModel 模型的保存与加载](#tensorflow-savedmodel-模型的保存与加载)
  	- [SavedModel 保存与加载](#savedmodel-保存与加载)
  	- [使用 SavedModelBuilder 保存模型](#使用-savedmodelbuilder-保存模型)
  	- [指定 Signature 保存模型](#指定-signature-保存模型)
  	- [SavedModel 中的 SignatureDefs 定义类型](#savedmodel-中的-signaturedefs-定义类型)
  	- [C++ 加载 SavedModel 模型](#c-加载-savedmodel-模型)
  	- [使用 keras 训练与保存模型 SavedModel](#使用-keras-训练与保存模型-savedmodel)
  - [TF 2.0 Beta 使用 SavedModel 格式](#tf-20-beta-使用-savedmodel-格式)
  	- [TensorFlow 2.0 安装](#tensorflow-20-安装)
  	- [保存与加载 keras 模型](#保存与加载-keras-模型)
  	- [SavedModel 文件结构](#savedmodel-文件结构)
  	- [tensorflow_model_server 使用模型启动服务](#tensorflowmodelserver-使用模型启动服务)
  	- [导出自定义模型](#导出自定义模型)
  	- [指定输出的接口功能 signature](#指定输出的接口功能-signature)
  	- [模型微调 Fine-tuning imported models](#模型微调-fine-tuning-imported-models)
  	- [Control flow in SavedModels](#control-flow-in-savedmodels)
  	- [Estimators 保存与加载 SavedModels](#estimators-保存与加载-savedmodels)
  	- [Insightface SavedModel Serving server](#insightface-savedmodel-serving-server)
  - [MMDNN 转化与 TensorFlow MTCNN](#mmdnn-转化与-tensorflow-mtcnn)
  	- [Insightface caffe MTCNN model to TensorFlow](#insightface-caffe-mtcnn-model-to-tensorflow)
  	- [MTCNN with all platforms](#mtcnn-with-all-platforms)
  	- [MTCNN pb 模型转化为 saved model](#mtcnn-pb-模型转化为-saved-model)
  	- [TF_1 加载 frozen MTCNN](#tf1-加载-frozen-mtcnn)
  	- [TF_2 加载 frozen MTCNN](#tf2-加载-frozen-mtcnn)
  - [TF_1 checkpoints](#tf1-checkpoints)
  	- [save and restore checkpoints models](#save-and-restore-checkpoints-models)
  	- [inspect_checkpoint](#inspectcheckpoint)
  	- [Insightface Checkpoints to SavedModel](#insightface-checkpoints-to-savedmodel)
  - [Keras h5 to pb](#keras-h5-to-pb)
  - [TF2 to TF1](#tf2-to-tf1)
  - [Pytorch](#pytorch)
  	- [Torch model inference](#torch-model-inference)
  	- [Save and load entire model](#save-and-load-entire-model)
  - [Replace UpSampling2D with Conv2DTranspose](#replace-upsampling2d-with-conv2dtranspose)
  	- [Conv2DTranspose output shape](#conv2dtranspose-output-shape)
  	- [Nearest interpolation](#nearest-interpolation)
  	- [Bilinear](#bilinear)
  	- [Clone model](#clone-model)

  <!-- /TOC -->
***

# MMDNN
## 安装
  - [microsoft/MMdnn](https://github.com/microsoft/MMdnn)
  - **安装**
    ```sh
    pip install mmdnn
    pip install -U git+https://github.com/Microsoft/MMdnn.git@master
    ```
## 基本命令
  - **模型可视化** [MMDNN Visualizer](http://mmdnn.eastasia.cloudapp.azure.com:8080/)
    ```sh
    mmdownload -f keras -n inception_v3
    mmtoir -f keras -w imagenet_inception_v3.h5 -o keras_inception_v3
    ```
    选择文件 keras_inception_v3.json
  - **mmdownload** 下载预训练好的模型
    ```py
    # 返回框架支持的模型
    mmdownload -f tensorflow

    # 下载指定的模型
    mmdownload -f tensorflow -n inception_v3
    ```
  - **mmvismeta** 可以使用 tensorboard 作为后端将计算图可视化
    ```py
    mmvismeta imagenet_inception_v3.ckpt.meta ./log
    ```
  - **mmtoir** 将模型转化为中间表达形式 IR (intermediate representation)，结果中的 json 文件用于可视化，proto / pb 用于描述网络结构模型，npy 用于保存网络数值参数
    ```py
    mmtoir -f tensorflow -n imagenet_inception_v3.ckpt.meta  -w inception_v3.ckpt --dstNode MMdnn_Output -o converted
    ```
  - **mmtocode** 将 IR 文件转化为指定框架下构造网络的原始代码，以及构建网络过程中用于设置权重的参数，结果生成一个 py 文件，与 npy 文件一起用于模型迁移学习或模型推断
    ```py
    mmtocode -f pytorch -n converted.pb -w converted.npy -d converted_pytorch.py -dw converted_pytorch.npy
    ```
  - **mmtomodel** 生成模型，可以直接使用对应的框架加载
    ```py
    mmtomodel -f pytorch -in converted_pytorch.py -iw converted_pytorch.npy -o converted_pytorch.pth
    ```
  - **mmconvert** 用于一次性转化模型，是 mmtoir / mmtocode / mmtomodel 三者的集成
    ```py
    mmconvert -sf tensorflow -in imagenet_inception_v3.ckpt.meta -iw inception_v3.ckpt --dstNode MMdnn_Output -df pytorch -om tf_to_pytorch_inception_v3.pth
    ```
  - **caffe alexnet -> tf tested**
    ```sh
    master branch with following scripts:
    $ python -m mmdnn.conversion._script.convertToIR -f caffe -d kit_imagenet -n examples/caffe/models/bvlc_alexnet.prototxt -w examples/caffe/models/bvlc_alexnet.caffemodel
    $ python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath kit_imagenet.pb --dstModelPath kit_imagenet.py -w kit_imagenet.npy
    $ python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n kit_imagenet.py -w kit_imagenet.npy --dump ./caffe_alexnet.ckpt
    Tensorflow file is saved as [./caffe_alexnet.ckpt], generated by [kit_imagenet.py] and [kit_imagenet.npy].
    ```
## mxnet 模型转化为 tensorflow
  - **Modify mmdnn**
    ```sh
    vi /opt/anaconda3/lib/python3.7/site-packages/mmdnn/conversion/tensorflow/saver.py
    # -1 import tensorflow as tf
    # +1 import tensorflow.compat.v1 as tf
    #
    # -6 tag_list = [tf.saved_model.tag_constants.SERVING]
    # +6 tag_list = [tf.saved_model.SERVING]
    ```
  - **Convert to generate `tf_resnet100.py`**
    ```sh
    cd model-r100-ii/
    # mmconvert -sf mxnet -in model-symbol.json -iw model-0000.params -df tensorflow -om resnet100 --dump_tag SERVING --inputShape 3,112,112
    mmtoir -f mxnet -n model-symbol.json -w model-0000.params -d resnet100 --inputShape 3,112,112
    mmtocode -f tensorflow --IRModelPath resnet100.pb --IRWeightPath resnet100.npy --dstModelPath tf_resnet100.py
    ```
  - **Modify `tf_resnet100.py`**
    ```sh
    vi tf_resnet100.py
    # +2 tf.compat.v1.disable_v2_behavior()
    #
    # -26 data            = tf.placeholder(...)
    # +26 data            = tf.compat.v1.placeholder(...)
    #
    # -267     pre_fc1_flatten = tf.contrib.layers.flatten(...)
    # -268     pre_fc1         = tf.layers.dense(...)
    # +267     pre_fc1_flatten = tf.compat.v1.layers.flatten(...)
    # +268     pre_fc1         = tf.compat.v1.layers.dense(...)
    ```
  - **Convert to savedmodel**
    ```sh
    mmtomodel -f tensorflow -in tf_resnet100.py -iw resnet100.npy -o tf_resnet100 --dump_tag SERVING
    ```
  - **Test**
    ```py
    from tensorflow.python.platform import gfile
    from tensorflow.python.util import compat
    from tensorflow.core.protobuf import saved_model_pb2

    with gfile.FastGFile('./saved_model.pb', 'rb') as ff:
        data = compat.as_bytes(ff.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)

    g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

    LOGDIR='./logdir'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()

    tensorboard --logdir ./logdir/
    ```
## 加载模型用于迁移学习
  - IR 文件转化为 tensorflow 模型
    ```sh
    # IR 文件转化为模型代码
    mmtocode -f tensorflow --IRModelPath converted.pb --IRWeightPath converted.npy --dstModelPath tf_inception_v3.py

    # 修改 is_train 为 True
    sed -i 's/is_train = False/is_train = True/' tf_inception_v3.py

    # IR 文件转化为 tensorflow 模型
    mmtomodel -f tensorflow -in tf_inception_v3.py -iw converted.npy -o tf_inception_v3 --dump_tag TRAINING
    ```
  - 模型加载
    ```py
    export_dir = "./tf_inception_v3"
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)

        x = sess.graph.get_tensor_by_name('input:0')
        y = sess.graph.get_tensor_by_name('xxxxxx:0')
        ......
        _y = sess.run(y, feed_dict={x: _x})
    ```
## MMdnn IR 层的表示方式
  - [IR 层的 proto 说明文件](https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/common/IR/graph.proto)
  - IR 文件包含以下几个部分
    - **GraphDef** NodeDef / version
    - **NodeDef** name / op / input / attr
    - **Attrvalue** list / type / shape / tensor
    - **Tensorshape**  dim
    - **LiteralTensor** type / tensor_shape / values
  - **Graph** 描述网络模型，内部包含若干 node，对应网络模型的层，node 中的 input 描述每一层之间的输入输出连接关系
  - **NodeDef**
    - **name** 本层 name，在 Graph 中唯一
    - **op** 本层算子，算子描述见链接，在算子描述文件中，算子 name 唯一
    - **input** 用于描述本层输入关系，各层之间的输入输出关系靠此成员描述
    - **attr** attr 成员可以是 listvalue / type / shape / tensor
      - **list** 存储 list 成员，如 list(int) / list(float) / list(shape) 等，**data** 保存数值数据，类型可以是 bytes / int64 / float / bool
      - **type** 描述数值类型
      - **shape** 描述 tensor 形状
      - **tensor** 各 node 之间传递的 tensor
  - **TensorShape** 描述 tensor 的维度信息
  - **LiteralTensor** 存储 tensor, 在各 node 之间传递
***

# ONNX
## 链接
  - [onnx/onnx](https://github.com/onnx/onnx)
  - [onnx/models](https://github.com/onnx/models)
  - [onnx/tutorials](https://github.com/onnx/tutorials)
  - [MMdnn/mmdnn/conversion/onnx/](https://github.com/microsoft/MMdnn/tree/master/mmdnn/conversion/onnx)
  - [Importing an ONNX model into MXNet](http://mxnet.incubator.apache.org/versions/master/tutorials/onnx/super_resolution.html)
  - ONNX为AI模型提供了一个开源格式。 它定义了一个可扩展的计算图模型，以及内置运算符和标准数据类型的定义，MMdnn也将支持ONNX格式
  ```sh
  pip install onnx
  ```
## Tensorflow to ONNX
  - [Train in Tensorflow, Export to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowExport.ipynb)
  - **tf2onnx**
    - `1.5.6` 支持 `TF1` + `--saved-model`
    - `1.6.3` 支持 `TF2` + `--saved-model` / `--keras`
    - `--opset` 设置 onnx 模式使用的操作数 opset 版本
    ```sh
    pip install -U tf2onnx

    python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx --opset 8
    python -m tf2onnx.convert --keras aa.h5 --output model.onnx --opset 8

    python -m tf2onnx.convert --input frozen_graph.pb  --inputs X:0 --outputs output:0 --output model.onnx
    python -m tf2onnx.convert --checkpoint checkpoint.meta  --inputs X:0 --outputs output:0 --output model.onnx
    ```
    ```py
    model = keras.applications.ResNet50(weights='imagenet')
    imgs = np.random.uniform(size=[1, 224, 224, 3]).astype('float32')
    preds = model(imgs)

    """ Convert """
    import tf2onnx
    spec = (tf.TensorSpec(model.input_shape, tf.float32, name="input"),)
    output_path = model.name + ".onnx"
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

    """ Run test """
    import onnxruntime as rt
    output_names = [n.name for n in model_proto.graph.output]
    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(output_path, providers=providers)
    onnx_pred = m.run(output_names, {"input": imgs})

    print(np.allclose(preds, onnx_pred[0], rtol=1e-5))
    ```
  - [keras2onnx](https://github.com/onnx/keras-onnx)
    ```py
    ! pip install keras2onnx
    import keras2onnx
    mm = keras.applications.MobileNet(include_top=True, weights='imagenet')
    onnx_model = keras2onnx.convert_keras(mm, mm.name)
    keras2onnx.save_model(onnx_model, 'mm.onnx')
    ```
  - **ONNX Inference**
    ```py
    import onnxruntime
    import onnx

    onnx_model = onnx.load('mm.onnx')
    content = onnx_model.SerializeToString()
    sess = onnxruntime.InferenceSession(content)
    feed = dict([(input.name, np.ones([1, 224, 224, 3], dtype='float32')) for n, input in enumerate(sess.get_inputs())])
    pred_onnx = sess.run(None, feed)[0]

    pp = mm(np.ones([1, 224, 224, 3])).numpy()
    print(pred_onnx.shape, pp.shape, np.allclose(pp, pred_onnx, atol=1e-5))
    # (1, 1000) (1, 1000) True
    ```
## MXNet to ONNX
  - [Exporting to ONNX format](https://mxnet.apache.org/api/python/docs/tutorials/deploy/export/onnx.html)
  - `MXNet-ONNX` 使用 `ONNX opset == 7`，对应 `ONNX == 1.2.1`
  - **Convert to ONNX**
    ```py
    import mxnet as mx
    path='http://data.mxnet.io/models/imagenet/'
    [mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
     mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
     mx.test_utils.download(path+'synset.txt')]

    #make sure to install onnx-1.2.1
    #pip install onnx==1.2.1
    import onnx
    assert onnx.__version__=='1.2.1'
    from mxnet.contrib import onnx as onnx_mxnet

    # Invoke export model API. It returns path of the converted onnx model
    converted_model_path = onnx_mxnet.export_model('./resnet-18-symbol.json', './resnet-18-0000.params', [(1, 3, 224, 224)], np.float32, "mxnet_exported_resnet18.onnx")
    ```
  - **MXNet load onnx model**
    ```py
    sym, arg, aux = onnx_mxnet.import_model('./mxnet_exported_resnet18.onnx')
    # all_layers = sym.get_internals()
    # sym = all_layers["softmax"]
    ctx = mx.cpu()

    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[("data", (1, 3, 112, 112))])
    model.set_params(arg, aux)

    model.forward(mx.io.DataBatch(data=(mx.nd.ones([1, 3, 224, 224]),)), is_train=False)
    aa = model.get_outputs()[0]
    print(aa.shape, aa.argmax(1))
    # (1, 1000) [111.]
    ```
## PyTorch to ONNX
  - [TORCH.ONNX](https://pytorch.org/docs/stable/onnx.html)
  - **Convert**
    ```py
    import torch
    import torchvision
    model = torchvision.models.alexnet(pretrained=True).cpu()
    dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=False, keep_initializers_as_inputs=True)
    ```
  - **Caffe2 inference ONNX**
    ```py
    import caffe2.python.onnx.backend as backend
    import onnx

    onnx_model = onnx.load('alexnet.onnx')
    rep = backend.prepare(onnx_model, device="CPU")
    x = torch.randn(10, 3, 224, 224)
    W = {onnx_model.graph.input[0].name: x.data.numpy()}
    outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))[0]
    print(outputs.shape, outputs.argmax(1))
    # (10, 1000) [533 735 474 735 735 735 735 735 735 533]

    import torch
    torch.save(model, 'tt')
    ```
    ```py
    import onnxruntime as ort
    ort_session = ort.InferenceSession('alexnet.onnx')
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: np.random.randn(10, 3, 224, 224).astype(np.float32)})
    print(outputs[0])
    ```
## ONNX to Caffe2
  - **Save caffe2 format**
    ```py
    import caffe2.python.onnx.backend as onnx_caffe2_backend
    import onnx
    model = onnx.load("model.onnx")

    init_net, predict_net = onnx_caffe2_backend.Caffe2Backend.onnx_graph_to_caffe2_net(model)

    with open("onnx-init.pb", "wb") as f:
        f.write(init_net.SerializeToString())
    with open("onnx-predict.pb", "wb") as f:
        f.write(predict_net.SerializeToString())

    with open("onnx-init.pbtxt", "w") as f:
        f.write(str(init_net))
    with open("onnx-predict.pbtxt", "w") as f:
        f.write(str(predict_net))
    ```
  - **Caffe2 mobile format**
    ```py
    # extract the workspace and the model proto from the internal representation
    c2_workspace = prepared_backend.workspace
    c2_model = prepared_backend.predict_net

    # Now import the caffe2 mobile exporter
    from caffe2.python.predictor import mobile_exporter

    # call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

    # Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
    with open('init_net.pb', "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open('predict_net.pb', "wb") as fopen:
        fopen.write(predict_net.SerializeToString())
    ```
## ONNX to Tensorflow
  - [Can not use converted ONNX -> TF graph independently ](https://github.com/onnx/onnx-tensorflow/issues/167)
  - [Github onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) needs `tf == 1.15.0`
    ```sh
    pip install onnx-tf
    onnx-tf convert -i /path/to/input.onnx -o /path/to/output.pb
    ```
    ```py
    tf.__version__
    # '1.15.0'
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load("input_path")  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    print(tf_rep.inputs, tf_rep.outputs)
    tf_rep.run(np.ones([1, 3, 256, 160]))

    tf_rep.export_graph("output_path")  # export the model
    ```
  - **Inference**
    ```py
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load("input_path")  # load onnx model
    output = prepare(onnx_model).run(input)  # run the loaded model
    ```
  - [Github onnx2keras](https://github.com/nerox8664/onnx2keras) seems
    ```py
    ! pip install onnx2keras
    ```
    ```py
    import onnx
    from onnx2keras import onnx_to_keras

    # Load ONNX model
    onnx_model = onnx.load('mxnet_exported_resnet18.onnx')
    k_model = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name])
    ```
## PyTorch to TensorFlow
  ```py
  import numpy as np
  import torch
  from pytorch2keras.converter import pytorch_to_keras
  from torch.autograd import Variable
  import tensorflow as tf
  from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


  # Create and load model
  model = Model()
  model.load_state_dict(torch.load('model-checkpoint.pth'))
  model.eval()

  # Make dummy variables (and checking if the model works)
  input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
  input_var = Variable(torch.FloatTensor(input_np))
  output = model(input_var)

  # Convert the model!
  k_model = pytorch_to_keras(model, input_var, (3, 224, 224), verbose=True, name_policy='short', change_ordering=True)

  # Save model to SavedModel format
  tf.saved_model.save(k_model, "./models")

  # Convert Keras model to ConcreteFunction
  full_model = tf.function(lambda x: k_model(x))
  full_model = full_model.get_concrete_function(tf.TensorSpec(k_model.inputs[0].shape, k_model.inputs[0].dtype))

  # Get frozen ConcreteFunction
  frozen_func = convert_variables_to_constants_v2(full_model)
  frozen_func.graph.as_graph_def()

  print("-" * 50)
  print("Frozen model layers: ")
  for layer in [op.name for op in frozen_func.graph.get_operations()]:
      print(layer)

  print("-" * 50)
  print("Frozen model inputs: ")
  print(frozen_func.inputs)
  print("Frozen model outputs: ")
  print(frozen_func.outputs)

  # Save frozen graph from frozen ConcreteFunction to hard drive
  tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./frozen_models", name="frozen_graph.pb", as_text=False)
  ```
## TensorFlow to MXNet
  - 使用 `keras-mxnet` 将 `Keras h5` 转化为 `MXNet param + json`
    ```sh
    $ pip install keras-mxnet
    $ KERAS_BACKEND='mxnet' ipython
    ```
  - **[Issue](https://github.com/awslabs/keras-apache-mxnet/pull/258)**
    ```py
    ''' Q: TypeError: tuple indices must be integers or slices, not list
    /opt/anaconda3/lib/python3.7/site-packages/keras/layers/normalization.py in build(self, input_shape)
         98
         99     def build(self, input_shape):
    --> 100         dim = input_shape[self.axis]
        101         print(input_shape, self.axis, dim)
        102         if dim is None
    '''
    ''' A: Modify normalization.py
    $ vi /opt/anaconda3/lib/python3.7/site-packages/keras/layers/normalization.py + 97
        else:
    -       self.axis = axis
    +       self.axis = axis if isinstance(axis, int) else axis[-1]

    def build(self, input_shape):
    '''
    ```
  - 将 `Keras h5` 模型转化为 `TF 1.13` [TF15 to TF13](#tf15-to-tf13)
    ```py
    # tf save
    mm = tf.keras.models.load_model("checkpoints/keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5", compile=False)
    json_config = mm.to_json()
    with open('model/model_config.json', 'w') as json_file:
        json_file.write(json_config)
    mm.save_weights("model/weights_only.h5")

    ''' Modify json file '''
    # For tf15 / tf20 saved json file, delete '"ragged": false,'
    !sed -i 's/"ragged": false, //' model/model_config.json
    # For tf-nightly saved json file, also replace '"class_name": "Functional"' by '"class_name": "Model"'
    !sed -i 's/"class_name": "Functional"/"class_name": "Model"/' model/model_config.json
    # For tf23 saved json file, delete '"groups": 1, '
    !sed -i 's/"groups": 1, //g' model/model_config.json
    ```
  - **keras-mxnet 模型转化** `KERAS_BACKEND='mxnet' ipython`
    ```py
    # mxnet load
    import numpy as np
    import keras
    # Using MXNet backend
    # from keras import backend as K
    # K.common.set_image_data_format('channels_first')
    from keras.initializers import glorot_normal, glorot_uniform
    from keras.utils import CustomObjectScope

    with open('model/model_config.json') as json_file:
        json_config = json_file.read()
    with CustomObjectScope({'GlorotNormal': glorot_normal(), "GlorotUniform": glorot_uniform()}):
        new_model = keras.models.model_from_json(json_config)
    new_model.load_weights('model/weights_only.h5')

    new_model.predict(np.zeros((1, 112, 112, 3))) # MUST do a predict
    # new_model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)
    new_model.compiled = True
    keras.models.save_mxnet_model(model=new_model, prefix='mm')
    ```
  - **Test**
    ```py
    import numpy as np
    import mxnet as mx

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='mm', epoch=0)
    mod = mx.mod.Module(symbol=sym, data_names=['/input_11'], context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('/input_11', (1, 112, 112, 3))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    data_iter = mx.io.NDArrayIter(np.ones((1, 112, 112, 3)), None, 1)
    mod.predict(data_iter)
    ```
## Caffe ONNX convert
  ```py
  import caffe
  deploy = './model/MobileNetV2.prototxt'
  net = caffe.Net(deploy, caffe.TEST)

  import convertCaffe
  onnx_path = './model/MobileNetV2.onnx'
  prototxt_path, caffemodel_path = "./model/MobileNetV2.prototxt", "./model/MobileNetV2.caffemodel"
  graph = convertCaffe.getGraph(onnx_path)
  net = convertCaffe.convertToCaffe(graph, prototxt_path, caffemodel_path)
  ```
## mxnet-model-server and ArcFace-ResNet100 (from ONNX model zoo)
  - [ArcFace-ResNet100 (from ONNX model zoo)](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#arcface-resnet100_onnx)
  - [onnx/models/vision/body_analysis/arcface/](https://github.com/onnx/models/tree/master/vision/body_analysis/arcface)
  - [awslabs/mxnet-model-server](ttps://github.com/awslabs/mxnet-model-server)
  ```sh
  pip install mxnet-model-server
  mxnet-model-server --start --models arcface=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-arcface-resnet100.mar

  curl -O https://s3.amazonaws.com/model-server/inputs/arcface-input1.jpg

  curl -O https://s3.amazonaws.com/model-server/inputs/arcface-input2.jpg

  curl -X POST http://127.0.0.1:8080/predictions/arcface -F "img1=@arcface-input1.jpg" -F "img2=@arcface-input2.jpg"

  mxnet-model-server --stop
  ```
***

# Tensorflow SavedModel 模型的保存与加载
## SavedModel 保存与加载
  - **SavedModel** 是一种独立于语言且可恢复的序列化格式，使较高级别的系统和工具可以创建、使用和转换 TensorFlow 模型，包含完整的 TensorFlow 程序，包括权重和计算图，可以使用 python 训练模型，然后在 Java 中非常方便的加载模型
  - TensorFlow 提供了多种与 SavedModel 交互的方式，包括 `tf.saved_model API` / `tf.estimator.Estimator` 和命令行界面，如使用 `TensorFlow Serving` 将训练好的模型部署至生产环境
  - 一个比较完整的 SavedModel 模型文件夹包含以下内容
    ```sh
    ├── assets  # 可选，可以添加可能需要的外部文件
    ├── assets.extra  # 可选，是一个库，可以添加其特定assets的地方
    ├── saved_model.pb  # 是 MetaGraphDef，包含图形结构，MetaGraphDef 是MetaGraph的Protocol Buffer表示
    └── variables # 保存训练所习得的权重
        ├── variables.data-00000-of-00001
        └── variables.index
    ```
  - **tf.saved_model.simple_save** 简单保存，创建 SavedModel 的最简单方法
    ```py
    tf.saved_model.simple_save(sess, "./model", inputs={"myInput": x, "Input_2": z}, outputs={"myOutput": y})
    ```
  - **加载** 加载后作为特定 MetaGraphDef 的一部分提供的变量、资源和签名子集将恢复到提供的会话中
    ```py
    # Loads the model from a SavedModel as specified by tags. (deprecated)
    # This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.
    load(sess, tags, export_dir, import_scope=None, **saver_kwargs)
    ```
    ```py
    export_dir = ...
    ...
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
      ...
    ```
  - **MNIST 示例**
    ```py
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, ["serve"], "./model")
      graph = tf.get_default_graph()

      input = np.expand_dims(mnist.test.images[0], 0)
      x = sess.graph.get_tensor_by_name('myInput:0')
      y = sess.graph.get_tensor_by_name('myOutput:0')
      batch_xs, batch_ys = mnist.test.next_batch(1)
      scores = sess.run(y,
               feed_dict={x: batch_xs})
      print("predict: %d, actual: %d" % (np.argmax(scores, 1), np.argmax(batch_ys, 1)))    
    ```
  - 在 **TensorFlow Serving** 中加载和提供 SavedModel
    ```sh
    tensorflow_model_server --port=port-numbers --model_name=your-model-name --model_base_path=your_model_base_path
    ```
  - **simple_save** 配置 SavedModel 的模型能够通过 `TensorFlow Serving` 进行加载，并支持 `Predict API`，要访问 `classify API` / `regress API` / `multi-inference API`，需要使用 builder API 或 tf.estimator.Estimator 手动构建 SavedModel
## 使用 SavedModelBuilder 保存模型
  - **tf.saved_model.builder.SavedModelBuilder** 构造 SavedModelBuilder 对象，提供了保存多个 MetaGraphDef 的功能，初始化方法只需要传入用于保存模型的目录名，目录不用预先创建
    ```py
    class tf.saved_model.builder.SavedModelBuilder
    __init__(export_dir)
    ```
    - **MetaGraph** 是一种数据流图，并包含相关变量、资源和签名
    - **MetaGraphDef** 是 MetaGraph 的协议缓冲区表示法
    - **签名** 是一组与图有关的输入和输出
  - **add_meta_graph_and_variables** 方法导入graph的信息以及变量，这个方法假设变量都已经初始化好了，对于每个 SavedModelBuilder 一定要执行一次用于导入第一个meta graph
    ```py
    # 导入graph与变量信息
    add_meta_graph_and_variables(
            self, sess, tags, signature_def_map=None, assets_collection=None,
            legacy_init_op=None, clear_devices=False, main_op=None,
            strip_default_attrs=False, saver=None)
    ```
    - **sess** 当前的session，包含 graph 的结构与所有变量
    - **tags** 给当前需要保存的 meta graph 一个标签，在载入模型的时候，需要根据这个标签名去查找对应的 MetaGraphDef，找不到会报 RuntimeError
    - **signature_def_map** 定义模型的 Signature，指定模型的输入 / 输出 tensor 等
    - 通过 `strip_default_attrs=True` 确保向前兼容性
  - 必须使用用户指定的 **标签** 对每个添加到 SavedModel 的 MetaGraphDef 进行标注
    - 标签提供了一种方法来识别要加载和恢复的特定 MetaGraphDef，以及共享的变量和资源子集
    - 标签一般会标注 MetaGraphDef 的功能（例如服务或训练），有时也会标注特定的硬件方面的信息（如 GPU）
    - 标签可以选用系统定义好的参数，如 `tf.saved_model.tag_constants.SERVING` 与 `tf.saved_model.tag_constants.TRAINING`
  - **save** 将模型序列化到指定目录底下，保存好以后，目录下会有一个 saved_model.pb 文件以及 variables 文件夹
    ```py
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
    builder.add_meta_graph_and_variables(sess, ['tag_string'])
    builder.save()
    ```
  - 使用 SavedModelBuilder 构建 SavedModel 的典型方法
    ```py
    export_dir = ...
    ...
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      ...
      builder.add_meta_graph_and_variables(sess,
                                           [tag_constants.TRAINING],
                                           signature_def_map=foo_signatures,
                                           assets_collection=foo_assets,
                                           strip_default_attrs=True)
    ...
    # Add a second MetaGraphDef for inference.
    with tf.Session(graph=tf.Graph()) as sess:
      ...
      builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
    ...
    builder.save()
    ```
  - **tf.saved_model.loader.load** 载入模型
    ```py
    tf.saved_model.loader.load(sess, tags, export_dir, import_scope=None, **saver_kwargs)
    ```
    ```py
    meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], saved_model_dir)
    ```
  - load 完以后，可以从 sess 对应的 graph 中获取需要的 tensor 来 inference
    ```py
    x = sess.graph.get_tensor_by_name('input_x:0')
    y = sess.graph.get_tensor_by_name('predict_y:0')

    # 实际的待inference的样本
    _x = ...
    sess.run(y, feed_dict={x: _x})
    ```
## 指定 Signature 保存模型
  - add_meta_graph_and_variables 的参数可以指定博阿村模型时的 Signature
    ```py
    tf.saved_model.signature_def_utils.build_signature_def(inputs=None, outputs=None, method_name=None)

    # 构建tensor info
    tf.saved_model.utils.build_tensor_info(tensor)
    ```
    - inputs / outputs 都是 dict，key 是约定的输入输出别名，value 是对具体 tensor 包装得到的 TensorInfo
  - 典型用法
    ```py
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
    # x 为输入tensor, keep_prob为dropout的prob tensor
    inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x),
                'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)}

    # y 为最终需要的输出结果tensor
    outputs = {'output' : tf.saved_model.utils.build_tensor_info(y)}

    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')

    builder.add_meta_graph_and_variables(sess, ['test_saved_model'], {'test_signature':signature})
    builder.save()
    ```
    - add_meta_graph_and_variables 的 signature_def_map 参数接收的是一个 dict，key 是自己命名的 signature 名称，value 是 SignatureDef 对象
  - **模型载入**
    ```py
    ## 略去构建sess的代码

    signature_key = 'test_signature'
    input_key = 'input_x'
    output_key = 'output'

    meta_graph_def = tf.saved_model.loader.load(sess, ['test_saved_model'], saved_model_dir)
    # 从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    # 从signature中找出具体输入输出的tensor name
    x_tensor_name = signature[signature_key].inputs[input_key].name
    y_tensor_name = signature[signature_key].outputs[output_key].name

    # 获取tensor 并inference
    x = sess.graph.get_tensor_by_name(x_tensor_name)
    y = sess.graph.get_tensor_by_name(y_tensor_name)

    # _x 实际输入待inference的data
    sess.run(y, feed_dict={x:_x})
    ```
## SavedModel 中的 SignatureDefs 定义类型
  - **SignatureDefs** 定义函数的输入输出，在使用 SavedModel 时可以指定
  - SignatureDef 结构
    - **inputs** TensorInfo 的字典格式
    - **outputs** TensorInfo 的字典格式
    - **method_name** 用于加载时的方法名称
    - TensorInfo 包含 tensor 名称 name / 类型 device_type / 维度 / shape 等信息
  - **Classification SignatureDef** 支持 TensorFlow Serving 的分类 API 调用，其中 Input tensor 是必需的，两个 output Tensors 至少有一个
    ```py
    signature_def: {
      key  : "my_classification_signature"
      value: {
        inputs: {
          key  : "inputs"
          value: {
            name: "tf_example:0"
            dtype: DT_STRING
            tensor_shape: ...
          }
        }
        outputs: {
          key  : "classes"
          value: {
            name: "index_to_string:0"
            dtype: DT_STRING
            tensor_shape: ...
          }
        }
        outputs: {
          key  : "scores"
          value: {
            name: "TopKV2:0"
            dtype: DT_FLOAT
            tensor_shape: ...
          }
        }
        method_name: "tensorflow/serving/classify"
      }
    }
    ```
  - **Predict SignatureDef** 支持 TensorFlow Serving 的预测 API 调用，可以使用任意数量的 input / output tensor，其中 output 还可以添加额外的 tensor
    ```py
    signature_def: {
      key  : "my_prediction_signature"
      value: {
        inputs: {
          key  : "images"
          value: {
            name: "x:0"
            dtype: ...
            tensor_shape: ...
          }
        }
        outputs: {
          key  : "scores"
          value: {
            name: "y:0"
            dtype: ...
            tensor_shape: ...
          }
        }
        method_name: "tensorflow/serving/predict"
      }
    }
    ```
  - **Regression SignatureDef** 支持 TensorFlow Serving 的回归 API 调用，必须只有一个 input / output tensor
    ```py
    signature_def: {
      key  : "my_regression_signature"
      value: {
        inputs: {
          key  : "inputs"
          value: {
            name: "x_input_examples_tensor_0"
            dtype: ...
            tensor_shape: ...
          }
        }
        outputs: {
          key  : "outputs"
          value: {
            name: "y_outputs_0"
            dtype: DT_FLOAT
            tensor_shape: ...
          }
        }
        method_name: "tensorflow/serving/regress"
      }
    }
    ```
## C++ 加载 SavedModel 模型
  - **C++ 加载模型** SavedModel 加载后的版本称为 SavedModelBundle，其中包含 MetaGraphDef 和加载时所在的会话。
    ```c
    const string export_dir = ...
    SavedModelBundle bundle;
    ...
    LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain}, &bundle);
    ```
## 使用 keras 训练与保存模型 SavedModel
  - **在 Fashion MNIST 上训练 keras 分类器**
    ```py
    from tensorflow import keras

    # Error in 1.14: could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
    # set_session is removed in tf 2.0
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    fasion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    # (60000, 28, 28, 1) (60000,) (10000, 28, 28, 1) (10000,)

    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=3, strides=2, activation='relu', name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation=tf.nn.softmax, name="Softmax")
    ])
    model.summary()

    testing = False
    epochs = 5
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('\nTest accuracy: {}'.format(test_acc))
    # Test accuracy: 0.8719000220298767
    ```
  - **simple_save 保存 SavedModel**
    ```py
    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors,
    # and stored with the default serving key
    import tempfile

    MODEL_DIR = tempfile.mktemp()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))
    # export_path = /tmp/tmp36fw8qod/1

    if os.path.isdir(export_path):
      print('\nAlready saved a model, cleaning up\n')
      !rm -r {export_path}

    tf.saved_model.simple_save(
        keras.backend.get_session(),
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})

    print('\nSaved model:')
    !ls -l {export_path}
    # Saved model:
    # total 72
    # -rw-r--r-- 1 leondgarse leondgarse 65984 八月 13 12:00 saved_model.pb
    # drwxr-xr-x 2 leondgarse leondgarse  4096 八月 13 12:00 variables
    ```
  - **Export keras model to saved_model**
    ```py
    tf.__version__
    # '1.13.0'
    tf.contrib.saved_model.save_keras_model(mm, 'aa')
    ```
    ```py
    tf.__version__
    # '2.3.0'
    tf.keras.experimental.export_saved_model(mm, './saved_model')
    ```
  - **模型测试与 serving**
    ```py
    !saved_model_cli show --dir {export_path} --all

    %%bash --bg
    nohup tensorflow_model_server \
      --rest_api_port=8501 \
      --model_name=fashion_model \
      --model_base_path="${MODEL_DIR}" >server.log 2>&1
    ```
***

# TF 2.0 Beta 使用 SavedModel 格式
## TensorFlow 2.0 安装
    ```sh
    conda create -n tf-20 python=3.7
    conda activate tf-20

    pip install tensorflow==2.0.0-beta1
    pip install tensorflow-gpu==2.0.0-beta1

    conda install ipython
    conda install pandas matplotlib pillow

    ipython
    ```
    ```py
    tf.__version__
    # Out[1]: '2.0.0-beta1'
    ```
## 保存与加载 keras 模型
  - **获取 keras mobilenet 模型**
    ```py
    file = tf.keras.utils.get_file(
        "grace_hopper.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
    img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
    plt.imshow(img)
    plt.axis('off')

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(
        x[tf.newaxis,...])
    ```
    ![](images/tf_serve_grace_hopper.jpg)
  - **模型使用与保存 tf.saved_model.save**
    ```py
    #tf.keras.applications.vgg19.decode_predictions
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    pretrained_model = tf.keras.applications.MobileNet()
    result_before_save = pretrained_model(x)
    decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]
    print("Result before saving:", decoded)
    # Result before saving:
    #  ['military uniform' 'bow tie' 'suit' 'bearskin' 'pickelhaube']

    tf.saved_model.save(pretrained_model, "/tmp/mobilenet/1/")
    ```
    其中保存模型的路径需要有一个版本号 `/1`，用于 Tensorflow Serving 加载
  - **saved_model_cli show 命令** 查看 pb 模型，saved_model 的 `signatures` 方法保存了模型信息，keras 将前向传输过程保存在 `serving_default` 签名键下
    ```py
    !saved_model_cli show --dir /tmp/mobilenet/1 --tag_set serve --signature_def serving_default
    # The given SavedModel SignatureDef contains the following input(s):
    #   inputs['input_1'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: (-1, 224, 224, 3)
    #       name: serving_default_input_1:0
    # The given SavedModel SignatureDef contains the following output(s):
    #   outputs['act_softmax'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: (-1, 1000)
    #       name: StatefulPartitionedCall:0
    # Method name is: tensorflow/serving/predict
    ```
  - **模型加载 tf.saved_model.load**
    ```py
    loaded = tf.saved_model.load("/tmp/mobilenet/1/")
    print(list(loaded.signatures.keys()))
    # ["serving_default"]

    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    # {'act_softmax': TensorSpec(shape=(None, 1000), dtype=tf.float32, name='act_softmax')}

    labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]
    decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]
    print("Result after saving and loading:\n", decoded)
    # Result after saving and loading:
    #  ['military uniform' 'bow tie' 'suit' 'bearskin' 'pickelhaube']
    ```
## SavedModel 文件结构
  - SavedModel 目录包含序列化后的模型 / 参数与单词表，serialized signatures / variable / vocabularies
    ```sh
    ls /tmp/mobilenet/1
    # assets  saved_model.pb  variables
    ```
  - **saved_model.pb** 包含 named signatures，定义了模型的不同功能
    ```sh
    saved_model_cli show --dir /tmp/mobilenet/1 --tag_set serve
    # The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
    # SignatureDef key: "__saved_model_init_op"
    # SignatureDef key: "serving_default"
    ```
  - **variables** 目录包含一个标准的训练过程中的 checkpoint
    ```sh
    ls /tmp/mobilenet/1/variables
    # variables.data-00000-of-00001  variables.index
    ```
  - **assets** 目录包含 TensorFlow graph 会使用的文件，如初始化单词表的文件
  - **assets.extra** 目录包含任何 TensorFlow graph 不用的文件，如接口调用示例等
## tensorflow_model_server 使用模型启动服务
  - **Q / A**
    ```sh
    ''' Q: Tensorflow serving No versions of servable <MODEL> found under base path <path>
    '''
    ''' A: `model_base_path` 指定的路径下需要包含 `version` 文件夹，`version` 下是模型文件
    $ tree /tmp/mobilenet
    /tmp/mobilenet
    └── 1
        ├── assets
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index
    '''
    ```
  - **启动服务**
    ```sh
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
    sudo apt update

    sudo apt-get install tensorflow-model-server

    nohup tensorflow_model_server \
        --rest_api_port=8501 \
        --model_name=mobilenet \
        --model_base_path="/tmp/mobilenet" >server.log 2>&1
    ```
  - **发送请求**
    ```py
    import json
    import numpy
    import requests

    data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/mobilenet:predict', data=data, headers=headers)
    predictions = numpy.array(json.loads(json_response.text)["predictions"])
    ```
## 导出自定义模型
  - **tf.function** 对于自定义的模型，需要指定 signature 中用于提供服务的函数接口，其他没有修饰的函数，加载后将不可见
    ```py
    class CustomModule(tf.Module):

      def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(1.)

      @tf.function
      def __call__(self, x):
        return x * self.v

      @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
      def mutate(self, new_v):
        self.v.assign(new_v)

    module = CustomModule()
    ```
  - **input_signature** 指定函数调用时的输入形式，对于没有指定的函数，需要在保存之前调用过，加载时将使用模型保存之前的输入形式
    ```py
    module(tf.constant(0.))
    tf.saved_model.save(module, "/tmp/module_no_signatures")
    ```
  - **tf.saved_model.load** 加载模型
    ```py
    imported = tf.saved_model.load("/tmp/module_no_signatures")
    print(imported(tf.constant(3.)).numpy())
    # 3.0

    imported.mutate(tf.constant(2.))
    print(imported(tf.constant(3.)).numpy())
    # 6.0

    # ValueError: Could not find matching function to call loaded from the SavedModel.
    imported(tf.constant([3.]))
    ```
  - **get_concrete_function** 可以在不调用模型方法的情况下，指定方法的输入形式，包括类型 / 维度 / 自定义名称，维度信息可以是 None，表示任意维度
    ```py
    module.__call__.get_concrete_function(x=tf.TensorSpec([None], tf.float32))
    tf.saved_model.save(module, "/tmp/module_no_signatures")
    imported = tf.saved_model.load("/tmp/module_no_signatures")
    print(imported(tf.constant([3.])).numpy())
    # [3.]
    ```
  - **saved_model_cli** 查看模型信息
    ```sh
    saved_model_cli show --dir /tmp/module_no_signatures --tag_set serve
    # The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
    # SignatureDef key: "__saved_model_init_op"

    saved_model_cli show --dir /tmp/module_no_signatures --all
    ```
## 指定输出的接口功能 signature
  - 通过在保存时通过指定 `signatures` 参数，指定作为接口功能的函数
    ```py
    call = module.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
    tf.saved_model.save(module, "/tmp/module_with_signature", signatures=call)    

    !saved_model_cli show --dir /tmp/module_with_signature --tag_set serve --signature_def serving_default
    # The given SavedModel SignatureDef contains the following input(s):
    #   inputs['x'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: unknown_rank
    #       name: serving_default_x:0
    # The given SavedModel SignatureDef contains the following output(s):
    #   outputs['output_0'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: unknown_rank
    #       name: StatefulPartitionedCall:0
    # Method name is: tensorflow/serving/predict
    ```
  - **模型导入**
    ```py
    imported = tf.saved_model.load("/tmp/module_with_signature")
    signature = imported.signatures["serving_default"]

    print(signature(x=tf.constant([3.]))["output_0"].numpy())
    # [3.]
    imported.mutate(tf.constant(2.))
    print(signature(x=tf.constant([3.]))["output_0"].numpy())
    # [6.]
    print(imported.v.numpy())
    # 2.
    ```
  - 默认的 signature 是 `serving_default`，可以通过指定一个字典输出多个 signature
    ```py
    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def parse_string(string_input):
      return imported(tf.strings.to_number(string_input))

    signatures = {"serving_default": parse_string, "from_float": imported.signatures["serving_default"]}
    tf.saved_model.save(imported, "/tmp/module_with_multiple_signatures", signatures)

    !saved_model_cli show --dir /tmp/module_with_multiple_signatures --tag_set serve
    # The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
    # SignatureDef key: "__saved_model_init_op"
    # SignatureDef key: "from_float"
    # SignatureDef key: "serving_default"
    ```
  - **saved_model_cli** 也可以在命令行直接运行模型
    ```sh
    saved_model_cli run --dir /tmp/module_with_multiple_signatures --tag_set serve --signature_def serving_default --input_exprs="string_input='3.'"
    saved_model_cli run --dir /tmp/module_with_multiple_signatures --tag_set serve --signature_def from_float --input_exprs="x=3."
    ```
## 模型微调 Fine-tuning imported models
  - 导入模型的变参 Variable 可以通过反向传播调整
    ```py
    optimizer = tf.optimizers.SGD(0.05)

    def train_step():
        with tf.GradientTape() as tape:
            loss = (10. - imported(tf.constant(2.))) ** 2
        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    for _ in range(10):
        # "v" approaches 5, "loss" approaches 0
        print("loss={:.2f} v={:.2f}".format(train_step(), imported.v.numpy()))

    # loss=36.00 v=3.20
    # loss=12.96 v=3.92
    # ...
    # loss=0.01 v=4.97
    # loss=0.00 v=4.98
    ```
## Control flow in SavedModels
  - 任何可以进入 tf.function 的方法都可以保存到 SavedModel 中，如 python 的控制流
    ```py
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def control_flow(x):
      if x < 0:
        tf.print("Invalid!")
      else:
        tf.print(x % 3)

    to_export = tf.Module()
    to_export.control_flow = control_flow
    tf.saved_model.save(to_export, "/tmp/control_flow")

    imported = tf.saved_model.load("/tmp/control_flow")
    imported.control_flow(tf.constant(-1))  # Invalid!
    imported.control_flow(tf.constant(2))   # 2
    imported.control_flow(tf.constant(3))   # 0    
    ```
## Estimators 保存与加载 SavedModels
  - **tf.Estimator.export_saved_model** 导出 SavedModels
    ```py
    input_column = tf.feature_column.numeric_column("x")
    estimator = tf.estimator.LinearClassifier(feature_columns=[input_column])

    def input_fn():
        return tf.data.Dataset.from_tensor_slices(
            ({"x": [1., 2., 3., 4.]}, [1, 1, 0, 0])).repeat(200).shuffle(64).batch(16)
    estimator.train(input_fn)

    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
                              tf.feature_column.make_parse_example_spec([input_column]))
    export_path = estimator.export_saved_model("/tmp/from_estimator/", serving_input_fn)    
    ```
  - **tf.saved_model.load** 加载模型
    ```py
    imported = tf.saved_model.load(export_path)

    def predict(x):
        example = tf.train.Example()
        example.features.feature["x"].float_list.value.extend([x])
        return imported.signatures["predict"](examples=tf.constant([example.SerializeToString()]))

    print(predict(1.5))
    print(predict(3.5))    
    ```
  - **tf.estimator.export.build_raw_serving_input_receiver_fn** 可以用于创建输入功能
## Insightface SavedModel Serving server
  - `saved_model_cli` 显示模型 signature_def 信息
    ```sh
    cd /home/leondgarse/workspace/models/insightface_mxnet_model/model-r100-ii/tf_resnet100
    tree
    # .
    # ├── 1
    # │   ├── saved_model.pb
    # │   └── variables
    # │       ├── variables.data-00000-of-00001
    # │       └── variables.index

    saved_model_cli show --dir ./1
    # The given SavedModel contains the following tag-sets:
    # serve

    saved_model_cli show --dir ./1 --tag_set serve
    # The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
    # SignatureDef key: "serving_default"

    saved_model_cli show --dir ./1 --tag_set serve --signature_def serving_default
    # The given SavedModel SignatureDef contains the following input(s):
    #   inputs['input'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: (-1, 112, 112, 3)
    #       name: data:0
    # The given SavedModel SignatureDef contains the following output(s):
    #   outputs['output'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: (-1, 512)
    #       name: fc1/add_1:0
    # Method name is: tensorflow/serving/predict
    ```
  - `tensorflow_model_server` 启动服务
    ```sh
    # model_base_path 需要绝对路径
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=arcface --model_base_path=/home/leondgarse/workspace/models/insightface_mxnet_model/model-r100-ii/tf_resnet100
    ```
  - `requests` 请求返回特征值结果
    ```py
    import json
    import requests
    from skimage.transform import resize

    rr = requests.get("http://localhost:8501/v1/models/arcface")
    print(rr.json())
    # {'model_version_status': [{'version': '1', 'state': 'AVAILABLE', 'status': {'error_code': 'OK', 'error_message': ''}}]}

    x = plt.imread('grace_hopper.jpg')
    print(x.shape)
    # (600, 512, 3)

    xx = resize(x, [112, 112])
    data = json.dumps({"signature_name": "serving_default", "instances": [xx.tolist()]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/arcface:predict',data=data, headers=headers)
    rr = json_response.json()
    print(rr.keys(), np.shape(rr['predictions']))
    # dict_keys(['predictions']) (1, 512)
    ```
  - `MTCNN` 提取人脸位置后请求结果
    ```py
    from mtcnn.mtcnn import MTCNN

    img = plt.imread('grace_hopper.jpg')
    detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7])
    aa = detector.detect_faces(img)
    bb = aa[0]['box']
    cc = img[bb[1]: bb[1] + bb[3], bb[0]: bb[0] + bb[2]]
    dd = resize(cc, [112, 112])

    data = json.dumps({"signature_name": "serving_default", "instances": [dd.tolist()]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/arcface:predict',data=data, headers=headers)
    rr = json_response.json()
    print(rr.keys(), np.shape(rr['predictions']))
    # dict_keys(['predictions']) (1, 512)
    ```
***

# MMDNN 转化与 TensorFlow MTCNN
## Insightface caffe MTCNN model to TensorFlow
  - [Github microsoft/MMdnn](https://github.com/microsoft/MMdnn)
  ```sh
  pip install mmdnn
  python -m mmdnn.conversion._script.convertToIR -f mxnet -n det1-symbol.json -w det1-0001.params -d det1 --inputShape 3,112,112
  mmconvert -sf mxnet -iw det1-0001.params -in det1-symbol.json -df tensorflow -om det1 --inputShape 3,224,224
  ```
  ```sh
  cd ~/workspace/face_recognition_collection/facenet/src
  cp align align_bak -r

  cd ~/workspace/face_recognition_collection/insightface/deploy/mtcnn-model
  mmtoir -f caffe -n det1.prototxt -w det1.caffemodel -o det1
  mmtoir -f caffe -n det2.prototxt -w det2.caffemodel -o det2
  mmtoir -f caffe -n det3.prototxt -w det3.caffemodel -o det3

  mmtocode -f tensorflow --IRModelPath det1.pb --IRWeightPath det1.npy --dstModelPath det1.py
  mmtocode -f tensorflow --IRModelPath det2.pb --IRWeightPath det2.npy --dstModelPath det2.py
  mmtocode -f tensorflow --IRModelPath det3.pb --IRWeightPath det3.npy --dstModelPath det3.py

  cp *.npy ~/workspace/face_recognition_collection/facenet/src/align/
  ```
## MTCNN with all platforms
  - [Github imistyrain/MTCNN](https://github.com/imistyrain/MTCNN)
  - **使用 RGB 图像替换 BGR 图像用于检测** 生成的模型默认使用 `BGR` 图像用于检测
    ```py
    # tensorflow/caffe2tf.py +357
    images = images[:, :, ::-1]
    images = tf.expand_dims(images, 0)    
    ```
    Tensorflow 1.14 环境下生成 PB 模型
    ```sh
    python caffe2tf.py
    ```
## MTCNN pb 模型转化为 saved model
  ```py
  import tensorflow as tf
  from tensorflow.python.saved_model import signature_constants
  from tensorflow.python.saved_model import tag_constants

  export_dir = './saved'
  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

  model_path = './mtcnn.pb'
  graph = tf.Graph()
  with graph.as_default():
      with open(model_path, 'rb') as f:
          graph_def = tf.compat.v1.GraphDef.FromString(f.read())

  sigs = {}

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      # name="" is important to ensure we don't get spurious prefixing
      tf.import_graph_def(graph_def, name="")
      graph = tf.compat.v1.get_default_graph()
      feeds = {
        'input': graph.get_tensor_by_name('input:0'),
        'min_size': graph.get_tensor_by_name('min_size:0'),
        'thresholds': graph.get_tensor_by_name('thresholds:0'),
        'factor': graph.get_tensor_by_name('factor:0'),
      }
      fetches = {
        'prob': graph.get_tensor_by_name('prob:0'),
        'landmarks': graph.get_tensor_by_name('landmarks:0'),
        'box': graph.get_tensor_by_name('box:0'),
      }

      sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
          tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(feeds, fetches)

      builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map=sigs)

  builder.save()
  ```
## TF_1 加载 frozen MTCNN
  ```py
  tf.__version__
  # '1.14.0'

  graph_def = tf.GraphDef()
  loaded = graph_def.ParseFromString(open('./mtcnn.pb', 'rb').read())
  def _imports_graph_def():
      tf.import_graph_def(graph_def, name="")
  wrapped_import = tf.wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  fe = tf.nest.map_structure(import_graph.as_graph_element, ['input:0', 'min_size:0', 'thresholds:0', 'factor:0'])
  ft = tf.nest.map_structure(import_graph.as_graph_element, ['prob:0', 'landmarks:0', 'box:0'])
  inception_func = wrapped_import.prune(fe, ft)
  input_img = tf.ones([224,224,3], dtype=tf.float32)
  out = inception_func(input_img, tf.constant(20.0), tf.constant([0.6, 0.6, 0.6]), tf.constant(0.7))
  ```
## TF_2 加载 frozen MTCNN
  ```py
  import tensorflow as tf
  tf.__version__
  # '2.0.0'

  class MTCNN:

      def __init__(self, model_path, min_size=40, factor=0.709, thresholds=[0.6, 0.7, 0.7]):
          self.min_size = min_size
          self.factor = factor
          self.thresholds = thresholds

          graph = tf.Graph()
          with graph.as_default():
              with open(model_path, 'rb') as f:
                  graph_def = tf.compat.v1.GraphDef.FromString(f.read())
                  tf.import_graph_def(graph_def, name='')
          self.graph = graph
          config = tf.compat.v1.ConfigProto(
              gpu_options = tf.compat.v1.GPUOptions(allow_growth=True),
              allow_soft_placement=True,
              intra_op_parallelism_threads=4,
              inter_op_parallelism_threads=4
              )
          config.gpu_options.allow_growth = True
          self.sess = tf.compat.v1.Session(graph=graph, config=config)

      def detect_faces(self, img):
          feeds = {
              self.graph.get_operation_by_name('input').outputs[0]: img,
              self.graph.get_operation_by_name('min_size').outputs[0]: self.min_size,
              self.graph.get_operation_by_name('thresholds').outputs[0]: self.thresholds,
              self.graph.get_operation_by_name('factor').outputs[0]: self.factor
          }
          fetches = [self.graph.get_operation_by_name('prob').outputs[0],
                    self.graph.get_operation_by_name('landmarks').outputs[0],
                    self.graph.get_operation_by_name('box').outputs[0]]
          prob, landmarks, box = self.sess.run(fetches, feeds)
          return box, prob, landmarks
  ```
  ```py
  # cd ~/workspace/face_recognition_collection/MTCNN/tensorflow
  img = plt.imread('../test_images/2.jpg')
  det = MTCNN("./mtcnn.pb")
  det.detect_faces(img)
  # (array([[ 65.71266,  74.45414, 187.65063, 172.71921]], dtype=float32),
  #  array([0.99999845], dtype=float32),
  #  array([[113.4738  , 113.50406 , 138.02603 , 159.49994 , 158.71802 ,
  #          102.397964, 147.4054  , 125.014786, 105.924614, 145.5773  ]],
  #        dtype=float32))

  bb, cc, pp = det.detect_faces(img)
  bb = np.array([[ii[1], ii[0], ii[3], ii[2]] for ii in bb])
  # array([[ 74.45414,  65.71266, 172.71921, 187.65063]], dtype=float32)
  pp = np.array([ii.reshape(2, 5)[::-1].T for ii in pp])
  # array([[[102.397964, 113.4738  ],
  #         [147.4054  , 113.50406 ],
  #         [125.014786, 138.02603 ],
  #         [105.924614, 159.49994 ],
  #         [145.5773  , 158.71802 ]]], dtype=float32)
  ```
***

# TF_1 checkpoints
## save and restore checkpoints models
  - [A quick complete tutorial to save and restore Tensorflow models](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)
  ```py
  import tensorflow as tf
  tf.__version__
  # '1.14.0'

  ''' Save checkpoints'''
  #Define a test operation that we will restore
  w1 = tf.placeholder("float", name="w1")
  w2 = tf.placeholder("float", name="w2")
  b1= tf.Variable(2.0, name="bias")
  feed_dict ={w1:4, w2:8}

  w3 = tf.multiply(w1, w2)
  w4 = tf.add(w3, b1, name="op_to_restore")

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.save(sess, 'test/my_test_model', global_step=1000)
  os.listdir('test')
  # ['my_test_model-1000.data-00000-of-00001', 'checkpoint', 'my_test_model-1000.index', 'my_test_model-1000.meta']

  ''' Load checkpoints'''
  sess = tf.Session()
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('./test/my_test_model-1000.meta')
  saver.restore(sess, tf.train.latest_checkpoint('./test'))

  # Now, let's access and create placeholders variables and
  # create feed-dict to feed new data
  graph = tf.get_default_graph()  # Or graph = sess.graph
  w1 = graph.get_tensor_by_name("w1:0")
  w2 = graph.get_tensor_by_name("w2:0")
  feed_dict ={w1:13.0, w2:17.0}

  #Now, access the op that you want to run.
  op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
  print(sess.run(op_to_restore, feed_dict))
  # 223.0

  ''' Add more to the current graph '''
  add_on_op = tf.multiply(op_to_restore,2)

  print(sess.run(add_on_op, feed_dict))
  # 446.0
  ```
  **一个更复杂的示例**
  ```py
  ......
  ......
  saver = tf.train.import_meta_graph('vgg.meta')
  # Access the graph
  graph = tf.get_default_graph()
  ## Prepare the feed_dict for feeding data for fine-tuning

  #Access the appropriate output for fine-tuning
  fc7= graph.get_tensor_by_name('fc7:0')

  #use this if you only want to change gradients of the last layer
  fc7 = tf.stop_gradient(fc7) # It's an identity function
  fc7_shape= fc7.get_shape().as_list()

  new_outputs=2
  weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
  biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
  output = tf.matmul(fc7, weights) + biases
  pred = tf.nn.softmax(output)
  ```
## inspect_checkpoint
  - **tensorflow.python.tools.inspect_checkpoint** 可以快速检测某个 checkpoint 中的变量
  ```py
  # import the inspect_checkpoint library
  from tensorflow.python.tools import inspect_checkpoint as chkp

  chkp.print_tensors_in_checkpoint_file("./test/my_test_model-1000", tensor_name='', all_tensors=True)
  # tensor_name:  bias
  # 2.0
  # Total number of params: 1

  chkp.print_tensors_in_checkpoint_file("./test/my_test_model-1000", tensor_name='bias', all_tensors=True)
  # tensor_name:  bias
  # 2.0
  # Total number of params: 1
  ```
## Insightface Checkpoints to SavedModel
  - [Github luckycallor/InsightFace-tensorflow](https://github.com/luckycallor/InsightFace-tensorflow)
  - **Tensorflow 1.14 加载并保存模型**
    ```py
    tf.__version__  # '1.14.0'
    import yaml
    from model import get_embd

    ''' 加载模型 '''
    config = yaml.load(open('./configs/config_ms1m_100.yaml'))
    images = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_image')
    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, '/home/leondgarse/Downloads/config_ms1m_100_334k/best-m-334000')

    ''' 从 graph 中获取输入 / 输出 tensor '''
    gg = sess.graph
    oo = gg.get_operations()
    print(len(oo))
    # 4046

    ''' 输入 tensor，必须包括所有的 Placeholder '''
    oo[:5]
    # [<tf.Operation 'input_image' type=Placeholder>,
    #  <tf.Operation 'train_phase' type=Placeholder>,
    #  <tf.Operation 'train_phase_last' type=Placeholder>,
    #  <tf.Operation 'embd_extractor/resnet_v2_50/conv1/weights/Initializer/truncated_normal/shape' type=Const>,
    #  <tf.Operation 'embd_extractor/resnet_v2_50/conv1/weights/Initializer/truncated_normal/mean' type=Const>]
    [ii for ii in oo if ii.type == 'Placeholder']
    # [<tf.Operation 'input_image' type=Placeholder>,
    #  <tf.Operation 'train_phase' type=Placeholder>,
    #  <tf.Operation 'train_phase_last' type=Placeholder>]
    feeds = {
      'input_image': gg.get_tensor_by_name('input_image:0'),
      'train_phase': gg.get_tensor_by_name('train_phase:0'),
      'train_phase_last': gg.get_tensor_by_name('train_phase_last:0')
    }
    feeds = {ii.name: ii.outputs[0]  for ii in oo if ii.type == 'Placeholder'}

    ''' 输出 tensor，首先获取到名称 [ ??? ] '''
    oo[-5:] # NOT these
    # [<tf.Operation 'save/Assign_277' type=Assign>,
    #  <tf.Operation 'save/Assign_278' type=Assign>,
    #  <tf.Operation 'save/Assign_279' type=Assign>,
    #  <tf.Operation 'save/Assign_280' type=Assign>,
    #  <tf.Operation 'save/restore_all' type=NoOp>]
    [(id, ii) for id, ii in enumerate(oo) if len(ii.outputs) != 0 and
              ii.outputs[0].shape.dims != None and ii.outputs[0].shape.as_list() == [None, 512]]
    # [(3693, <tf.Operation 'embd_extractor/fully_connected/MatMul' type=MatMul>),
    #  (3694, <tf.Operation 'embd_extractor/fully_connected/BiasAdd' type=BiasAdd>),
    #  (3752, <tf.Operation 'embd_extractor/BatchNorm_1/Reshape_1' type=Reshape>)]
    oo[3750:3756]
    # [<tf.Operation 'embd_extractor/BatchNorm_1/cond_1/Merge_1' type=Merge>,
    #  <tf.Operation 'embd_extractor/BatchNorm_1/Shape' type=Shape>,
    #  <tf.Operation 'embd_extractor/BatchNorm_1/Reshape_1' type=Reshape>,
    #  <tf.Operation 'init' type=NoOp>,
    #  <tf.Operation 'save/filename/input' type=Const>,
    #  <tf.Operation 'save/filename' type=PlaceholderWithDefault>]
    fetches = {
      'output': gg.get_tensor_by_name('embd_extractor/BatchNorm_1/Reshape_1:0')
    }

    tf.saved_model.simple_save(sess, './1', inputs=feeds, outputs=fetches)
    ```
  - **TF 1.14 加载 saved_model**
    ```py
    # Tensorflow 1.14
    ''' 提取特征值 '''
    sess = tf.InteractiveSession()
    meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], "./tf_resnet100")
    x = sess.graph.get_tensor_by_name("data:0")
    y = sess.graph.get_tensor_by_name("fc1/add_1:0")

    emb = sess.run(y, feed_dict={x: np.ones([3, 112, 112, 3], dtype=np.float32)})
    print(emb.shape)
    # (3, 512)
    ```
  - **Tensorflow 2.0 加载 saved_model**
    ```py
    tf.__version__  # '2.0.0'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    loaded = tf.saved_model.load('./1')
    loaded.signatures
    # _SignatureMap({'serving_default': <tensorflow.python.eager.wrap_function.WrappedFunction object at 0x7fee8c051fd0>})
    _interp = loaded.signatures['serving_default']
    interp = lambda xx: _interp(tf.convert_to_tensor(xx, dtype=tf.float32))["output"].numpy()
    emb = interp(np.ones([3, 112, 112, 3]))
    print(emb.shape)
    # (3, 512)
    ```
***

# Keras h5 to pb
  ```py
  # tf.__version__
  # '1.15.0'
  from tensorflow.python.framework import graph_util, graph_io
  def h5_to_pb(h5_model, output_dir, output_model_name, out_prefix="output_", log_tensorboard=True):
      if os.path.exists(output_dir) == False:
          os.mkdir(output_dir)
      out_nodes = []
      for i in range(len(h5_model.outputs)):
          out_nodes.append(out_prefix + str(i + 1))
          tf.identity(h5_model.output[i], out_prefix + str(i + 1))
      sess = tf.keras.backend.get_session()

      # 写入pb模型文件
      init_graph = sess.graph.as_graph_def()
      main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
      graph_io.write_graph(main_graph, output_dir, name=output_model_name, as_text=False)
      # 输出日志文件
      if log_tensorboard:
          from tensorflow.python.tools import import_pb_to_tensorboard
          import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir, output_model_name), output_dir)
  ```
  **Save frozen pb**
  ```py
  # tf.__version__
  # '1.15.0'
  def save_frozen(model, filename):
      # First freeze the graph and remove training nodes.
      sess = tf.keras.backend.get_session()
      frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [model.output.op.name])
      frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
      # Save the model
      with open(filename, "wb") as ofile:
          ofile.write(frozen_graph.SerializeToString())
  ```
  **TF 1 Inference**
  ```py
  import numpy as np
  import tensorflow as tf
  from tensorflow.python.platform import gfile

  name = "tf.pb"

  with tf.Session() as persisted_sess:
      print("load graph")
      with gfile.FastGFile(name, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())

      persisted_sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')

      inp = persisted_sess.graph.get_tensor_by_name('0:0')
      out = persisted_sess.graph.get_tensor_by_name('LogSoftmax:0')
      #test = np.random.rand(1, 1, 28, 28).astype(np.float32)
      #feed_dict = {inp: test}

      img = np.load("./assets/image.npz")
      feed_dict = {inp: img.reshape([1, 1,28,28])}

      classification = persisted_sess.run(out, feed_dict)
      print(out)
      print(classification)
  ```
***

# TF2 to TF1
  - [Error in loading a keras model saved by tf 1.15 from tf 1.14](https://github.com/tensorflow/tensorflow/issues/33479)
  - Load in TF > 1.15 and convert to `weights` + `json model`
    ```py
    tf.__version__
    # '2.1.0'

    from tensorflow import keras
    mm = keras.models.load_model('./model/se_mobile_facenet_256.h5')
    mm.save_weights("model/weights_only.h5")
    json_config = mm.to_json()
    with open('model/model_config.json', 'w') as json_file:
        json_file.write(json_config)
    ```
  - Modify `model/model_config.json`, delete `"ragged": false`
    ```py
    # For tf15 / tf20 saved json file, delete '"ragged": false,'
    !sed -i 's/"ragged": false, //' model/model_config.json
    # For tf-nightly saved json file, also replace '"class_name": "Functional"' by '"class_name": "Model"'
    !sed -i 's/"class_name": "Functional"/"class_name": "Model"/' model/model_config.json
    # For tf23 saved json file, delete '"groups": 1, '
    !sed -i 's/"groups": 1, //g' model/model_config.json
    ```
  - Reload in TF13 and save `h5`
    ```py
    tf.__version__
    # '1.13.1'

    from tensorflow import keras
    from keras.initializers import glorot_normal, glorot_uniform
    from keras.utils import CustomObjectScope

    with open('model/model_config.json') as json_file:
        json_config = json_file.read()
    with CustomObjectScope({'GlorotNormal': glorot_normal(), "GlorotUniform": glorot_uniform()}):
        new_model = keras.models.model_from_json(json_config)
    new_model.load_weights('model/weights_only.h5')
    new_model.save('./model/se_mobile_facenet_256_13.h5')
    ```
  - Reload EfficientNet in TF13
    ```py
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import CustomObjectScope

    def swish(inputs):
        return (K.sigmoid(inputs) * inputs)

    class FixedDropout(keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = K.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    with open('model/model_config.json') as json_file:
        json_config = json_file.read()
    with CustomObjectScope({'swish': swish, 'FixedDropout': FixedDropout}):
        new_model = keras.models.model_from_json(json_config)
    new_model.load_weights('model/weights_only.h5')
    new_model.save('tf13/EB4.h5')
    ```
  - Error in save `mobilenetv3` `h5` model in `TF1.15` [ValueError: Unable to create group (Name already exists)](https://www.gitmemory.com/issue/keras-team/keras/12195/523749332)
    ```py
    ''' Q:
      new_model.save('aa.h5')
      # ValueError: Unable to create group (name already exists)
    '''
    ''' A: It's caused by a layer named `foo` is in a network after a layer named `foo/bar`
      !vi /opt/anaconda3/envs/tf14/lib/python3.7/site-packages/tensorflow_core/python/keras/saving/hdf5_format.py +618
      # 618 for layer in layers:
      # 619   try:
      # 620     g = group.create_group(layer.name)
      # 621   except ValueError:
      # 622     raise ValueError('An error occurred creating weights group for {0}.'.format(layer.name))

      # Re-run to detect where is the error layer.
      new_model.save('aa.h5')
      # ValueError: An error occurred creating weights group for expanded_conv/depthwise

      # Change layer name
      - 524  name=prefix + 'depthwise')
      + 524  name=prefix + 'depthwise/DConv')
    '''
    ```
***

# Pytorch
## Torch model inference
  ```py
  class Torch_model_interf:
      def __init__(self, model_file, image_size=(112, 112)):
          import torch
          self.torch = torch
          cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
          device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
          self.device = self.torch.device(device_name)
          self.model = self.torch.jit.load(model_file, map_location=device_name)

      def __call__(self, imgs):
          # print(imgs.shape, imgs[0])
          imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
          imgs = (imgs - 127.5) * 0.0078125
          output = self.model(self.torch.from_numpy(imgs).to(self.device).float())
          return output.cpu().detach().numpy()
  ```
## Save and load entire model
  - [SAVING AND LOADING MODELS](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
  ```py
  import data_distiller
  mm = data_distiller.Torch_model_interf('../epoch_16_7.pth')

  import torch
  from backbones import iresnet50, iresnet100
  resnet = iresnet50().cuda()
  weight = torch.load('/media/SD/tdtest/partial_fc/pytorch/partial_fc_glint360k_r50/16backbone.pth')
  resnet.load_state_dict(weight)
  resnet.eval()

  xx = torch.randn(1, 3, 112, 112).cuda()
  traced_cell = torch.jit.trace(resnet, (xx))
  torch.jit.save(traced_cell, 'aa.pth')
  aa = torch.jit.load('aa.pth')

  mm = Torch_model_interf('aa.pth')
  mm(np.ones([1, 112, 112, 3])).shape
  ```
***

# Replace UpSampling2D with Conv2DTranspose
## Conv2DTranspose output shape
  ```py
  for strides in range(1, 4):
      for kernel_size in range(1, 4):
          aa = keras.layers.Conv2DTranspose(3, kernel_size, padding='same', strides=strides)
          aa.build([1, 3, 3, 3])
          print("[SAME] kernel_size: {}, strides: {}, shape: {}".format(kernel_size, strides, aa(tf.ones([1, 3, 3, 3], dtype='float32')).shape.as_list()))
  # [SAME] kernel_size: 1, strides: 1, shape: [1, 3, 3, 3]
  # [SAME] kernel_size: 2, strides: 1, shape: [1, 3, 3, 3]
  # [SAME] kernel_size: 3, strides: 1, shape: [1, 3, 3, 3]
  # [SAME] kernel_size: 1, strides: 2, shape: [1, 6, 6, 3]
  # [SAME] kernel_size: 2, strides: 2, shape: [1, 6, 6, 3]
  # [SAME] kernel_size: 3, strides: 2, shape: [1, 6, 6, 3]
  # [SAME] kernel_size: 1, strides: 3, shape: [1, 9, 9, 3]
  # [SAME] kernel_size: 2, strides: 3, shape: [1, 9, 9, 3]
  # [SAME] kernel_size: 3, strides: 3, shape: [1, 9, 9, 3]

  for strides in range(1, 4):
      for kernel_size in range(1, 5):
          aa = keras.layers.Conv2DTranspose(3, kernel_size, padding='valid', strides=strides)
          aa.build([1, 3, 3, 3])
          print("[VALID] kernel_size: {}, strides: {}, shape: {}".format(kernel_size, strides, aa(tf.ones([1, 3, 3, 3], dtype='float32')).shape.as_list()))
  # [VALID] kernel_size: 1, strides: 1, shape: [1, 3, 3, 3]
  # [VALID] kernel_size: 2, strides: 1, shape: [1, 4, 4, 3]
  # [VALID] kernel_size: 3, strides: 1, shape: [1, 5, 5, 3]
  # [VALID] kernel_size: 4, strides: 1, shape: [1, 6, 6, 3]
  # [VALID] kernel_size: 1, strides: 2, shape: [1, 6, 6, 3]
  # [VALID] kernel_size: 2, strides: 2, shape: [1, 6, 6, 3]
  # [VALID] kernel_size: 3, strides: 2, shape: [1, 7, 7, 3]
  # [VALID] kernel_size: 4, strides: 2, shape: [1, 8, 8, 3]
  # [VALID] kernel_size: 1, strides: 3, shape: [1, 9, 9, 3]
  # [VALID] kernel_size: 2, strides: 3, shape: [1, 9, 9, 3]
  # [VALID] kernel_size: 3, strides: 3, shape: [1, 9, 9, 3]
  # [VALID] kernel_size: 4, strides: 3, shape: [1, 10, 10, 3]
  ```
## Nearest interpolation
  - **Image methods**
    ```py
    imsize = 3
    x, y = np.ogrid[:imsize, :imsize]
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(imsize + imsize)
    plt.imshow(img, interpolation='none')

    import tensorflow.keras.backend as K
    iaa = tf.image.resize(img, (6, 6), method='nearest')
    ibb = K.resize_images(tf.expand_dims(tf.cast(img, 'float32'), 0), 2, 2, K.image_data_format(), interpolation='nearest')
    ```
  - **UpSampling2D**
    ```py
    aa = keras.layers.UpSampling2D((2, 2), interpolation='nearest')
    icc = aa(tf.expand_dims(tf.cast(img, 'float32'), 0)).numpy()[0]

    print(np.allclose(iaa, icc))
    # True
    ```
  - **tf.nn.conv2d_transpose**
    ```py
    def nearest_upsample_weights(factor, number_of_classes=3):
        filter_size = 2 * factor - factor % 2
        weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes), dtype=np.float32)
        upsample_kernel = np.zeros([filter_size, filter_size])
        upsample_kernel[1:factor + 1, 1:factor + 1] = 1

        for i in range(number_of_classes):
            weights[:, :, i, i] = upsample_kernel
        return weights

    channel, factor = 3, 2
    idd = tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor, img.shape[1] * factor, channel], strides=factor, padding='SAME')
    print(np.allclose(iaa, idd))
    # True

    # Output shape can be different values
    channel, factor = 3, 3
    print(tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor, img.shape[1] * factor, channel], strides=factor, padding='SAME').shape)
    # (1, 9, 9, 3)
    print(tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor - 1, img.shape[1] * factor - 1, channel], strides=factor, padding='SAME').shape)
    # (1, 8, 8, 3)
    print(tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), nearest_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor - 2, img.shape[1] * factor - 2, channel], strides=factor, padding='SAME').shape)
    # (1, 7, 7, 3)
    ```
  - **Conv2DTranspose**
    ```py
    bb = keras.layers.Conv2DTranspose(channel, 2 * factor - factor % 2, padding='same', strides=factor, use_bias=False)
    bb.build([None, None, None, channel])
    bb.set_weights([nearest_upsample_weights(factor, channel)])
    iee = bb(tf.expand_dims(img.astype('float32'), 0)).numpy()[0]
    print(np.allclose(iaa, iee))
    # True
    ```
## Bilinear
  - [pytorch_bilinear_conv_transpose.py](https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183)
  - [Upsampling and Image Segmentation with Tensorflow and TF-Slim](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)
  - **UpSampling2D**
    ```py
    imsize = 3
    x, y = np.ogrid[:imsize, :imsize]
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(imsize + imsize)
    plt.imshow(img, interpolation='none')

    channel, factor = 3, 3
    iaa = tf.image.resize(img, (img.shape[0] * factor, img.shape[1] * factor), method='bilinear')

    aa = keras.layers.UpSampling2D((factor, factor), interpolation='bilinear')
    ibb = aa(tf.expand_dims(tf.cast(img, 'float32'), 0)).numpy()[0]
    print(np.allclose(iaa, ibb))
    # True
    ```
  - **Pytorch BilinearConvTranspose2d**
    ```py
    import torch
    import torch.nn as nn

    class BilinearConvTranspose2d(nn.ConvTranspose2d):
        def __init__(self, channels, stride, groups=1):
            if isinstance(stride, int):
                stride = (stride, stride)

            kernel_size = (2 * stride[0] - stride[0] % 2, 2 * stride[1] - stride[1] % 2)
            # padding = (stride[0] - 1, stride[1] - 1)
            padding = 1
            super().__init__(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

        def reset_parameters(self):
            nn.init.constant(self.bias, 0)
            nn.init.constant(self.weight, 0)
            bilinear_kernel = self.bilinear_kernel(self.stride)
            for i in range(self.in_channels):
                j = i if self.groups == 1 else 0
                self.weight.data[i, j] = bilinear_kernel

        @staticmethod
        def bilinear_kernel(stride):
            num_dims = len(stride)

            shape = (1,) * num_dims
            bilinear_kernel = torch.ones(*shape)

            # The bilinear kernel is separable in its spatial dimensions
            # Build up the kernel channel by channel
            for channel in range(num_dims):
                channel_stride = stride[channel]
                kernel_size = 2 * channel_stride - channel_stride % 2
                # e.g. with stride = 4
                # delta = [-3, -2, -1, 0, 1, 2, 3]
                # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
                # delta = torch.arange(1 - channel_stride, channel_stride)
                delta = torch.arange(0, kernel_size)
                delta = delta - (channel_stride - 0.5) if channel_stride % 2 == 0 else delta - (channel_stride - 1)
                channel_filter = (1 - torch.abs(delta / float(channel_stride)))
                # Apply the channel filter to the current channel
                shape = [1] * num_dims
                shape[channel] = kernel_size
                bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
            return bilinear_kernel

    aa = BilinearConvTranspose2d(channel, factor)
    cc = aa(torch.from_numpy(np.expand_dims(img.transpose(2, 0, 1), 0).astype('float32')))
    icc = cc.detach().numpy()[0].transpose(1, 2, 0)
    print(np.allclose(iaa, icc))
    # False
    ```
  - **tf.nn.conv2d_transpose**
    ```py
    # This is same with pytorch bilinear kernel
    def upsample_filt(size):
        factor = (size + 1) // 2
        center = factor - 1 if size % 2 == 1 else factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    def bilinear_upsample_weights(factor, number_of_classes=3):
        filter_size = 2 * factor - factor % 2
        weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes), dtype=np.float32)
        upsample_kernel = upsample_filt(filter_size)

        for i in range(number_of_classes):
            weights[:, :, i, i] = upsample_kernel
        return weights

    idd = tf.nn.conv2d_transpose(tf.expand_dims(tf.cast(img, 'float32'), 0), bilinear_upsample_weights(factor, channel), output_shape=[1, img.shape[0] * factor, img.shape[1] * factor, channel], strides=factor, padding='SAME')[0]
    print(np.allclose(icc, idd))
    # True
    ```
  - **Conv2DTranspose**
    ```py
    aa = keras.layers.Conv2DTranspose(channel, 2 * factor - factor % 2, padding='same', strides=factor, use_bias=False)
    aa.build([None, None, None, channel])
    aa.set_weights([bilinear_upsample_weights(factor, channel)])
    iee = aa(tf.expand_dims(tf.cast(img, 'float32'), 0)).numpy()[0]
    ```
  - **Plot**
    ```py
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    imgs = [img, iaa, ibb, icc, idd, iee]
    names = ["Orignal", "tf.image.resize", "UpSampling2D", "Pytorch ConvTranspose2d", "tf.nn.conv2d_transpose", "TF Conv2DTranspose"]
    for ax, imm, nn in zip(axes, imgs, names):
        ax.imshow(imm)
        ax.axis('off')
        ax.set_title(nn)
    plt.tight_layout()
    ```
    ```py
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1])
    ```
## Clone model
  ```py
  def convert_UpSampling2D_layer(layer):
      print(layer.name)
      if isinstance(layer, keras.layers.UpSampling2D):
          print(">>>> Convert UpSampling2D <<<<")
          channel = layer.input.shape[-1]
          factor = 2
          aa = keras.layers.Conv2DTranspose(channel, 2 * factor - factor % 2, padding='same', strides=factor, use_bias=False)
          aa.build(layer.input.shape)
          aa.set_weights([bilinear_upsample_weights(factor, number_of_classes=channel)])
          return aa
      return layer

  mm = keras.models.load_model('aa.h5', compile=False)
  mmn = keras.models.clone_model(mm, clone_function=convert_UpSampling2D_layer)
  ```
***
