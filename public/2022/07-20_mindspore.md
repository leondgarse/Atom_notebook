# ___2022 - 07 - 20 MindSpore___
***

# MindSpore 初学教程
## 安装
  [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.7/index.html)
  ```sh
  pip install mindspore mindvision
  ```
## 通用函数
  ```py
  def show_image_with_labels(images, labels, preds=None, rows=1):
      cols = int(np.ceil(len(images) / rows))
      fig, axes = plt.subplots(rows, cols)
      axes = axes.flatten()
      for id, (image, label, ax) in enumerate(zip(images, labels, axes)):
          title = "label: {}".format(label)
          title += ", pred: {}".format(preds[id]) if preds is not None else ""
          ax.imshow(image, interpolation="None", cmap="gray")  # [0] is the channel axis
          ax.set_title(title)
      plt.tight_layout()
  ```
## MNIST
  ```py
  import os
  from mindvision.dataset import Mnist
  from mindvision.classification.models import lenet
  from mindvision.engine.callback import LossMonitor, ValAccMonitor
  from mindspore import nn
  import mindspore as ms

  dataset_path = os.path.expanduser('~/mindvision/dataset/mnist/')
  dataset_train = Mnist(path=dataset_path, split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True).run()
  dataset_eval = Mnist(path=dataset_path, split="test", batch_size=32, resize=32, download=True).run()
  steps = dataset_train.get_dataset_size()
  network = lenet(num_classes=10, pretrained=False)
  net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
  model = ms.train.Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})

  num_epochs = 10
  config_ck = ms.train.callback.CheckpointConfig(save_checkpoint_steps=steps, keep_checkpoint_max=10)
  ckpoint = ms.train.callback.ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)
  val_acc = ValAccMonitor(model, dataset_eval, num_epochs=num_epochs, ckpt_directory="./lenet", metric_name="accuracy")
  callbacks=[ckpoint, val_acc, LossMonitor(0.01, steps)]

  model.train(num_epochs, dataset_train, callbacks)
  # Epoch:[  0/ 10], step:[ 1875/ 1875], loss:[0.320/1.257], time:9.973 ms, lr:0.01000
  # --------------------
  # Epoch: [  1 /  10], Train Loss: [0.320], accuracy:  0.960
  # Epoch time: 10528.515 ms, per step time: 5.615 ms, avg loss: 1.257
  # Epoch:[  1/ 10], step:[ 1875/ 1875], loss:[0.012/0.083], time:8.976 ms, lr:0.01000
  # --------------------
  # Epoch: [  2 /  10], Train Loss: [0.012], accuracy:  0.979
  # Epoch time: 10802.161 ms, per step time: 5.761 ms, avg loss: 0.083
  # Epoch:[  2/ 10], step:[ 1875/ 1875], loss:[0.009/0.052], time:9.973 ms, lr:0.01000
  # --------------------
  # Epoch: [  3 /  10], Train Loss: [0.009], accuracy:  0.988
  # Epoch time: 11147.733 ms, per step time: 5.945 ms, avg loss: 0.052
  # Epoch:[  3/ 10], step:[ 1875/ 1875], loss:[0.005/0.039], time:0.000 ms, lr:0.01000
  # --------------------
  # Epoch: [  4 /  10], Train Loss: [0.005], accuracy:  0.985
  # Epoch time: 10869.443 ms, per step time: 5.797 ms, avg loss: 0.039
  # Epoch:[  4/ 10], step:[ 1875/ 1875], loss:[0.071/0.033], time:18.494 ms, lr:0.01000
  # --------------------
  # Epoch: [  5 /  10], Train Loss: [0.071], accuracy:  0.988
  # Epoch time: 11175.194 ms, per step time: 5.960 ms, avg loss: 0.033
  # Epoch:[  5/ 10], step:[ 1875/ 1875], loss:[0.006/0.027], time:36.828 ms, lr:0.01000
  # --------------------
  # Epoch: [  6 /  10], Train Loss: [0.006], accuracy:  0.988
  # Epoch time: 12265.399 ms, per step time: 6.542 ms, avg loss: 0.027
  # Epoch:[  6/ 10], step:[ 1875/ 1875], loss:[0.017/0.022], time:0.000 ms, lr:0.01000
  # --------------------
  # Epoch: [  7 /  10], Train Loss: [0.017], accuracy:  0.991
  # Epoch time: 11569.820 ms, per step time: 6.171 ms, avg loss: 0.022
  # Epoch:[  7/ 10], step:[ 1875/ 1875], loss:[0.002/0.020], time:0.000 ms, lr:0.01000
  # --------------------
  # Epoch: [  8 /  10], Train Loss: [0.002], accuracy:  0.989
  # Epoch time: 11493.535 ms, per step time: 6.130 ms, avg loss: 0.020
  # Epoch:[  8/ 10], step:[ 1875/ 1875], loss:[0.000/0.017], time:15.622 ms, lr:0.01000
  # --------------------
  # Epoch: [  9 /  10], Train Loss: [0.000], accuracy:  0.989
  # Epoch time: 11047.880 ms, per step time: 5.892 ms, avg loss: 0.017
  # Epoch:[  9/ 10], step:[ 1875/ 1875], loss:[0.001/0.015], time:12.965 ms, lr:0.01000
  # --------------------
  # Epoch: [ 10 /  10], Train Loss: [0.001], accuracy:  0.992
  # Epoch time: 11415.099 ms, per step time: 6.088 ms, avg loss: 0.015
  # ================================================================================
  # End of validation the best accuracy is:  0.992, save the best ckpt file in ./lenet\best.ckpt
  ```
  ```py
  import mindspore as ms
  from mindspore import nn
  from mindvision.classification.models import lenet
  from mindvision.dataset import Mnist

  network = lenet(num_classes=10, pretrained=False)
  param_dict = ms.load_checkpoint('lenet/lenet-10_1875.ckpt')
  ms.load_param_into_net(network, param_dict)

  model = ms.train.Model(network, loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean'), metrics={'accuracy'})
  dataset_path = os.path.expanduser('~/mindvision/dataset/mnist/')
  dataset_eval = Mnist(path=dataset_path, split="test", batch_size=32, resize=32, download=True).run()
  print(model.eval(dataset_eval))
  # {'accuracy': 0.9885817307692307}

  preds, labels, images = [], [], []
  for image, label in dataset_eval.create_tuple_iterator():
      pred = model.predict(image).asnumpy().argmax(-1)
      preds.extend(pred)
      labels.extend(label.asnumpy())
      images.extend(image.asnumpy())
  images, preds, labels = np.array(images), np.array(preds), np.array(labels)
  print(f"accuracy: {(preds == labels).sum() / labels.shape[0] = }")
  # accuracy: (preds == labels).sum() / labels.shape[0] = 0.9885817307692307

  error_ids = np.where(preds != labels)[0]
  pick_ids = error_ids[:6]
  show_image_with_labels(images[pick_ids][:, 0], labels[pick_ids], preds[pick_ids], rows=2)
  ```
## Tensor
  ```py
  import mindspore as ms

  # `init` 主要用于并行模式下的延后初始化，在正常情况下不建议使用init对参数进行初始化。
  xx = ms.Tensor(shape=[3, 4], dtype=ms.float32, init=ms.common.initializer.Normal())
  xx
  # Tensor(shape=[3, 4], dtype=Float32, value= <uninitialized>)
  print(xx)
  # [[-0.00681436 -0.01074277 -0.00554867 -0.00841471]
  #  [-0.01261512  0.01515119  0.00152964  0.00128001]
  #  [ 0.01272571  0.00421324  0.00698019 -0.0029147 ]]

  aa = ms.ops.ones_like(xx)
  print(f"{aa.shape = }, {aa.dtype = }")
  # aa.shape = (3, 4), aa.dtype = mindspore.float32

  # shape 必须使用 tuple，dtype 必须指定
  aa = ms.ops.zeros((2, 3), ms.float32)
  ```
  **Concat / Stack**
  ```py
  import mindspore as ms
  aa = ms.ops.zeros((2, 3), ms.float32)
  ms.ops.Concat(axis=-1)([aa, aa]).shape
  # (2, 6)
  ```

  - **稀疏张量 CSRTensor / COOTensor / RowTensor** COO -> Coordinate, CSR -> Compressed Sparse Row 行压缩格式, CSR -> Compressed Sparse Column 列压缩格式
## mindvision 图像数据处理
  - **加载数据集**
    ```py
    from mindvision.dataset import Cifar10

    dataset_path = os.path.expanduser('~/mindvision/dataset/cifar10/')
    dataset = Cifar10(path=dataset_path, split='train', batch_size=6, resize=32, download=True).run()
    data = next(dataset.create_dict_iterator())
    print({kk: (vv.dtype, vv.shape) for kk, vv in data.items()})
    # {'image': (mindspore.float32, (6, 3, 32, 32)), 'label': (mindspore.int32, (6,))}

    # `output_numpy=True` 输出 numpy 格式
    data = next(dataset.create_dict_iterator(output_numpy=True))
    print({kk: (vv.dtype, vv.shape) for kk, vv in data.items()})
    {'image': (dtype('float32'), (6, 3, 32, 32)), 'label': (dtype('int32'), (6,))}
    ```
  - **数据增强**
    ```py
    # Cifar10 默认 transform
    def default_transform(self):
        """Set the default transform for Cifar10 dataset."""
        trans = []
        if self.split == "train":
            trans += [
                transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
                transforms.RandomHorizontalFlip(prob=0.5)
            ]

        trans += [
            transforms.Resize(self.resize),
            transforms.Rescale(1.0 / 255.0, 0.0),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            transforms.HWC2CHW()
        ]

        return trans
    ```
    ```py
    from mindspore import nn
    from mindvision.dataset import Cifar10
    import mindspore.dataset.vision.c_transforms as transforms

    trans = [
        transforms.RandomCrop(size=32, padding=[4, 4, 4, 4]),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.HWC2CHW(),
    ]

    dataset_path = os.path.expanduser('~/mindvision/dataset/cifar10/')
    # dataset = Cifar10(path=dataset_path, split='train', batch_size=6, resize=32, download=True)
    dataset = Cifar10(path=dataset_path, batch_size=6, resize=32, download=True, transform=trans)
    data = next(dataset.run().create_dict_iterator())
    images, labels = data['image'].transpose(0, 2, 3, 1).asnumpy(), data["label"].asnumpy()
    show_image_with_labels(images, [dataset.index2label[ii] for ii in labels], rows=2)
    ```
## LENet
  ```py
  from mindspore import nn

  class LENet5(nn.Cell):
      def __init__(self, num_classes=10, input_shape=(3, 32, 32)):
          super().__init__()
          height, width, input_channels = input_shape
          self.conv1 = nn.Conv2d(input_channels, 6, 5, pad_mode="valid")
          self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
          self.relu = nn.ReLU()
          self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
          self.flatten = nn.Flatten()
          flatten_height = ((height - 5 + 1) // 2 - 5 + 1) // 2  # conv 5x5 -> maxpool 2 -> conv 5x5 -> max_pool 2
          flatten_width = ((width - 5 + 1) // 2 - 5 + 1) // 2  # conv 5x5 -> maxpool 2 -> conv 5x5 -> max_pool 2
          self.fc1 = nn.Dense(16 * flatten_height * flatten_width, 120)
          self.fc2 = nn.Dense(120, 84)
          self.fc3 = nn.Dense(84, num_classes)

      def construct(self, inputs):
          xx = self.conv1(inputs)
          xx = self.relu(xx)
          xx = self.max_pool2d(xx)
          xx = self.conv2(xx)
          xx = self.relu(xx)
          xx = self.max_pool2d(xx)
          xx = self.flatten(xx)
          xx = self.fc1(xx)
          xx = self.relu(xx)
          xx = self.fc2(xx)
          xx = self.relu(xx)
          xx = self.fc3(xx)
          return xx

  mm = LENet5()
  print([(ii.name, ii.shape, ii.dtype) for ii in mm.get_parameters()])
  # [('conv1.weight', (6, 3, 5, 5), mindspore.float32),
  #  ('conv2.weight', (16, 6, 5, 5), mindspore.float32),
  #  ('fc1.weight', (120, 400), mindspore.float32),
  #  ('fc1.bias', (120,), mindspore.float32),
  #  ('fc2.weight', (84, 120), mindspore.float32),
  #  ('fc2.bias', (84,), mindspore.float32),
  #  ('fc3.weight', (10, 84), mindspore.float32),
  #  ('fc3.bias', (10,), mindspore.float32)]
  ```
## 自动微分
  - `MindSpore` 使用 `ops.GradOperation` 计算一阶导数
    - `get_all` 计算梯度，如果等于False，获得第一个输入的梯度，如果等于True，获得所有输入的梯度。默认值为False
    - `get_by_list` 是否对权重参数进行求导，默认值为False
    - `sens_param` 是否对网络的输出值做缩放以改变最终梯度，默认值为False
  - **对输入求一阶导**
    ```py
    from mindspore import nn, ops, Parameter, Tensor
    import mindspore as ms

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(np.array([6.0]), name='w')
            self.b = Parameter(np.array([1.0]), name='b')

        def construct(self, x):
            return self.w * x + self.b

    grad_op = ops.GradOperation()
    xx = Tensor([100], dtype=ms.float32)
    grad_op(Net())(xx)
    # Tensor(shape=[1], dtype=Float32, value= [6.00000000e+000])
    ```
  - **`get_by_list=True` 对权重求一阶导**
    ```py
    from mindspore import nn, ops, Parameter, Tensor, ParameterTuple
    import mindspore as ms

    net = Net()
    grad_op = ops.GradOperation(get_by_list=True)
    params = ParameterTuple(net.trainable_params())
    xx = Tensor([100], dtype=ms.float32)
    grad_op(net, params)(xx)
    # (Tensor(shape=[1], dtype=Float64, value= [1.00000000e+002]),
    #  Tensor(shape=[1], dtype=Float64, value= [1.00000000e+000]))
    ```
  - **`requires_grad=False` 指定相应的权重参数不需要求导**
    ```py
    from mindspore import nn, Parameter, Tensor

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
            self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b', requires_grad=False)

        def construct(self, x):
            return x * self.w + self.b
    ```
  - **`sens_param=True` 梯度值缩放**
    ```py
    from mindspore import nn, ops, Parameter, Tensor, ParameterTuple
    import mindspore as ms

    net = Net()
    grad_op = ops.GradOperation(sens_param=True)
    grad_wrt_output = Tensor([0.1], dtype=ms.float32)  # 缩放指数
    xx = Tensor([100], dtype=ms.float32)
    grad_op(net)(xx, grad_wrt_output)
    # Tensor(shape=[1], dtype=Float32, value= [6.00000024e-001])
    ```
  - **`ops.stop_gradient` 梯度截断**
    ```py
    from mindspore import nn, ops, Parameter, Tensor, ParameterTuple
    import mindspore as ms


    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
            self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b')

        def construct(self, x):
            out = x * self.w + self.b
            out = ops.stop_gradient(out)  # 梯度截断
            return out

    net = Net()
    grad_op = ops.GradOperation(get_by_list=True)
    params = ParameterTuple(net.trainable_params())
    xx = Tensor([100], dtype=ms.float32)
    grad_op(net, params)(xx)
    # (Tensor(shape=[1], dtype=Float32, value= [0.00000000e+000]),
    #  Tensor(shape=[1], dtype=Float32, value= [0.00000000e+000]))
    ```
## 模型训练
  - **`ms.nn.loss` 损失函数**
    ```py
    import mindspore as ms
    from mindspore import nn, Tensor
    loss = nn.L1Loss()
    y_pred = Tensor([[1, 2, 3], [2, 3, 4]]).astype(ms.float32)
    y_true = Tensor([[0, 2, 5], [3, 1, 1]]).astype(ms.float32)
    loss(y_pred, y_true)
    # Tensor(shape=[], dtype=Float32, value= 1.5)
    ```
  - **`ms.nn.optim` 优化器**
    ```py
    from mindvision.classification.models import lenet
    from mindspore.nn import optim

    net = lenet(num_classes=10, pretrained=False)
    opt = optim.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
    ```
  - **模型训练**
    ```py
    import mindspore.nn as nn
    from mindspore.train import Model

    from mindvision.classification.dataset import Mnist
    from mindvision.classification.models import lenet
    from mindvision.engine.callback import LossMonitor

    dataset_path = os.path.expanduser('~/mindvision/dataset/mnist/')
    dataset_train = Mnist(path=dataset_path, split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32, download=True).run()
    steps = dataset_train.get_dataset_size()

    learning_rate = 0.01
    network = lenet(num_classes=10, pretrained=False)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=0.9)

    epochs = 5
    model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})
    model.train(epochs, dataset_train, callbacks=[LossMonitor(learning_rate, steps)])
    ```
    **保存/加载/验证**
    ```py
    import mindspore as ms
    from mindvision.classification.dataset import Mnist
    from mindvision.classification.models import lenet

    ms.save_checkpoint(network, 'aa.ckpt')

    new_net = lenet(num_classes=10, pretrained=False)
    param_dict = ms.load_checkpoint('aa.ckpt')
    ms.load_param_into_net(new_net, param_dict)

    new_model = ms.train.Model(new_net, loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean'), metrics={'accuracy'})
    dataset_path = os.path.expanduser('~/mindvision/dataset/mnist/')
    dataset_eval = Mnist(path=dataset_path, split="test", batch_size=32, resize=32, download=True).run()
    acc = new_model.eval(dataset_eval)
    print(acc)
    # {'acc': 0.9869791666666666}
    ```
## Callbacks
  - **CheckpointConfig + ModelCheckpoint 训练过程中保存模型**
    ```py
    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="lenet", directory="./lenet", config=config_ck)

    ...
    model.train(..., callbacks=[ckpoint_cb])

    !tree lenet
    # lenet
    # ├── lenet-1_1875.ckpt
    # ├── lenet-2_1875.ckpt
    # ├── lenet-3_1875.ckpt
    # ├── lenet-4_1875.ckpt
    # ├── lenet-5_1875.ckpt
    # └── lenet-graph.meta
    ```
    使用相同的前缀名在同一个文件夹下多次调用，会保存成 `{prefix}_{num}-{epoch}_{step}.ckpt` 文件
  - **LossMonitor 统计打印 per step time / avg loss / lr 信息**
    ```py
    from mindvision.engine.callback import LossMonitor
    LossMonitor(0.01, steps)
    ```
  - **ValAccMonitor 统计验证集准确度**
    ```py
    import mindspore as ms
    from mindvision.engine.callback import ValAccMonitor

    dataset_eval = ...
    model = ms.train.Model(...)
    num_epochs = 10
    val_acc = ValAccMonitor(model, dataset_eval, num_epochs=num_epochs, ckpt_directory="./lenet", metric_name="accuracy")
    ```
## 狗和牛角包分类训练推理与部署
  - **下载并解压数据集**
    ```py
    from mindvision.dataset import DownLoad
    dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip"
    DATASET_PATH = os.path.expanduser('~/mindvision/dataset/')
    DownLoad().download_and_extract_archive(dataset_url, DATASET_PATH)

    # !tree mindvision/dataset/DogCroissants/
    # mindvision/dataset/DogCroissants/
    # ├── infer
    # │   ├── croissants.jpg
    # │   └── dog.jpg
    # ├── train
    # │   ├── croissants
    # │   │   ├── 1.PNG
    # │   │   ├── 10.jpg
    # │   │   └── 99.PNG
    # │   └── dog
    # │       ├── 1.jpg
    # │       ├── 98.PNG
    # │       └── 99.jpg
    # └── val
    #     ├── croissants
    #     │   ├── 1.PNG
    #     │   └── 9.jpg
    #     └── dog
    #         ├── 1.jpg
    #         └── 9.jpg
    ```
  - **加载数据集与训练**
    ```py
    import mindspore as ms
    from mindspore import nn
    import mindspore.dataset.vision.c_transforms as transforms
    from mindvision.classification.models import mobilenet_v2
    from mindvision.engine.loss import CrossEntropySmooth
    from mindvision.engine.callback import LossMonitor, ValAccMonitor

    """ 加载数据集 """
    batch_size = 32
    image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    DATASET_PATH = os.path.expanduser('~/mindvision/dataset/')
    train_ds = ms.dataset.ImageFolderDataset(os.path.join(DATASET_PATH, 'DogCroissants/train'), num_parallel_workers=8)
    train_trans = [
        transforms.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(prob=0.5),
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW(),
    ]
    train_ds = train_ds.map(train_trans, input_columns="image", num_parallel_workers=8).batch(batch_size, drop_remainder=True)

    val_ds = ms.dataset.ImageFolderDataset(os.path.join(DATASET_PATH, 'DogCroissants/val'), num_parallel_workers=8)
    val_trans = [
        transforms.Decode(),
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=mean, std=std),
        transforms.HWC2CHW(),
    ]
    val_ds = val_ds.map(val_trans, input_columns="image", num_parallel_workers=8).batch(batch_size, drop_remainder=True)

    """ 下载预训练模型 """
    models_url = "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt"
    dl = DownLoad().download_url(models_url)  # 默认路径 ./

    network = mobilenet_v2(num_classes=2, resize=224)
    param_dict = ms.load_checkpoint("./mobilenet_v2_1.0_224.ckpt")
    _ = ms.load_checkpoint("./mobilenet_v2_1.0_224.ckpt", net=network, strict_load=False, filter_prefix=['head', 'moments'])

    """ 模型训练 """
    network_opt = nn.Momentum(params=network.trainable_params(), learning_rate=0.01, momentum=0.9)
    network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=2)
    metrics = {"Accuracy": nn.Accuracy()}
    model = ms.train.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    num_epochs = 10
    callbacks = [ValAccMonitor(model, val_ds, num_epochs), ms.train.callback.TimeMonitor()]
    model.train(num_epochs, train_ds, callbacks=callbacks)
    # --------------------
    # Epoch: [  1 /  10], Train Loss: [0.382], Accuracy:  0.969
    # epoch time: 16450.251 ms, per step time: 1827.806 ms
    # --------------------
    # Epoch: [  2 /  10], Train Loss: [0.369], Accuracy:  1.000
    # epoch time: 14526.028 ms, per step time: 1614.003 ms
    # --------------------
    # Epoch: [  3 /  10], Train Loss: [0.372], Accuracy:  1.000
    # epoch time: 16713.495 ms, per step time: 1857.055 ms
    # --------------------
    # Epoch: [  4 /  10], Train Loss: [0.342], Accuracy:  1.000
    # epoch time: 15472.170 ms, per step time: 1719.130 ms
    # --------------------
    # Epoch: [  5 /  10], Train Loss: [0.366], Accuracy:  1.000
    # epoch time: 14454.406 ms, per step time: 1606.045 ms
    # --------------------
    # Epoch: [  6 /  10], Train Loss: [0.349], Accuracy:  1.000
    # epoch time: 14473.458 ms, per step time: 1608.162 ms
    # --------------------
    # Epoch: [  7 /  10], Train Loss: [0.334], Accuracy:  1.000
    # epoch time: 15732.421 ms, per step time: 1748.047 ms
    # --------------------
    # Epoch: [  8 /  10], Train Loss: [0.348], Accuracy:  1.000
    # epoch time: 15154.196 ms, per step time: 1683.800 ms
    # --------------------
    # Epoch: [  9 /  10], Train Loss: [0.342], Accuracy:  1.000
    # epoch time: 15763.029 ms, per step time: 1751.448 ms
    # --------------------
    # Epoch: [ 10 /  10], Train Loss: [0.337], Accuracy:  1.000
    # epoch time: 16540.558 ms, per step time: 1837.840 ms
    # ================================================================================
    # End of validation the best Accuracy is:  1.000, save the best ckpt file in ./best.ckpt
    ```
  - **推理预测**
    ```py
    import mindspore as ms
    from mindvision.classification.models import mobilenet_v2

    net = mobilenet_v2(num_classes=2, resize=224)
    _ = ms.load_checkpoint('./best.ckpt', net=net)
    ade5bdbbdf1d54c4561aa41511525855 = ms.train.Model(net)

    def predict(model, image):
        index_2_label = {0: "croissants", 1: "dog"}
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        img = plt.imread(image)
        input_img = (ms.Tensor(img).astype(ms.float32) - mean) / std
        input_img = ms.ops.ResizeBilinear([224, 224])(input_img.transpose([2, 0, 1]).expand_dims(0))
        pred = model.predict(input_img)[0]
        return index_2_label[pred.asnumpy().argmax()], ms.ops.Softmax()(pred)

    DATASET_PATH = os.path.expanduser('~/mindvision/dataset/')
    predict(model, os.path.join(DATASET_PATH, 'DogCroissants/infer/croissants.jpg'))
    # ('croissants', Tensor(shape=[2], dtype=Float32, value= [9.25517678e-001, 7.44823441e-002]))

    predict(model, os.path.join(DATASET_PATH, 'DogCroissants/infer/dog.jpg'))
    # ('dog', Tensor(shape=[2], dtype=Float32, value= [4.49560225e-001, 5.50439715e-001]))
    ```
  - **模型导出** file_format 目前支持 'AIR' / 'ONNX' / 'MINDIR'，默认 'AIR'
    - CheckPoint：采用了Protocol Buffers机制，存储了网络中的所有的参数值。一般用于训练任务中断后恢复训练，或训练后的微调（Fine Tune）任务中。
    - AIR：全称Ascend Intermediate Representation，是华为定义的针对机器学习所设计的开放式的文件格式，同时存储了网络结构和权重参数值，能更好地适配Ascend AI处理器。一般用于Ascend 310上执行推理任务。
    - ONNX：全称Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，同时存储了网络结构和权重参数值。一般用于不同框架间的模型迁移或在推理引擎(TensorRT)上使用。
    - MindIR：全称MindSpore IR，是MindSpore的一种基于图表示的函数式IR，定义了可扩展的图结构以及算子的IR表示，同时存储了网络结构和权重参数值。它消除了不同后端的模型差异，一般用于跨硬件平台执行推理任务，比如把在Ascend 910训练好的模型，放在Ascend 310、GPU以及MindSpore Lite端侧上执行推理。
    ```py
    import mindspore as ms
    from mindvision.classification.models import mobilenet_v2

    net = mobilenet_v2(num_classes=2, resize=224)
    _ = ms.load_checkpoint('./best.ckpt', net=net)

    input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
    ms.export(net, ms.Tensor(input_np), file_name="mobilenet", file_format="MINDIR")

    from mindspore import export, Tensor

    # 定义并加载网络参数
    net = mobilenet_v2(num_classes=2, resize=224)
    param_dict = load_checkpoint("best.ckpt")
    load_param_into_net(net, param_dict)

    # 将模型由ckpt格式导出为MINDIR格式
    export(net, Tensor(input_np), file_name="mobilenet_v2_1.0_224", file_format="MINDIR")
    ```
## 手机侧推理与部署
  - [全场景推理框架 MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/converter_tool.html)
  - [下载MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/downloads.html)
  - **转换文件格式** 使用端侧应用的转换工具 `MindSpore Lite Converter`，将训练过程当中生成的 `{model_name}.mindir` 文件，转换为 `MindSpore Lite` 端侧推理框架可识别的文件格式 `{model_name}.ms` 文件
    ```sh
    # Linux
    # 下载解压后设置软件包的路径，{converter_path}为解压后工具包的路径，PACKAGE_ROOT_PATH为设置的环境变量
    export PACKAGE_ROOT_PATH={converter_path}

    # 将转换工具需要的动态链接库加入到环境变量LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}

    # 在mindspore-lite-linux-x64/tools/converter/converter执行转换命令
    ./converter_lite --fmk=MINDIR --modelFile=mobilenet_v2_1.0_224.mindir  --outputFile=mobilenet_v2_1.0_224
    ```
    ```sh
    # Windows
    # 下载解压后设置软件包的路径，{converter_path}为解压后工具包的路径，PACKAGE_ROOT_PATH为设置的环境变量
    set PACKAGE_ROOT_PATH={converter_path}

    # 将转换工具需要的动态链接库加入到环境变量PATH
    set PATH=%PACKAGE_ROOT_PATH%\tools\converter\lib;%PATH%

    # 在mindspore-lite-win-x64\tools\converter\converter路径下执行转换命令
    call converter_lite --fmk=MINDIR --modelFile=mobilenet_v2_1.0_224.mindir --outputFile=mobilenet_v2_1.0_224
    ```
    转换成功后打印 `CONVERT RESULT SUCCESS:0`，且在当前目录下生成 `{model_name}.ms` 文件
***

# MindSpore 进阶教程
## 线性拟合
