In [maxvit.py class Attention #L216-L238](https://github.com/google-research/maxvit/blob/main/maxvit/models/maxvit.py#L216-L238) implementation, we have `self.relative_bias = self.add_weight(...)` added as weight first, then rearranged `self.reindexed_bias = attn_utils.reindex_2d_einsum_lookup(...)` in `build`, not in `call`. I'm not sure about this, but in my understanding, the second rearranged `self.reindexed_bias` is a copy of the first initialized weights, and thus the original weights will not get updated during training. I think the `self.reindexed_bias = attn_utils.reindex_2d_einsum_lookup(...)` operation should be moved into `def call` block, like into [maxvit.py#L258](https://github.com/google-research/maxvit/blob/df590d74b615d2f1a8e52f95490168254aae6443/maxvit/models/maxvit.py#L258).

```py
from keras_cv_attention_models.imagenet import data
test_dataset = data.init_dataset('imagenet2012', batch_size=128, rescale_mode='raw')[1]
aa, bb = test_dataset.as_numpy_iterator().next()
for id, ii in enumerate(aa):
    plt.imsave("{}.jpg".format(id), ii.astype('uint8'))
```
```py
import os
import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import models, datasets, transforms as T
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# mobilenet_v2 = models.resnet50(pretrained=True)

image_height, image_width = 224, 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
_ = mobilenet_v2(x)

# Export the model
torch.onnx.export(mobilenet_v2,              # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "mobilenet_v2_float.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names


def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data


def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, image_height, image_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


# change it to your real calibration data set
calibration_data_folder = "calibration_imagenet"
dr = MobilenetDataReader(calibration_data_folder)

quantize_static('mobilenet_v2_float.onnx', 'mobilenet_v2_uint8.onnx', dr)
```
```py
from keras_cv_attention_models.imagenet import eval_func

eval_func.evaluation('mobilenet_v2_uint8.onnx', input_shape=[224, 224, 3])
```
