```py
CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py --seed 0 -m coatnet.CoAtNet0 --batch_size 128 -i 224 --lr_decay_steps 300 --magnitude 15 --additional_model_kwargs'{"drop_connect_rate": 0.3}' -s CoAtNet1_224  --restore_path checkpoints/CoAtNet1_224_latest.h5 -I 57

CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py --seed 0 -b 128 -i 160 --label_smoothing 0.1 --bce_threshold 1 --lr_base_512 0.000375 --weight_decay 0.05 --magnitude 15 --mixup_alpha 0.8 --cutmix_alpha 0 -r checkpoints/maxvit.MaxViT_Tiny_160_adamw_imagenet2012_batchsize_128_randaug_15_mixup_0.8_cutmix_0.0_RRC_0.08_lr512_0.000375_wd_0.05_latest.h5 -I 50
```
- **onnxruntime output all same labels within a batch on ARM**
```py
import onnxruntime
mm = onnxruntime.InferenceSession('resnet50_pytorch_1.4.onnx')
output_names = [mm.get_outputs()[0].name]

from glob2 improt glob
from skimage.transform import resize
aa = np.random.choice(glob('../datasets/val/*/*.JPEG'), 16)
print("True labels:", [ii.split(os.path.sep)[-2] for ii in aa])

mean = np.array([0.079, 0.05, 0]) + 0.406
std = np.array([0.005, 0, 0.001]) + 0.224
imgs = np.array([(resize(plt.imread(ii), [224, 224]) - mean) / std for ii in aa]).astype("float32")
preds = mm.run(output_names, {mm.get_inputs()[0].name: imgs.transpose([0, 3, 1, 2])})[0]

print("Predicted labels:", preds.argmax(-1))
```
```py
from keras_cv_attention_models.imagenet import data
test_dataset = data.init_dataset('imagenet2012', batch_size=128, rescale_mode='raw')[1]
aa, bb = test_dataset.as_numpy_iterator().next()
for id, ii in enumerate(aa):
    plt.imsave("{}.jpg".format(id), ii.astype('uint8'))
```
```py
import torch
from torchvision import models, datasets

mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# mobilenet_v2 = models.resnet50(pretrained=True)

image_height, image_width = 224, 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)
_ = mobilenet_v2(x)

import os
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms as T
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

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
    def __init__(self, calibration_image_folder, input_name="input", input_height=224, input_width=224):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.input_name, self.input_height, self.input_width = input_name, input_height, input_width

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, self.input_height, self.input_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{self.input_name: nhwc_data} for nhwc_data in nhwc_data_list])
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
