# GTSRB
  ```py
  from glob2 import glob
  from tqdm import tqdm
  from PIL import Image

  """ Train """
  aa = glob('Final_Training/Images/*/*.ppm')
  target_dir = 'Final_Training_png'
  for ii in tqdm(aa):
      label = str(int(os.path.basename(os.path.dirname(ii))))
      file_name = os.path.basename(ii)
      save_path = os.path.join(target_dir, label)
      if not os.path.exists(save_path):
          os.makedirs(save_path)
      save_name = os.path.join(save_path, file_name.replace('.ppm', '.png'))
      # print(ii, label, save_name)
      image = Image.open(ii)
      image.save(save_name)

  """ Test """
  ground_truth = pd.read_csv('GT-final_test.csv', sep=';').set_index('Filename')['ClassId']
  aa = glob('Final_Test/Images/*.ppm')
  target_dir = 'Final_Test_png'
  for ii in tqdm(aa):
      file_name = os.path.basename(ii)
      label = str(ground_truth[file_name])
      save_path = os.path.join(target_dir, label)
      if not os.path.exists(save_path):
          os.makedirs(save_path)
      save_name = os.path.join(save_path, file_name.replace('.ppm', '.png'))
      image = Image.open(ii)
      image.save(save_name)

  !python custom_dataset_script.py --train_images ../../datasets/GTSRB/Final_Training_png/ --test_images ../../datasets/GTSRB/Final_Test_png/ -s GTSRB --is_int_label

  """ Try loading """
  from tqdm import tqdm
  from keras_cv_attention_models.imagenet import data
  tt = data.recognition_dataset_from_custom_json('GTSRB.json')
  aa = [plt.imread(ii['image']).shape for ii in tt['train'].as_numpy_iterator()]

  plt.hist(aa[:, 0], bins=100, alpha=0.5, label='height')
  plt.hist(aa[:, 1], bins=100, alpha=0.5, label='width')
  plt.legend()
  plt.tight_layout()
  ```
## Custom model
```py
from keras_cv_attention_models.common_layers import conv2d_no_bias, batchnorm_with_activation

def custom_model(input_shape=(32, 32, 3), num_classes=43, activation='relu', pretrained=None, classifier_activation="softmax", **kwargs):
    inputs = keras.layers.Input(input_shape)
    nn = inputs

    down_sample_layers = [8]
    for id in range(12):
        if len(down_sample_layers) > 0 and id == down_sample_layers[0]:
            strides = 2
            down_sample_layers = down_sample_layers[1:]
        else:
            strides = 1

        nn = conv2d_no_bias(nn, 64, kernel_size=3, strides=strides, padding="SAME", name="blocks{}_".format(id + 1))
        nn = batchnorm_with_activation(nn, activation=activation, name="{}_".format(id + 1))

    nn = conv2d_no_bias(nn, 12, kernel_size=1, name="out1_")
    nn = tf.transpose(nn, [0, 3, 1, 2])  # To channel first
    nn = keras.layers.Flatten()(nn)
    nn = batchnorm_with_activation(nn, activation=activation, name="out1_")
    nn = keras.layers.Dropout(0.3)(nn)
    nn = keras.layers.Dense(1000, name="out2_dense")(nn)
    nn = batchnorm_with_activation(nn, activation=activation, name="out2_")
    nn = keras.layers.Dropout(0.3)(nn)

    output = keras.layers.Dense(num_classes, activation=classifier_activation, name="output")(nn)
    return keras.models.Model(inputs, output, name="plain")
```
```sh
TF_XLA_FLAGS="--tf_xla_auto_jit=2" CUDA_VISIBLE_DEVICES='0' ./train_script.py \
-m plain_imagenet.h5 -d GTSRB.json -i 32 -b 512 --lr_decay_steps 32 \
-p adamw --lr_base_512 0.002 --weight_decay 0.1 \
--bce_threshold 1 --label_smoothing 0 \
--mixup_alpha 0 --cutmix_alpha 0 --random_crop_min 1 --magnitude 0
```
```py
sys.path.append('../keras_cv_attention_models/')
from keras_cv_attention_models.imagenet import eval_func
import models
mm = keras.models.load_model('checkpoints/test_112.h5', compile=False)
eval_func.evaluation(mm, '../keras_cv_attention_models/GTSRB.json', rescale_mode='torch')
```
```py
tag='ccyy 构造conv model, 带惩罚，不带cutmix，dropout=0.3,weight_decay=1e-1,改conv结构:变动stride2 层位置，扩大感受野 从23到31 '
print(tag)
##############准确率99,37%，  改变stride2 位置 感受野从31到33，准确率不如以前; 带cutmix， 准确率不如从前

import copy

import torch
from torch import nn
import numpy as np
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import default_collate
import math
import os
from torch import Tensor
from typing import Tuple
from torchvision.transforms import functional as F1

class Cc2 (torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2,2), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1), padding = (1,1),bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.Conv2d(64,12,kernel_size=(1,1),bias=False),
            nn.Flatten(),
            nn.BatchNorm1d(3072),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(3072,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1000,43)
        )

    def forward(self,x):
        out= self.block1(x)
        return out


device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
print(device)
model_cy2=Cc2()


train_samples = torch.load('./train_samples_aug_6.pt')
train_samples={'train': train_samples}
test_samples = torch.load('./test.pt')


train_loader = DataLoader(train_samples['train'], batch_size=1024, shuffle=True)
test_loader= DataLoader(test_samples['test'],batch_size=256, shuffle=False)


opti_cy2= torch.optim.Adam(model_cy2.parameters(),lr=1e-3,weight_decay = 1e-1)
sche2=torch.optim.lr_scheduler.StepLR(opti_cy2,step_size=100,gamma=0.5)
loss_total2=[]
acc_total2=[0]
acc_total_t2=[0]
acc_total_t3=[0]
lr_total2=[]
loss_record2=[]
acc_record2=[]

model_cy2 = model_cy2.to(device)

best_model2 = copy.deepcopy(model_cy2.state_dict())

model_save = {'state': best_model2, 'acc': acc_total[-1]}
torch.save(model_save, './best_ccyy_conv2.pt')
iter2=0

for epoch in range(30):
    model_cy2.train()
    run_loss2=0
    run_correct2=0
    step2 = 0
    for inputs, lables in train_loader:
        inputs = inputs.to(device)
        lables = lables.to(device)
        opti_cy2.zero_grad()
        outputs= model_cy2(inputs)


        loss= F.cross_entropy(outputs,lables)
        preds = torch.max(outputs,1)[1]

        loss.backward()

        lr_total.append(opti_cy2.param_groups[0]['lr'])
        opti_cy2.step()

        sche2.step()
        step2=step2+1
        iter2=iter2+1
        # print('preds',preds.shape,'lab',lables.shape)
        correct2=torch.sum(preds == lables)
        # print (correct)

        # print('step {} : loss:{},correct:{}'.format(step,loss,correct))
        # print('lr:{}'.format(opti_cy.param_groups[0]['lr']))

        loss_perstep2=loss.item()*inputs.size(0)
        run_loss2 += loss.item()*inputs.size(0)
        run_correct2 += torch.sum(preds == lables)
        loss_record2.append((iter2,loss_perstep2))
        acc_record2.append((iter2,correct2/inputs.size(0)))

    loss_total2.append(run_loss2)
    acc_total2.append(run_correct2/len(train_samples['train']))

    print('epoch number:{}'.format(epoch),
          'acc_total2:{}'.format(acc_total2[-1]))    

    model_cy2.eval()


    run_correct_t2=0

    i_n_w=[]
    p_n_w=[]
    l_n_w=[]
    i_n_r=[]
    p_n_r=[]
    l_n_r=[]

    step_t2 = 0
    for inputs, lables in test_loader:
        inputs = inputs.to(device)
        lables = lables.to(device)
        outputs= model_cy2(inputs)
        preds = torch.max(outputs,1)[1]
        step_t2=step_t2+1
        # print('labels:',lables)
        # print('preds:',preds)
        correct_t2=torch.sum(preds == lables)
        # print('step:{} correct:{}'.format(step_t,correct_t))

        run_correct_t2 += torch.sum(preds == lables)
    acc_total_t2.append(run_correct_t2/len(test_samples['test']))



    if acc_total_t2[-1] >= max(acc_total_t2):
        best_model2 = copy.deepcopy(model_cy2.state_dict())
        model_save2={'state':best_model2,'acc':acc_total2[-1],
                    'record_loss':loss_record2,'record_acc':acc_record2,
                    'loss_perepoch':loss_total2,'acc_perepoch':acc_total2,
                    'record_lr':lr_total2,'acc_t':acc_total_t2[-1],
                     'acc_t_perepoch':acc_total_t2}

        torch.save(model_save2,'./best_ccyy_conv2.pt')

    print('epoch number:{}'.format(epoch),
          'acc_total_t2:{}'.format(acc_total_t2[-1]))
```
