# GTSRB
  ```py
  import torchvision

  torchvision.datasets.GTSRB('.', download=True)
  torchvision.datasets.GTSRB('.', split='test', download=True)

  from glob2 import glob
  from tqdm import tqdm
  from PIL import Image

  """ Train """
  aa = glob('gtsrb/GTSRB/Training/*/*.ppm')
  target_dir = 'gtsrb/train'
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
  ground_truth = pd.read_csv('gtsrb/GT-final_test.csv', sep=';').set_index('Filename')['ClassId']
  aa = glob('gtsrb/GTSRB/Final_Test/Images/*.ppm')
  target_dir = 'gtsrb/test'
  for ii in tqdm(aa):
      file_name = os.path.basename(ii)
      label = str(ground_truth[file_name])
      save_path = os.path.join(target_dir, label)
      if not os.path.exists(save_path):
          os.makedirs(save_path)
      save_name = os.path.join(save_path, file_name.replace('.ppm', '.png'))
      image = Image.open(ii)
      image.save(save_name)

  !python custom_dataset_script.py --train_images gtsrb/train/ --test_images gtsrb/test/
  # >>>> total_train_samples: 26640, total_test_samples: 12630, num_classes: 43
  # >>>> Saved to: gtsrb.json

  """ Disply heught and width distribution """
  from keras_cv_attention_models.imagenet import data
  tt = data.build_custom_dataset('gtsrb.json')
  aa = np.array([plt.imread(ii['image']).shape for ii in tt['train'].as_numpy_iterator()])

  plt.hist(aa[:, 0], bins=100, alpha=0.5, label='height')
  plt.hist(aa[:, 1], bins=100, alpha=0.5, label='width')
  plt.legend()
  plt.tight_layout()

  """ Show """
  from keras_cv_attention_models.imagenet import data
  tt = data.init_dataset('../datasets/gtsrb/recognition.json', input_shape=(64, 64), magnitude=-1)[0]
  import json
  bb = json.load(open('../datasets/gtsrb/recognition.json'))
  ax = data.show_batch_sample(tt, indices_2_labels=bb['indices_2_labels'])
  ax.figure.savefig('aa.jpg')
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
***

## Multi labels
"0": "20_speed",
"1": "30_speed",
"2": "50_speed",
"3": "60_speed",
"4": "70_speed",
"5": "80_speed",
"6": "80_lifted",
"7": "100_speed",
"8": "120_speed",
"9": "no_overtaking_general",
"10": "no_overtaking_trucks",
"11": "right_of_way_crossing",
"12": "right_of_way_general",
"13": "give_way",
"14": "stop",
"15": "no_way_general",
"16": "no_way_trucks",
"17": "no_way_one_way",
"18": "attention_general",
"19": "attention_left_turn",
"20": "attention_right_turn",
"21": "attention_curvy",
"22": "attention_bumpers",
"23": "attention_slippery",
"24": "attention_bottleneck",
"25": "attention_construction",
"26": "attention_traffic_light",
"27": "attention_pedestrian",
"28": "attention_children",
"29": "attention_bikes",
"30": "attention_snowflake",
"31": "attention_deer",
"32": "lifted_general",
"33": "turn_right",
"34": "turn_left",
"35": "turn_straight",
"36": "turn_straight_right",
"37": "turn_straight_left",
"38": "turn_right_down",
"39": "turn_left_down",
"40": "turn_circle",
"41": "lifted_no_overtaking_general",
"42": "lifted_no_overtaking_trucks"
```py
labels = """0\tCOLOR: red\tSHAPE: circle\tDETAIL: 20
1\tCOLOR: red\tSHAPE: circle\tDETAIL: 30
2\tCOLOR: red\tSHAPE: circle\tDETAIL: 50
3\tCOLOR: red\tSHAPE: circle\tDETAIL: 60
4\tCOLOR: red\tSHAPE: circle\tDETAIL: 70
5\tCOLOR: red\tSHAPE: circle\tDETAIL: 80
6\tCOLOR: black\tSHAPE: circle\tDETAIL: 80
7\tCOLOR: red\tSHAPE: circle\tDETAIL: 100
8\tCOLOR: red\tSHAPE: circle\tDETAIL: 120
9\tCOLOR: red\tSHAPE: circle\tDETAIL: car car
10\tCOLOR: red\tSHAPE: circle\tDETAIL: truck car
11\tCOLOR: red\tSHAPE: triangle\tDETAIL: crossing
12\tCOLOR: yellow\tSHAPE: square\tDETAIL: white
13\tCOLOR: red\tSHAPE: triangle\tDETAIL: white
14\tCOLOR: red\tSHAPE: octagon\tDETAIL: stop
15\tCOLOR: red\tSHAPE: circle\tDETAIL: white
16\tCOLOR: red\tSHAPE: circle\tDETAIL: truck white
17\tCOLOR: red\tSHAPE: circle\tDETAIL: rectangle white
18\tCOLOR: red\tSHAPE: triangle\tDETAIL: exclamation
19\tCOLOR: red\tSHAPE: triangle\tDETAIL: left
20\tCOLOR: red\tSHAPE: triangle\tDETAIL: right
21\tCOLOR: red\tSHAPE: triangle\tDETAIL: curvy
22\tCOLOR: red\tSHAPE: triangle\tDETAIL: bumper
23\tCOLOR: red\tSHAPE: triangle\tDETAIL: slippery car
24\tCOLOR: red\tSHAPE: triangle\tDETAIL: bottleneck
25\tCOLOR: red\tSHAPE: triangle\tDETAIL: construction person
26\tCOLOR: red\tSHAPE: triangle\tDETAIL: traffic light
27\tCOLOR: red\tSHAPE: triangle\tDETAIL: person
28\tCOLOR: red\tSHAPE: triangle\tDETAIL: children person
29\tCOLOR: red\tSHAPE: triangle\tDETAIL: bike
30\tCOLOR: red\tSHAPE: triangle\tDETAIL: snow
31\tCOLOR: red\tSHAPE: triangle\tDETAIL: deer
32\tCOLOR: black\tSHAPE: circle\tDETAIL: white
33\tCOLOR: blue\tSHAPE: circle\tDETAIL: right
34\tCOLOR: blue\tSHAPE: circle\tDETAIL: left
35\tCOLOR: blue\tSHAPE: circle\tDETAIL: straight
36\tCOLOR: blue\tSHAPE: circle\tDETAIL: straight right
37\tCOLOR: blue\tSHAPE: circle\tDETAIL: straight left
38\tCOLOR: blue\tSHAPE: circle\tDETAIL: right down
39\tCOLOR: blue\tSHAPE: circle\tDETAIL: left down
40\tCOLOR: blue\tSHAPE: circle\tDETAIL: circle
41\tCOLOR: black\tSHAPE: circle\tDETAIL: car car
42\tCOLOR: black\tSHAPE: circle\tDETAIL: truck, car
"""

import json
aa = json.load(open('datasets/gtsrb/recognition.json'))
dd = {int(ii.split('\t')[0]): ii.split('\t')[1:] for ii in labels.split('\n')}

bb = ['base_path\t{}'.format(aa['info']['base_path'])]
bb.append('TEST\tTEST')

train = []
for ii in aa['train']:
    key = ii['image']
    for label in dd[int(ii["label"])]:
        train.append('{}\t{}'.format(key, label))
test = []
for ii in aa['test']:
    key = ii['image']
    for label in dd[int(ii["label"])]:
        test.append('{}\t{}'.format(key, label))
print(f"{len(train) = }, {len(test) = }")
# len(train) = 79920, len(test) = 37890

np.random.shuffle(train)
np.random.shuffle(test)
bb = ['base_path\t{}'.format(aa['info']['base_path'])] + train + ['TEST\tTEST'] + test
with open('datasets/gtsrb/captions_detail.tsv', 'w') as ff:
    ff.write('\n'.join(bb))

colors = np.unique([ii[0] for ii in dd.values()]).tolist()
# ['COLOR: black', 'COLOR: blue', 'COLOR: red', 'COLOR: yellow']
shapes = np.unique([ii[1] for ii in dd.values()]).tolist()
# ['SHAPE: circle', 'SHAPE: octagon', 'SHAPE: square', 'SHAPE: triangle']
details = np.unique([ii[2] for ii in dd.values()]).tolist()
[
    'DETAIL: 100', 'DETAIL: 120', 'DETAIL: 20', 'DETAIL: 30', 'DETAIL: 50', 'DETAIL: 60', 'DETAIL: 70', 'DETAIL: 80',
    'DETAIL: bike', 'DETAIL: bottleneck', 'DETAIL: bumper', 'DETAIL: car car', 'DETAIL: children person', 'DETAIL: circle',
    'DETAIL: construction person', 'DETAIL: crossing', 'DETAIL: curvy', 'DETAIL: deer', 'DETAIL: exclamation', 'DETAIL: left',
    'DETAIL: left down', 'DETAIL: person', 'DETAIL: rectangle white', 'DETAIL: right', 'DETAIL: right down', 'DETAIL: slippery car',
    'DETAIL: snow', 'DETAIL: stop', 'DETAIL: straight', 'DETAIL: straight left', 'DETAIL: straight right', 'DETAIL: traffic light',
    'DETAIL: truck car', 'DETAIL: truck white', 'DETAIL: truck, car', 'DETAIL: white'
]
```

black -l 120 --skip-string-normalization ait/components/benchmark/test/generate_add_model.py ait/components/benchmark/test/ST_SIMPLE/test_infer_resnet50.py ait/components/benchmark/test/UT_SIMPLE/
black -l 120 --skip-string-normalization ait/components/benchmark/test/ST/test_result.py
