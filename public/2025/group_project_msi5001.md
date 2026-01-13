## Presentation
We have a mission of animal sound classification. The files are our code and report
- `ipynb` files for the code of each sub-task
- `pdf` for the execution steps.
- `MSI5001_Team3_Project Report.docx` for the final report
- `train_eval_log.md` is the actual train and evaluation logs for many of my tests.
- `Animal_Sounds_Classifying_Audio_Files.pptx` is our completed slides for my parts. Please notice the diagram and charts there.

For me it's the GRU with sequential feature, transfer learning using YAMNet and Wav2vec2, efficientnet part. Then a comparison of traditional vector feature, 1D raw feature, transfer learning and the final image feature using efficientnet. Other classmates while present other parts, like dataset introduction, data prepossessing, traditional machine learning methods

- Some presentation requirement:
  - Slide Design: Slides are visually clean, with readable fonts, meaningful headings, and minimal clutter. Visuals aid understanding.
  - Flow: Problem → Data → Model → Results → Insights. Your presentation should follow a clear and logical flow, starting from the problem definition, followed by an overview of the dataset, explanation of the model(s) used, presentation of the results, and concluding with key insights or takeaways.
  - Delivery: All team members actively participate. Presenters speak clearly, stay within time limits, and demonstrate a strong grasp of the content.
  - Q&A: The team responds to questions with clarity, thoughtfulness, and evidence.

Now help me prepare a presentation script I can follow. Could be a little technology detailed.

## **Slide 1 – GRU Model with Sequential Features**
  - **Goal:** Show how temporal modeling was tested and why it plateaued.
  - **Content:**
    * **Title:** “MFCC Sequence + BiGRU Model: Temporal Modeling of Audio”
    * **Visuals:**
      * Diagram of MFCC sequence → BiGRU → Attention → Softmax.
      * Training vs validation accuracy/loss curve from `train_eval_log.md`.
    * **Key points:**
      * Input: MFCC frames (T × 60), 2× BiGRU + Attention.
      * Captures time-dependent patterns.
      * Results: Val F1 ≈ 0.86, Test Acc ≈ 0.89.
      * Limitation: Overfitting due to small dataset; temporal gain minor vs pooled features.
## **Slide 2 – Transfer Learning with YAMNet and Wav2Vec2**
  - **Goal:** Emphasize the benefit of pretrained representations.
  - **Content:**
    * **Title:** “Pretrained Embeddings for General Sound Understanding”
    * **Visuals:**
      * Flowchart: Audio → YAMNet → 2048-D Vector → SVM.
        (Optional: Wav2Vec2 branch.)
      * Comparison bar chart: A1 vs A2 accuracy (0.86 → 0.90).
    * **Key points:**
      * YAMNet embeddings trained on AudioSet → strong domain match.
      * Wav2Vec2 (mainly speech) → lower generalization.
      * Transfer learning adds ≈ 4–5 % absolute gain over handcrafted features.
      * Efficient for limited data tasks.
## **Slide 3 – EfficientNet on Mel-Spectrogram Images**
  - **Goal:** Show how image-based formulation achieved best results.
  - **Content:**
    * **Title:** “Mel-Spectrogram Images + EfficientNet + GRU (AKA CRNN)”
    * **Visuals:**
      * Architecture: Mel-Spec → EfficientNet-B0 → GRU + Attention → Softmax.
      * Training curve and confusion matrix (95.6 % accuracy).
    * **Key points:**
      * Converts audio to 2D representation (5 s window, 128 mels).
      * Combines frequency and temporal patterns.
      * Outperforms others by ≈ 6 % absolute accuracy.
      * Tested different durations (3–12 s) → 5 s best balance.
## **Slide 4 – Comparative Summary**
  - **Goal:** Consolidate results and insights across approaches.

  | Model                | Input Type         | Val F1 | Test Acc  | Comment         |
  | :------------------- | :----------------- | :----- | :-------- | :-------------- |
  | Traditional SVM (A1) | Handcrafted vector | 0.84   | 0.87      | Baseline robust |
  | YAMNet + SVM (A2)    | Pretrained vector  | 0.85   | 0.90      | Strong transfer |
  | BiGRU Seq (B)        | MFCC sequence      | 0.84   | 0.89      | Temporal model  |
  | CRNN (C)             | Mel-Spec image     | 0.93   | 0.95–0.96 | Best overall    |

  - **Takeaways:**
    * Image formulation > Temporal > Statistical.
    * Temporal modeling adds small gain but limited by sample size.
    * Pretrained features reduce data dependence. Dataset size (~575) → benefits most from transfer learning.
    * Spectrogram as image bridges audio and vision domains. EfficientNet (frozen) + GRU = state-of-the-art design for small datasets.
## **Slide 5 – Deployment Demo: Animal Sound Recognizer**
  - **Goal:** Demonstrate real-world application.
  - **Content**
    * **Title:** “Model Deployment — Web Interface”
    * Browser-based demo built with FastAPI + MediaRecorder.
    * Records or uploads short clips (≤ 3 s) for instant animal-sound classification.
    * Converts audio to Mel-spectrogram → EfficientNet-B0 → live confidence display.
    * End-to-end pipeline: record → predict → view results.
    * Deployable on AWS (Lightsail / S3 + CloudFront).
***

# Presentation:
This is another group project for MSI5001 in NUS, Power Transformer Oil Temperature Prediction - A Multi-Horizon Time-Series Forecasting Project. The files uploaded include background introduction, experiment details, process, results, and the final report `MSI5001_Team16_Project_Report.pdf`, which is the most important one. We have four people in our project, and I'm in charge of presenting the problem introduction and data processing parts. The `Transformer Oil Temperature Prediction.pptx` is the actual slides. Help prepare a presentation transcript:

## Other presentation requirements:
Slide Design: Slides are visually clean, with readable fonts, meaningful headings, and minimal clutter. Visuals aid understanding.
• Flow: Problem → Data → Model → Results → Insights. Your presentation should follow a clear and logical flow, starting from the problem definition, followed by an overview of the dataset, an explanation of the model(s) used, a presentation of the results, and concluding with key insights or takeaways.
• Delivery: All team members actively participate. Presenters speak clearly, stay within time limits, and demonstrate a strong grasp of the content.
• Q&A: The team responds to questions with clarity, thoughtfulness, and evidence

***
We have a webpage ready, firstly, help create a python backend with the webpage could be used on AWS server that:
Entire function is recognizing animal sound. Could also display the sound wave on the webpage.

1. read user sound input by recording or uploading
2. upload to the same AWS server
3. call a backend model prediction, actual model usage is showing below.
4. return user the recognition result and probability.

|                                   | macro avg f1-score | accuracy |
| --------------------------------- | ------------------ | -------- |
| orginal (hidden 128, dropout 0.2) | 0.72               | 0.7739   |
| + hidden 256                      | 0.86               | 0.8957   |
| + Global CMVN                     | 0.80               | 0.8435   |
| ++ dropout 0.3                    | 0.79               | 0.8435   |
| ++ 2×BiGRU                        | 0.80               | 0.8609   |


| Model              | Params | FLOPs/s audio | Main Domain            | accuracy | macro avg f1-score |
|:------------------ |:------:|:-------------:|:---------------------- | -------- | ------------------ |
| YAMNet             | ~4.3 M |  ~0.5 GFLOPs  | Environmental sounds   | 0.9043   | 0.87               |
| Wav2Vec2-Base-960h | ~95 M  |  ~4–6 GFLOPs  | Speech / general audio | 0.7569   | 0.69               |


| strategy                | macro avg f1-score | accuracy |
| ----------------------- | ------------------ | -------- |
| orginal                 | 0.90               | 0.9478   |
| +Weighted Loss Function | 0.92               | 0.9478   |
| ++heavier augmenter     | 0.91               | 0.9391   |
| ++GRU                   | 0.93               | 0.9565   |

```py
# heavier augmenter
augmenter = Compose([
    # Increase the amount of noise and probability
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.025, p=0.6),
    # Widen the range for stretching/compressing time
    TimeStretch(min_rate=0.75, max_rate=1.35, p=0.6),
    # Widen the range for pitch shifting
    PitchShift(min_semitones=-5, max_semitones=5, p=0.6),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

# lighter augmenter
# A more balanced augmentation pipeline
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-3, max_semitones=3, p=0.5), # Reduced range
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.4), # Reduced probability
])
```
```py
import torch, timm, librosa
import numpy as np

class CFG:
    # Data parameters
    SAMPLE_RATE = 32000  # Higher sample rate for more detail
    DURATION = 3  # Fixed duration in seconds to handle variable length
    TARGET_LENGTH = DURATION * SAMPLE_RATE
    HOP_LENGTH = 512 # Hop length for the FFT
    N_MELS = 128  # Number of Mel bands
    N_FFT = 2048   # Size of the FFT

    # Model parameters
    NUM_CLASSES = 10
    MODEL_NAME = "efficientnet_b0" # A good balance of size and performance

def audio_to_melspectrogram(audio_wave, is_train=False):
    """
    Handles audio loading, padding/trimming, and conversion to Mel Spectrogram. Applies augmentations if is_train is True.
    """
    # 1) Trim or pad to fixed length, but: - TRAIN: random crop if longer than target, - EVAL:  center crop if longer than target
    L = len(audio_wave)
    if L > CFG.TARGET_LENGTH:
        start = (L - CFG.TARGET_LENGTH) // 2
        audio_wave = audio_wave[start:start + CFG.TARGET_LENGTH]
    else:
        pad = CFG.TARGET_LENGTH - L
        left = pad // 2
        right = pad - left
        audio_wave = np.pad(audio_wave, (left, right), mode='constant')

    # 3) Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_wave, sr=CFG.SAMPLE_RATE, n_fft=CFG.N_FFT, hop_length=CFG.HOP_LENGTH, n_mels=CFG.N_MELS)

    # 4) dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def read_audio_file_as_image(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    if sr != CFG.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=CFG.SAMPLE_RATE)
    mel_spec = audio_to_melspectrogram(y)
    image = torch.tensor(mel_spec, dtype=torch.float32)
    image = image.unsqueeze(0)
    image = image.repeat(3, 1, 1)
    return image

model = timm.create_model(CFG.MODEL_NAME, pretrained=False, num_classes=CFG.NUM_CLASSES)
model = model.eval()
ss = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(ss)
image = read_audio_file_as_image('Audio_Files/Cat/cat_102.wav')
labels = ['Bird', 'Cat', 'Cow', 'Dog', 'Donkey', 'Frog', 'Lion', 'Maymun', 'Sheep', 'Tavuk']
with torch.no_grad():
    pred_arg = model(image[None]).detach().numpy().argmax()
print("pred:", labels[pred_arg])

""" Accuracy test on dataset """

from tqdm import tqdm
y_true, y_pred = [], []
dataset_path = os.path.expanduser('~/msi5001/Audio_Files')
for animal_class in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, animal_class)
    if os.path.isdir(class_dir):
        for filename in tqdm(os.listdir(class_dir), animal_class):
            # all_files.append({"file_path": os.path.join(class_dir, filename), "label": animal_class})
            image = read_audio_file_as_image(os.path.join(class_dir, filename))
            with torch.no_grad():
                pred_arg = model(image[None]).detach().numpy().argmax()
            y_pred.append(labels[pred_arg])
            y_true.append(animal_class)
print("accuracy:", (np.array(y_true) == np.array(y_pred)).sum() / len(y_pred))
```
***
We have a mission of animal sound classification. Help give a draft pytorch model training for this. We are expecting to run it in colab, so make it a structure suitable, like split each method separately and stand along besides a common preprocessing part. The final part should be a thoroughly pytorch implementation with a SOTA model + data augment method + learning rate strategy like cosine + optimizer at least AdamW:

Following is our group project info

## Task Expectation
  - Traditional ML feature extraction technique exploration
  - Current "learn from raw feature" type ML exploration
  - Formulate the audio data as feature vectors, sequences and images --> see how different apporaches perform
  - The samples are of variable length, need to handle this as well
## Dataset Description
  - An audio classification dataset
  - This data consisting of 875 audio files with 10 types of animal sounds (200 cat, 200 dog, 200 bird, 75 cow, 45 lion, 40 sheep, 35 frog, 30 chicken, 25 donkey, 25 monkey sounds)
  - You need to divide each class into 80% training and 20% test data
  - Perform 5-fold cross validation on the training dataset to develop and choose your model
  - The remaining 20% should be kept only for testing the final model
  - file is in wav format.
  ```py
  In [51]: dd.sample_rate.value_counts()
  Out[51]:
  sample_rate
  16000    176
  11025    146
  44100    116
  22050     63
  8000      47
  11127      7
  22000      4
  22255      4
  48000      2
  24000      2
  22254      1
  16393      1
  21276      1
  11000      1
  32000      1
  8012       1
  20000      1
  Name: count, dtype: int64

  In [52]: dd.num_channels.value_counts()
  Out[52]:
  num_channels
  1    552
  2     22
  Name: count, dtype: int64

  In [53]: dd.bits_per_sample.value_counts()
  Out[53]:
  bits_per_sample
  16    347
  8     190
  0      37
  Name: count, dtype: int64

  In [54]: dd.num_frames.describe()
  Out[54]:
  count       574.000000
  mean      66304.592334
  std       65551.885602
  min        2445.000000
  25%       22624.000000
  50%       44100.000000
  75%       85996.000000
  max      441000.000000
  Name: num_frames, dtype: float64
  ```
## My training script
```py

```
training log
```py

```
final testing result
```py

```
***

- [A] Traditional ML
- [ ] learn from raw feature <-
- [A] sequences
- [x] feature vectors
- [x] images
- [x] The samples are of variable length, need to handle this as well <-

***
Epoch 19/20 -> Train Loss: 0.0483, Train Acc: 0.9783 | Val Loss: 0.5214, Val Acc: 0.9348
Epoch 20/20 -> Train Loss: 0.0424, Train Acc: 0.9755 | Val Loss: 0.5037, Val Acc: 0.9348

Fold 5 finished. Best validation loss: 0.4950, accuracy: 0.9348

***

***

爽快直给：下面是一份**可直接跑**的 PyTorch 草案，满足你所有约束——按**每类 80/20**分层划分、在**训练 80% 上做 5 折交叉验证**选择模型/超参，然后只在**最终 20% 测试集**上做一次评估。它同时覆盖三条路线：

1. 传统特征 → **MFCC+GRU（序列）**
2. 原始波形 → **1D CNN（learn-from-raw）**
3. 语谱图图像 → **Mel Spectrogram + 2D CNN（ResNet18，图像）**

同时处理**变长音频**：训练时随机裁剪/填充，验证/测试时中心裁剪/对称填充。用到了 `torchaudio`。

---

### 目录假设
  ```
  dataset_root/
    cat/*.wav
    dog/*.wav
    bird/*.wav
    cow/*.wav
    lion/*.wav
    sheep/*.wav
    frog/*.wav
    chicken/*.wav
    donkey/*.wav
    monkey/*.wav
  ```
### 一份脚本走天下（保存为 `train_animals.py`）
  ```python
  import os, json, math, random, argparse, time, pathlib
  from dataclasses import dataclass
  from typing import List, Tuple, Dict

  import numpy as np
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torch.utils.data import Dataset, DataLoader, Subset
  from sklearn.model_selection import StratifiedKFold
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  import torchaudio
  from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC
  from torchvision.models import resnet18
  from torchvision.ops.misc import Conv2dNormActivation

  # --------------------- 可调参数 ---------------------
  @dataclass
  class Config:
      sample_rate: int = 16000
      mono: bool = True
      target_sec: float = 4.0       # 统一到 4s（训练随机裁剪，验证/测试中心裁剪）
      n_mels: int = 64
      n_mfcc: int = 40
      n_fft: int = 1024
      hop_length: int = 320         # ~20ms
      fmin: int = 20
      fmax: int = 8000
      batch_size: int = 32
      epochs: int = 30
      lr: float = 3e-4
      weight_decay: float = 1e-4
      patience: int = 6             # 早停
      num_workers: int = 4
      seed: int = 27
      device: str = "cuda" if torch.cuda.is_available() else "cpu"

  # --------------------- 实用工具 ---------------------
  def set_seed(seed: int):
      random.seed(seed); np.random.seed(seed)
      torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  def list_wavs_by_class(root: str) -> Tuple[List[str], List[int], Dict[int, str]]:
      classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
      cls2idx = {c:i for i,c in enumerate(classes)}
      paths, labels = [], []
      for c in classes:
          droot = os.path.join(root, c)
          for fn in os.listdir(droot):
              if fn.lower().endswith(".wav"):
                  paths.append(os.path.join(droot, fn))
                  labels.append(cls2idx[c])
      return paths, labels, {v:k for k,v in cls2idx.items()}

  def stratified_per_class_split(paths, labels, train_ratio=0.8, seed=27):
      # 按类别分层 80/20
      rng = np.random.RandomState(seed)
      labels = np.array(labels)
      paths = np.array(paths)
      train_idx, test_idx = [], []
      for cls in np.unique(labels):
          idx = np.where(labels==cls)[0]
          rng.shuffle(idx)
          n_train = int(len(idx)*train_ratio)
          train_idx.extend(idx[:n_train].tolist())
          test_idx.extend(idx[n_train:].tolist())
      return np.array(train_idx), np.array(test_idx)

  def pad_or_crop(wav: torch.Tensor, target_len: int, train: bool) -> torch.Tensor:
      L = wav.shape[-1]
      if L == target_len:
          return wav
      if L > target_len:
          if train:
              start = random.randint(0, L - target_len)
          else:
              start = (L - target_len)//2
          return wav[..., start:start+target_len]
      # L < target_len -> 对称填充
      pad_total = target_len - L
      left = pad_total//2
      right = pad_total - left
      return F.pad(wav, (left, right))

  # --------------------- 数据集 ---------------------
  class AnimalDataset(Dataset):
      def __init__(self, paths, labels, cfg: Config, mode: str, approach: str):
          self.paths = paths
          self.labels = labels
          self.cfg = cfg
          self.mode = mode
          self.approach = approach   # "raw", "mfcc", "mel2d"
          self.resampler = None
          self.mel = MelSpectrogram(
              sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
              f_min=cfg.fmin, f_max=cfg.fmax, n_mels=cfg.n_mels
          )
          self.to_db = AmplitudeToDB()
          self.mfcc = MFCC(
              sample_rate=cfg.sample_rate, n_mfcc=cfg.n_mfcc,
              melkwargs=dict(n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                             f_min=cfg.fmin, f_max=cfg.fmax, n_mels=cfg.n_mels)
          )
          self.target_len = int(cfg.sample_rate * cfg.target_sec)

      def __len__(self): return len(self.paths)

      def _load_wav(self, pth: str) -> torch.Tensor:
          wav, sr = torchaudio.load(pth)   # [C, L]
          if self.cfg.mono and wav.shape[0] > 1:
              wav = wav.mean(0, keepdim=True)
          if sr != self.cfg.sample_rate:
              if self.resampler is None or getattr(self.resampler, "orig_freq", None) != sr:
                  self.resampler = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
              wav = self.resampler(wav)
          # 振幅标准化
          wav = wav / (wav.abs().max() + 1e-9)
          return wav

      def __getitem__(self, i):
          p = self.paths[i]
          y = self.labels[i]
          wav = self._load_wav(p)  # [1, L]
          wav = pad_or_crop(wav, self.target_len, train=(self.mode=="train"))

          if self.approach == "raw":
              # 原始波形 [1, T]
              x = wav
          elif self.approach == "mfcc":
              # MFCC 序列 [n_mfcc, T']
              x = self.mfcc(wav)   # [1, n_mfcc, T']
              x = x.squeeze(0)
          elif self.approach == "mel2d":
              # 语谱图图像 [1, n_mels, T']
              x = self.mel(wav)
              x = self.to_db(x)
          else:
              raise ValueError("Unknown approach")

          return x, y

  def collate_raw(batch):
      xs, ys = zip(*batch)
      xs = torch.stack(xs, dim=0)  # [B, 1, T]
      ys = torch.tensor(ys, dtype=torch.long)
      return xs, ys

  def collate_mfcc(batch):
      # 对时间维做 padding 到同一长度
      xs, ys = zip(*batch)
      maxT = max(x.shape[-1] for x in xs)
      padded = []
      for x in xs:
          T = x.shape[-1]
          if T < maxT:
              x = F.pad(x, (0, maxT-T))
          padded.append(x)
      xs = torch.stack(padded, dim=0)  # [B, n_mfcc, T]
      ys = torch.tensor(ys, dtype=torch.long)
      return xs, ys

  def collate_mel2d(batch):
      # [B, 1, n_mels, T] -> 已经同长（因为波形已对齐）
      xs, ys = zip(*batch)
      xs = torch.stack(xs, dim=0)
      ys = torch.tensor(ys, dtype=torch.long)
      return xs, ys

  # --------------------- 模型 ---------------------
  class Raw1DCNN(nn.Module):
      def __init__(self, n_classes: int):
          super().__init__()
          self.net = nn.Sequential(
              nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
              nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
              nn.Conv1d(64, 128, 9, stride=2, padding=4), nn.BatchNorm1d(128), nn.ReLU(),
              nn.Conv1d(128, 256, 9, stride=2, padding=4), nn.BatchNorm1d(256), nn.ReLU(),
              nn.AdaptiveAvgPool1d(1)
          )
          self.fc = nn.Linear(256, n_classes)

      def forward(self, x):  # x: [B, 1, T]
          z = self.net(x).squeeze(-1)
          return self.fc(z)

  class MFCC_GRU(nn.Module):
      def __init__(self, n_mfcc: int, n_classes: int, hidden: int = 128, num_layers: int = 2, bidir: bool = True):
          super().__init__()
          self.gru = nn.GRU(input_size=n_mfcc, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidir)
          self.fc = nn.Linear(hidden*(2 if bidir else 1), n_classes)

      def forward(self, x):  # [B, n_mfcc, T]
          x = x.transpose(1,2)  # [B, T, n_mfcc]
          out, _ = self.gru(x)  # [B, T, H*dir]
          h = out[:, -1, :]     # 取最后时刻
          return self.fc(h)

  class MelResNet18(nn.Module):
      def __init__(self, n_classes: int):
          super().__init__()
          base = resnet18(weights=None)
          # 将第一层改成接收 1 通道
          base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
          self.backbone = base
          self.backbone.fc = nn.Linear(512, n_classes)

      def forward(self, x):  # [B, 1, n_mels, T]
          return self.backbone(x)

  def build_model(approach: str, n_classes: int, cfg: Config):
      if approach == "raw": return Raw1DCNN(n_classes)
      if approach == "mfcc": return MFCC_GRU(cfg.n_mfcc, n_classes)
      if approach == "mel2d": return MelResNet18(n_classes)
      raise ValueError()

  # --------------------- 训练/验证 ---------------------
  def run_one_epoch(model, loader, optim, cfg: Config, train: bool):
      model.train(train)
      total, correct, loss_sum = 0, 0, 0.0
      for x,y in loader:
          x, y = x.to(cfg.device), y.to(cfg.device)
          with torch.set_grad_enabled(train):
              logits = model(x)
              loss = F.cross_entropy(logits, y)
          if train:
              optim.zero_grad()
              loss.backward()
              nn.utils.clip_grad_norm_(model.parameters(), 5.0)
              optim.step()
          loss_sum += loss.item() * y.size(0)
          pred = logits.argmax(dim=1)
          correct += (pred==y).sum().item()
          total += y.size(0)
      return loss_sum/total, correct/total

  def evaluate(model, loader, cfg: Config):
      model.eval()
      ys, ps = [], []
      with torch.no_grad():
          for x,y in loader:
              x = x.to(cfg.device)
              logits = model(x)
              pred = logits.argmax(dim=1).cpu().numpy().tolist()
              ys.extend(y.numpy().tolist())
              ps.extend(pred)
      acc = accuracy_score(ys, ps)
      return acc, np.array(ys), np.array(ps)

  # --------------------- 主流程 ---------------------
  def main():
      parser = argparse.ArgumentParser()
      parser.add_argument("--data_root", type=str, required=True)
      parser.add_argument("--approach", type=str, default="mel2d",
                          choices=["raw","mfcc","mel2d"])
      parser.add_argument("--epochs", type=int, default=Config.epochs)
      parser.add_argument("--batch_size", type=int, default=Config.batch_size)
      parser.add_argument("--lr", type=float, default=Config.lr)
      parser.add_argument("--outdir", type=str, default="runs_animals")
      args = parser.parse_args()

      cfg = Config()
      cfg.epochs = args.epochs
      cfg.batch_size = args.batch_size
      cfg.lr = args.lr
      set_seed(cfg.seed)
      os.makedirs(args.outdir, exist_ok=True)

      # 数据索引与 80/20 分层划分
      paths, labels, idx2cls = list_wavs_by_class(args.data_root)
      n_classes = len(idx2cls)
      tr_idx, te_idx = stratified_per_class_split(paths, labels, 0.8, seed=cfg.seed)
      tr_paths, tr_labels = [paths[i] for i in tr_idx], [labels[i] for i in tr_idx]
      te_paths, te_labels = [paths[i] for i in te_idx], [labels[i] for i in te_idx]

      # 5 折交叉验证（仅在训练集上）
      skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
      fold_metrics = []
      best_fold = None
      fold_id = 0

      for train_ids, val_ids in skf.split(tr_paths, tr_labels):
          fold_id += 1
          print(f"\n===== Fold {fold_id}/5 using approach={args.approach} =====")
          ds_tr = AnimalDataset([tr_paths[i] for i in train_ids],
                                [tr_labels[i] for i in train_ids], cfg, "train", args.approach)
          ds_va = AnimalDataset([tr_paths[i] for i in val_ids],
                                [tr_labels[i] for i in val_ids], cfg, "val", args.approach)

          if args.approach == "raw":
              collate = collate_raw
          elif args.approach == "mfcc":
              collate = collate_mfcc
          else:
              collate = collate_mel2d

          dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)
          dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

          model = build_model(args.approach, n_classes, cfg).to(cfg.device)
          optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=2, verbose=True)

          best_acc, best_state, wait = 0.0, None, 0
          for ep in range(1, cfg.epochs+1):
              tr_loss, tr_acc = run_one_epoch(model, dl_tr, optim, cfg, train=True)
              va_acc, _, _ = evaluate(model, dl_va, cfg)
              scheduler.step(va_acc)
              print(f"Fold {fold_id} | Epoch {ep:02d} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
              if va_acc > best_acc:
                  best_acc = va_acc
                  best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                  wait = 0
              else:
                  wait += 1
                  if wait >= cfg.patience:
                      print("Early stopping.")
                      break

          fold_metrics.append(best_acc)
          # 记录最优折
          if best_fold is None or best_acc > best_fold["acc"]:
              best_fold = {"acc": best_acc, "state": best_state, "fold": fold_id}

      print("\n===== CV done =====")
      print("Fold val accuracies:", [f"{m:.4f}" for m in fold_metrics])
      print(f"Best fold: {best_fold['fold']} val_acc={best_fold['acc']:.4f}")

      # 用最佳折的权重在「同配置+全训练集80%」上再训练一遍（可选）
      # 简化：直接加载最优折权重做测试（实践中可再全量微调若干 epoch）
      # 准备最终 DataLoader
      ds_tr_full = AnimalDataset(tr_paths, tr_labels, cfg, "val", args.approach)  # mode='val' 关闭随机裁剪
      ds_te = AnimalDataset(te_paths, te_labels, cfg, "test", args.approach)
      if args.approach == "raw":
          collate = collate_raw
      elif args.approach == "mfcc":
          collate = collate_mfcc
      else:
          collate = collate_mel2d
      dl_tr_full = DataLoader(ds_tr_full, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)
      dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

      model = build_model(args.approach, n_classes, cfg).to(cfg.device)
      model.load_state_dict({k: v.to(cfg.device) for k,v in best_fold["state"].items()}, strict=True)

      # 最终评估（只在 20% 测试集）
      tr_acc, y_tr, p_tr = evaluate(model, dl_tr_full, cfg)
      te_acc, y_te, p_te = evaluate(model, dl_te, cfg)
      print(f"\nFinal (80% train portion, eval mode) acc: {tr_acc:.4f}")
      print(f"Final (20% held-out test) acc:       {te_acc:.4f}")
      print("\nClassification report (test):")
      print(classification_report(y_te, p_te, target_names=[idx2cls[i] for i in range(n_classes)]))
      print("Confusion matrix (test):")
      print(confusion_matrix(y_te, p_te))

      # 保存
      torch.save({
          "approach": args.approach,
          "state_dict": model.state_dict(),
          "idx2cls": idx2cls,
          "config": vars(cfg),
          "cv_val_accs": fold_metrics,
          "best_fold": best_fold["fold"],
          "test_acc": float(te_acc),
      }, os.path.join(args.outdir, f"{args.approach}_best.pth"))
      with open(os.path.join(args.outdir, f"{args.approach}_meta.json"), "w") as f:
          json.dump({"idx2cls": idx2cls}, f, indent=2)

  if __name__ == "__main__":
      main()
  ```
### 如何跑
  ```bash
  # 1) 安装
  pip install torch torchaudio torchvision scikit-learn

  # 2) 三种路线分别试
  python train_animals.py --data_root /path/to/dataset_root --approach raw
  python train_animals.py --data_root /path/to/dataset_root --approach mfcc
  python train_animals.py --data_root /path/to/dataset_root --approach mel2d

  # 可调参
  python train_animals.py --data_root ... --approach mel2d --epochs 40 --batch_size 48 --lr 2e-4
  ```
### 设计要点 & 硬核细节
  * **分层 80/20**：不是整体打乱，而是**每类内部**按比例分，这保证了你给的类分布在测试集里也合理。
  * **5 折 CV 只在训练 80% 上做**：避免泄漏；记录每折最佳 `val_acc`，并用最佳折权重进行最终评估（简化版）。实际生产你可以：

    1. 用 CV 找到**方案/超参**；
    2. **用整个 80% 训练集**重新训练若干 epoch；
    3. 最终只在**那 20% 测试集**评估一次。
  * **变长处理**：

    * 训练：随机裁剪到 `target_sec`，有助于泛化；
    * 验证/测试：中心裁剪/对称填充，保证可重复性。
  * **三视角建模**：

    * 向量序列（MFCC+GRU）：传统方法的强 baseline；
    * 原始波形（1D CNN）：端到端，适合充足数据或配合强正则；
    * 图像（Mel+ResNet18）：借力成熟 2D CNN 结构。
  * **评估**：输出 `classification_report` 和**混淆矩阵**，方便看易混类别（比如狗/狼嚎这类在别数据里常混；你这套是十类动物，可具体看 cat/dog/bird 等交叉）。
  * **不平衡**：你的类样本差异（200 vs 25）不算极端，但可进一步加**类别权重**（`CrossEntropyLoss(weight=class_weights)`）、或**WeightedRandomSampler** 以平衡 batch。
  * **数据增强（可选）**：

    * 时域：随机增益、少量噪声、时间遮挡；
    * 频域：SpecAugment（对 mel2d 最有效）。
  * **更强模型（可渐进）**：

    * mel2d 可换成 **CNN + Transformer**（Conformer）或小 ViT；
    * raw 可用 **1D ResNet** 或 **wav2vec2 / Hubert** 这类 SSL 预训练特征（冻结或微调）。
  * **可复现实验**：固定随机种子，关闭 cudnn benchmark，保证 CV 可复现。
***

行，直接给你一份“能跑的样板工程”，覆盖三条路子：
1）传统特征 + 传统ML（MFCC → 统计特征 → SVM/RandomForest）；
2）端到端原始波形（1D CNN）；
3）把音频变成图像（Mel-Spec → 2D CNN）。
包含：按类别分层 80/20 切分（留 20% 仅用于最终测试），在训练集上做 5-fold 交叉验证选最优模型，然后在整训练集上重训并在测试集上评估。可处理变长样本（按需要 pad/trim）。

把下面脚本保存为 `train_animals.py`，数据目录组织为 `data_root/<class_name>/*.wav`（类名用目录名）。需要 `torch torchaudio torchvision scikit-learn`。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animal Sound Classification (PyTorch + torchaudio + sklearn)
- Stratified 80/20 split per class
- 5-fold CV on training split to pick model
- Three approaches:
  1) classical_mfcc_svm   (MFCC统计特征 + SVM/RandomForest)
  2) cnn1d_raw            (原始波形 1D CNN)
  3) cnn2d_melspec        (Mel-Spectrogram 2D CNN)
"""

import os, json, math, random, argparse, warnings, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from torchaudio import transforms as T

# 仅在 classical 模式下需要
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# --------------------------
# Utilities
# --------------------------

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def find_wavs(root):
    root = Path(root)
    samples = []
    for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = cls_dir.name
        for wav in cls_dir.rglob("*.wav"):
            samples.append((str(wav), label))
    if not samples:
        raise FileNotFoundError(f"No wav files found under {root}. Expect data_root/class/*.wav")
    return samples

def build_label_map(samples):
    classes = sorted({lab for _, lab in samples})
    idx = {c:i for i,c in enumerate(classes)}
    return idx, classes

def stratified_80_20(samples, label_to_idx, split_json):
    """按类别分层：每类 80% 训练, 20% 测试。固定化后写入 JSON，复用可重现。"""
    if Path(split_json).exists():
        with open(split_json, "r") as f:
            return json.load(f)

    by_class = defaultdict(list)
    for path, lab in samples:
        by_class[lab].append(path)
    for lab in by_class:
        random.shuffle(by_class[lab])

    train_idx, test_idx = [], []
    for lab, paths in by_class.items():
        n = len(paths)
        n_test = max(1, int(round(n * 0.2)))
        test_paths = paths[:n_test]
        train_paths = paths[n_test:]
        train_idx.extend([(p, lab) for p in train_paths])
        test_idx.extend([(p, lab) for p in test_paths])

    split = {"train": train_idx, "test": test_idx}
    with open(split_json, "w") as f:
        json.dump(split, f, indent=2)
    return split

def seconds_to_samples(sec, sr): return int(round(sec * sr))

# --------------------------
# Datasets & Collate
# --------------------------

class AudioDataset(Dataset):
    """
    通用数据集；根据 mode 不同，把样本转成：
      - 'raw': 1D waveform [1, L]
      - 'melspec': 2D [1, n_mels, T]
      - 'mfcc_stats': 向量特征（传统ML），返回 numpy 1D
    """
    def __init__(self, items, label_to_idx, target_sr=16000, mode="melspec",
                 target_duration=4.0, n_mels=64, n_mfcc=20):
        self.items = items
        self.label_to_idx = label_to_idx
        self.sr = target_sr
        self.mode = mode
        self.target_len = seconds_to_samples(target_duration, target_sr)
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        self.resample = None
        self.mel = None
        self.ampl2db = None
        self.mfcc = None

        # lazy build transforms
        self.ampl2db = T.AmplitudeToDB(stype="power")
        if mode in ("melspec", "cnn2d_melspec"):
            self.mel = T.MelSpectrogram(sample_rate=self.sr, n_fft=1024, hop_length=320, win_length=1024,
                                        f_min=20, f_max=8000, n_mels=self.n_mels, power=2.0)
        if mode in ("mfcc", "mfcc_stats", "classical_mfcc_svm"):
            self.mfcc = T.MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc, melkwargs={"n_fft":1024,"hop_length":320,"n_mels":64})

    def _load(self, path):
        wav, sr = torchaudio.load(path)           # [C, L]
        wav = wav.mean(dim=0, keepdim=True)       # mono
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _pad_or_trim(self, wav):
        L = wav.shape[-1]
        if L == self.target_len:
            return wav
        if L > self.target_len:
            # 中心裁剪（也可随机裁剪用于数据增强）
            start = (L - self.target_len)//2
            return wav[:, start:start+self.target_len]
        # pad
        pad_len = self.target_len - L
        return F.pad(wav, (0, pad_len))  # 右侧补零

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, lab = self.items[idx]
        label = self.label_to_idx[lab]
        wav = self._load(path)  # [1, L]
        if self.mode in ("raw", "cnn1d_raw"):
            wav = self._pad_or_trim(wav)
            # 简单归一化
            wav = (wav - wav.mean()) / (wav.std() + 1e-6)
            return wav, label, path

        if self.mode in ("melspec", "cnn2d_melspec"):
            wav = self._pad_or_trim(wav)
            spec = self.mel(wav)                   # [1, n_mels, T]
            spec = self.ampl2db(spec)
            # 按通道做 instance norm
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
            return spec, label, path

        if self.mode in ("mfcc", "mfcc_stats", "classical_mfcc_svm"):
            # 传统ML：提 MFCC -> 按时间做统计（mean/std/median/max/min）
            mfcc = self.mfcc(wav)                  # [1, n_mfcc, T]
            x = mfcc.squeeze(0).numpy()            # [n_mfcc, T]
            feats = []
            for stat in (np.mean, np.std, np.median, np.max, np.min):
                feats.append(stat(x, axis=1))
            feats = np.concatenate(feats, axis=0)  # [n_mfcc * 5]
            return feats.astype(np.float32), label, path

        raise ValueError(f"Unknown mode: {self.mode}")

def collate_pad_1d(batch):
    xs, ys, paths = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    X = torch.stack(xs, dim=0)  # 已经统一长度
    y = torch.tensor(ys, dtype=torch.long)
    return X, y, lengths, paths

def collate_pad_2d(batch):
    xs, ys, paths = zip(*batch)
    # 统一到当前 batch 的最大 T（我们前面已固定时长，则长度一致）
    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    return X, y, lengths, paths

# --------------------------
# Models
# --------------------------

class CNN1D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 9, stride=2, padding=4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 9, stride=2, padding=4), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):  # x: [B,1,L]
        x = self.net(x)
        return self.head(x)

class CNN2D_Small(nn.Module):
    def __init__(self, n_classes, in_ch=1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):  # x: [B,1,F,T]
        x = self.feat(x)
        return self.head(x)

# --------------------------
# Train / Eval
# --------------------------

def train_one_epoch(model, loader, device, optim, scheduler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y, _, _ in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad(); loss.backward(); optim.step()
        if scheduler: scheduler.step()

        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []
    for X, y, _, _ in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        all_pred.append(pred.cpu().numpy()); all_true.append(y.cpu().numpy())
    acc = correct/total
    return loss_sum/total, acc, np.concatenate(all_true), np.concatenate(all_pred)

def make_loader(dataset, idxs, batch_size, mode):
    if mode in ("cnn1d_raw", "raw"): collate = collate_pad_1d
    elif mode in ("cnn2d_melspec", "melspec"): collate = collate_pad_2d
    else: raise ValueError("Loader only for DL modes")
    subset = Subset(dataset, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate)

# --------------------------
# Cross Validation (DL)
# --------------------------

def cross_validate_dl(train_items, label_to_idx, args, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(train_items)
    y_all = np.array([label_to_idx[lab] for _, lab in train_items], dtype=np.int64)
    idx_all = np.arange(N)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_metrics = []
    best_fold_state = None
    best_fold_acc = -1.0

    # 预构建 Dataset（避免重复IO）
    ds = AudioDataset(train_items, label_to_idx, target_sr=args.sr, mode=args.approach,
                      target_duration=args.duration, n_mels=args.n_mels, n_mfcc=args.n_mfcc)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(idx_all, y_all), 1):
        if args.approach == "cnn1d_raw":
            model = CNN1D(n_classes=len(classes))
        else:
            model = CNN2D_Small(n_classes=len(classes))
        model.to(device)

        train_loader = make_loader(ds, tr_idx, args.batch_size, args.approach)
        val_loader   = make_loader(ds, va_idx, args.batch_size, args.approach)

        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_loader)*args.epochs)

        best_acc, best_state = 0.0, None
        for ep in range(1, args.epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optim, scheduler)
            va_loss, va_acc, y_true, y_pred = eval_epoch(model, val_loader, device)
            print(f"[Fold {fold}][Ep {ep}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        fold_metrics.append(best_acc)
        if best_acc > best_fold_acc:
            best_fold_acc = best_acc
            best_fold_state = best_state

    print(f"CV val acc per fold: {['%.4f'%m for m in fold_metrics]}, mean={np.mean(fold_metrics):.4f}")
    return best_fold_state

# --------------------------
# Classical ML (MFCC + SVM/RF) with CV
# --------------------------

def cross_validate_classical(train_items, label_to_idx, args):
    # 提取 MFCC 统计特征
    ds = AudioDataset(train_items, label_to_idx, target_sr=args.sr, mode="classical_mfcc_svm",
                      target_duration=args.duration, n_mfcc=args.n_mfcc)
    X = []; y = []
    for i in range(len(ds)):
        feats, lab, _ = ds[i]
        X.append(feats); y.append(lab)
    X = np.stack(X, axis=0); y = np.array(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = {
        "svm_rbf": Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=5.0, gamma="scale", kernel="rbf"))]),
        "rf_300":  Pipeline([("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=args.seed))]),
    }

    best_name, best_acc = None, -1.0
    for name, model in candidates.items():
        fold_accs = []
        for tr_idx, va_idx in skf.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_accs.append(accuracy_score(y[va_idx], pred))
        mean_acc = float(np.mean(fold_accs))
        print(f"[{name}] 5-fold val acc: {mean_acc:.4f}")
        if mean_acc > best_acc:
            best_acc, best_name = mean_acc, name
    print(f"Best classical model: {best_name} (val_acc={best_acc:.4f})")
    # 最终在全训练集拟合最佳 classical 模型并返回
    best_model = candidates[best_name]
    best_model.fit(X, y)
    return best_model

# --------------------------
# Final Train on Full Train Split (DL)
# --------------------------

def train_full_dl(train_items, label_to_idx, args, classes, init_state_dict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AudioDataset(train_items, label_to_idx, target_sr=args.sr, mode=args.approach,
                      target_duration=args.duration, n_mels=args.n_mels, n_mfcc=args.n_mfcc)
    idx_all = np.arange(len(ds))
    loader = make_loader(ds, idx_all, args.batch_size, args.approach)

    if args.approach == "cnn1d_raw":
        model = CNN1D(n_classes=len(classes))
    else:
        model = CNN2D_Small(n_classes=len(classes))
    if init_state_dict:
        model.load_state_dict(init_state_dict, strict=False)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(loader)*args.final_epochs)

    best_state, best_acc = None, -1.0
    for ep in range(1, args.final_epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, loader, device, optim, scheduler)
        # 用训练集自身做监控（没有验证集了），只为挑选一个最稳定的checkpoint
        if tr_acc > best_acc:
            best_acc = tr_acc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        print(f"[FullTrain][Ep {ep}] acc={tr_acc:.4f}")

    model.load_state_dict(best_state)
    return model

# --------------------------
# Evaluation on Held-out Test
# --------------------------

def evaluate_on_test(model_or_pipeline, test_items, label_to_idx, args, classes, mode):
    if mode == "classical_mfcc_svm":
        # 提特征
        ds = AudioDataset(test_items, label_to_idx, target_sr=args.sr, mode=mode,
                          target_duration=args.duration, n_mfcc=args.n_mfcc)
        X = []; y = []; paths=[]
        for i in range(len(ds)):
            feats, lab, p = ds[i]
            X.append(feats); y.append(lab); paths.append(p)
        X = np.stack(X, axis=0); y = np.array(y)
        pred = model_or_pipeline.predict(X)
        acc = accuracy_score(y, pred)
        print("\n=== TEST RESULTS (Classical) ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y, pred, target_names=classes, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y, pred))
        return acc

    # DL 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AudioDataset(test_items, label_to_idx, target_sr=args.sr, mode=mode,
                      target_duration=args.duration, n_mels=args.n_mels, n_mfcc=args.n_mfcc)
    idx_all = np.arange(len(ds))
    loader = make_loader(ds, idx_all, args.batch_size, mode)

    model = model_or_pipeline.to(device)
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, loader, device)
    print("\n=== TEST RESULTS (DL) ===")
    print(f"Accuracy: {test_acc:.4f}")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    return test_acc

# --------------------------
# Main
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="data_root/class_name/*.wav")
    parser.add_argument("--approach", type=str, default="cnn2d_melspec",
                        choices=["classical_mfcc_svm","cnn1d_raw","cnn2d_melspec"])
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0, help="pad/trim length (seconds)")
n_mels = 64
n_mfcc = 20
batch_size = 32
epochs = 12  # epochs per fold
final_epochs = 16  # full-train epochs
lr = 3e-4
seed = 42
split_json = "split_80_20.json"
    args = parser.parse_args()

    set_seed(args.seed)
    all_samples = find_wavs(args.data_root)
    label_to_idx, classes = build_label_map(all_samples)
    print("Classes:", classes)

    split = stratified_80_20(all_samples, label_to_idx, args.split_json)
    train_items = split["train"]; test_items = split["test"]
    print(f"Train size: {len(train_items)} | Test size: {len(test_items)}")
    cnt = Counter([lab for _, lab in train_items]); print("Train per class:", cnt)
    cnt = Counter([lab for _, lab in test_items]);  print("Test per class:", cnt)

    if args.approach == "classical_mfcc_svm":
        best_pipeline = cross_validate_classical(train_items, label_to_idx, args)
        evaluate_on_test(best_pipeline, test_items, label_to_idx, args, classes, mode="classical_mfcc_svm")
        return

    # Deep learning approaches
    best_state = cross_validate_dl(train_items, label_to_idx, args, classes)
    final_model = train_full_dl(train_items, label_to_idx, args, classes, init_state_dict=best_state)
    evaluate_on_test(final_model, test_items, label_to_idx, args, classes, mode=args.approach)

if __name__ == "__main__":
    main()
```

### 使用方法

```bash
# 传统ML（MFCC + SVM/随机森林）：
python train_animals.py --data_root /path/to/dataset --approach classical_mfcc_svm

# 端到端原始波形 1D CNN：
python train_animals.py --data_root /path/to/dataset --approach cnn1d_raw --duration 4.0

# Mel 频谱 2D CNN（默认）：
python train_animals.py --data_root /path/to/dataset --approach cnn2d_melspec --n_mels 64 --duration 4.0
```

### 设计要点（直说人话）

* **分层切分**：每类各自 80/20，避免长尾类在测试集被抽空（你这数据有多类少样本）。
* **交叉验证**：仅在训练集上做 5-fold，选择验证集均值最好的设定（DL 保存最佳 fold 的权重快照，classical 挑最好的模型管线）。
* **最终评估**：用“整训练集重训”后的模型，在**从未参与训练/验证**的 20% 测试集上给出准确率、分类报告和混淆矩阵。
* **变长处理**：统一采样率（16kHz），按 `--duration` 秒做 pad/trim。1D CNN 用定长波形；2D CNN 用定长 Mel-Spec；传统ML则对 MFCC 按时间做统计（mean/std/median/max/min）得到**固定长度向量**，不怕变长。
* **三条路线对比**：

  * 传统线（特征向量）：速度快、鲁棒、可解释，样本少时经常打得过浅层DL。
  * 原始波形（序列）：最少先验，但对数据量更敏感。
  * 频谱图像（图片）：和视觉 CNN 套路兼容，常常是音频分类的强 baseline。

### 可加的增强（留坑位，不花里胡哨）

* **数据增强**：随机裁剪、时间掩蔽（SpecAugment）、混合噪声、随机增益。
* **类不平衡**：在损失里加 `class_weight` 或者 `WeightedRandomSampler`。
* **更强骨干**：2D 分支换成 `torchvision.models.resnet18` 接 1 通道（头部改分类数）。
* **早停 & 模型保存**：按验证集指标保存 best.pt。
* **指标**：除了 accuracy，加 macro-F1 更公平。

需要我把 **class weight、SpecAugment、ResNet18** 和 **WeightedRandomSampler** 也直接塞进来？或者给你一键对比报告脚本，把三种方法的 CV/TEST 指标放成一张表。
***

# Report
## GRU not very high
  - Exactly — that’s the delicious irony of “deep” audio models: they often sweat profusely to gain a few percentage points over a well-engineered feature vector.
    And that small delta actually *teaches* something important:
  - 1️⃣ Handcrafted features already encode much of what GRU learns
    Your vector baseline wasn’t primitive — it had mean/std/p10/p90 over MFCCs, chroma, tonnetz, contrast, etc.
    That’s basically a **manually collapsed temporal summary**, so the GRU’s job (modeling sequence evolution) doesn’t add much unless you have:

    * rapid, non-stationary sounds (speech, music), or
    * long-range structure (>3–5 s).

    Animal calls are short and stationary. The “temporal dynamics” the GRU can learn are mostly noise patterns or repetition cadence, so the gain is modest.
  - 2️⃣ You’re data-limited, not capacity-limited
    With a few hundred clips, the GRU is under-fed. You can give it fancier ears (log-mel inputs, small conv front-end, SpecAugment — which you already did), but it won’t magically invent new discriminative cues without more examples.

    If you doubled the dataset or used heavy augmentation (pitch/time-stretch), the GRU would start separating classes better.
  - 3️⃣ Contextual takeaway for your report
    Phrase it like this:

    > “Compared with the traditional vector baseline (84 % accuracy), the sequence-based BiGRU reached 86 %, showing that explicit temporal modeling provides limited additional benefit for short, quasi-stationary animal sounds. This indicates that handcrafted statistical pooling already captures most of the discriminative information for such datasets, while sequence models mainly offer robustness under larger or more dynamic corpora.”

  - 4️⃣ If you’re tempted to squeeze a bit more
    * Use **log-mel sequence** instead of MFCC (MFCC truncates harmonics).
    * **Pretrain** the GRU as an autoencoder or contrastive model before classification.
    * Or just ensemble GRU + YAMNet + CRNN → you’ll see 0.90 + overall accuracy without much extra cost.

    So yes, 0.86 looks small — but it’s also **realistic and instructive**.
    You’ve now empirically shown that the *simple baseline was already very strong*, and that deep temporal models aren’t magic wands without the right data richness.
## Comparison
  Data Formulation Impact: Feature Engineering vs. Raw Learning

  1. Handcrafted Features vs. Raw Waveforms
  Handcrafted features (84.35%) significantly outperformed raw waveform learning (76.52%). With limited data (575 samples), learning from high-dimensional raw input (66,150 dimensions) leads to severe overfitting, while curated features provide robust, generalizable representations.

  2. Optimal Audio Duration and Sampling Rate

  Duration: Shorter clips (3-5 seconds) outperformed longer ones by focusing on core sound events rather than silence or background noise. Sampling rate: 22 kHz captures up to 11 kHz (vs. 8 kHz at 16 kHz), preserving high-frequency components critical for animal sound discrimination.

  3. Transfer Learning Advantage

  YAMNet embeddings (90.43%) substantially improved over handcrafted features (84.35%), demonstrating that pretrained models on broad audio datasets provide superior feature representations.

  4. Temporal Modeling Value

  MFCC sequences with BiGRU (86.09%) slightly outperformed statistical pooling (84.35%), confirming that preserving temporal order captures patterns lost in aggregation.

  5. Image Formulation Superiority

  The image-based CRNN (95.65%) dramatically outperformed all other approaches. Converting audio to 2D Mel-spectrograms enabled leveraging powerful pretrained computer vision models (EfficientNet), which excel at learning spatial hierarchies. Combining this with temporal modeling (GRU) captured frequency, timbral, and temporal information with highest fidelity.
***

# Presentation script
## **(Slide 4: Temporal Modeling with MFCC + Bidirectional GRU)**

“Thank you.
Our traditional models performed well, but we wanted to see if adding **temporal information** could push performance further.
To do that, we treated each audio clip as a **sequence** rather than a static vector, and fed the MFCC frame sequences into a **two-layer Bidirectional GRU with Attention**.

Initially, we **loaded the best model weights after each fold**, and it looked like the model was clearly **overfitting**—training accuracy reached 100 percent while validation lagged behind.
After we switched to **scratch training for each fold**, the curves normalized and the test accuracy became realistic.

Next, with **Global CMVN** for input normalization, a higher **dropout rate**, and the final **two-layer BiGRU**, the model achieved **0.80 macro F1** and **86 percent accuracy**.
You can see on the graph that the **red training and validation lines** now move closely together, showing a much healthier and more stable learning curve.

**Our key insight:** the BiGRU helped a little but taught us a lot.
Handcrafted features already summarize short animal sounds very well.
The model is limited more by **data size** than by **architecture depth**, and temporal modeling only becomes powerful for **longer, more dynamic audio** such as speech or music.”

## **(Slide 5: Feature Transfer Learning: YAMNet Embeddings Boost)**

“Because our dataset was limited, our next logical step was transfer learning.
Our architecture** is shown in this diagram. We used pretrained models purely as feature extractors — YAMNet and Wav2Vec2.

Each audio clip was transformed into a 2048-dimensional embedding using mean-and-standard-deviation pooling, and those embeddings were fed into an SVM classifier.
The results were very clear.
Wav2Vec2, which is trained on human speech, didn’t generalize well and gave about 76 percent accuracy.
But YAMNet, trained on environmental sounds, matched our domain perfectly and achieved over 90 percent accuracy.

The key takeaway: for audio, or more generic transfer learning, domain-matching is critical.”

## **(Slide 6 (Image-Based CRNN))**

“Next, we brought everything together in our final and best model — the Image-Based CRNN.

We converted each audio clip into a Mel-spectrogram, turning the problem into a computer vision task.
Using EfficientNet-B0 as the CNN backbone, we applied modern training techniques — AdamW optimizer, cosine decay, label smoothing, and data augmentation — achieving around 94.8 percent accuracy right out of the box, which is the **blue line**.

However, some minority classes were still under-represented, so we added a weighted loss function—the **orange line**, which slightly improved F1-score.
We also experimented with a heavier augmenter—the **Green Line**—but didn’t help much.

The **red line** is our final **CRNN architecture**, where we added a **GRU and an Attention layer** after the `EfficientNet` backbone. It captured both frequency patterns and temporal context, reaching our best result of 95.65 percent test accuracy.

So, the CRNN is conceptually the most complete — the CNN learns spatial frequency features, and the GRU refines them over time for higher-fidelity recognition.”

## **(Slides 7: Real-World Application: Live Web Deployment)**

“To prove our system works beyond notebooks, we deployed it as a web application.

On the left, you can see our browser interface.
Users can record or upload short .wav clips, and the model instantly displays the predicted animal type with a confidence score and a running history.

On the right is the backend pipeline.
It runs on a FastAPI / Uvicorn server and performs three steps:

1. Convert audio to a Mel-spectrogram.
2. Feed it into our trained CRNN model.
3. Return predictions in real time.

This demonstrates a complete end-to-end workflow — from model training, through optimization, to a functional, deployable AI application.

Thank you. I’ll now pass it to Kaixin for the conclusion.”
***

# Training log
  ```py
  log_text_layer1 = """
  Previous B. 序列模型 MFCC序列 + GRU is done. Seems not good
  Training

  Fold1 Ep1: tr 2.269/0.269 | va 2.009/0.467 mF1 0.352
  Fold1 Ep2: tr 1.817/0.573 | va 1.669/0.543 mF1 0.414
  Fold1 Ep3: tr 1.466/0.663 | va 1.536/0.641 mF1 0.561
  Fold1 Ep4: tr 1.176/0.755 | va 1.368/0.598 mF1 0.526
  Fold1 Ep5: tr 0.942/0.807 | va 1.410/0.641 mF1 0.559
  Fold1 Ep6: tr 0.830/0.861 | va 1.395/0.620 mF1 0.551
  Fold1 Ep7: tr 0.701/0.918 | va 1.331/0.717 mF1 0.632
  Fold1 Ep8: tr 0.605/0.935 | va 1.228/0.717 mF1 0.643
  Fold1 Ep9: tr 0.532/0.959 | va 1.221/0.707 mF1 0.624
  Fold1 Ep10: tr 0.498/0.970 | va 1.219/0.707 mF1 0.637
  Fold1 Ep11: tr 0.535/0.970 | va 1.433/0.641 mF1 0.583
  Fold1 Ep12: tr 0.531/0.965 | va 1.334/0.696 mF1 0.639
  Fold1 Ep13: tr 0.475/0.984 | va 1.292/0.674 mF1 0.587
  Fold1 Ep14: tr 0.441/0.995 | va 1.360/0.674 mF1 0.593
  Fold1 Ep15: tr 0.423/1.000 | va 1.360/0.674 mF1 0.589
  Fold1 Ep16: tr 0.422/0.997 | va 1.352/0.674 mF1 0.589
  Fold1 Ep17: tr 0.420/1.000 | va 1.344/0.674 mF1 0.589
  Fold1 Ep18: tr 0.415/1.000 | va 1.337/0.674 mF1 0.590
  Fold1 Ep19: tr 0.413/1.000 | va 1.336/0.674 mF1 0.590
  Fold2 Ep1: tr 2.275/0.220 | va 2.062/0.489 mF1 0.415
  Fold2 Ep2: tr 1.839/0.527 | va 1.732/0.576 mF1 0.462
  Fold2 Ep3: tr 1.474/0.625 | va 1.561/0.598 mF1 0.496
  Fold2 Ep4: tr 1.216/0.677 | va 1.473/0.674 mF1 0.606
  Fold2 Ep5: tr 1.005/0.758 | va 1.430/0.685 mF1 0.655
  Fold2 Ep6: tr 0.912/0.785 | va 1.513/0.533 mF1 0.498
  Fold2 Ep7: tr 0.817/0.840 | va 1.455/0.652 mF1 0.604
  Fold2 Ep8: tr 0.736/0.851 | va 1.416/0.674 mF1 0.627
  Fold2 Ep9: tr 0.666/0.886 | va 1.339/0.696 mF1 0.674
  Fold2 Ep10: tr 0.576/0.951 | va 1.311/0.663 mF1 0.601
  Fold2 Ep11: tr 0.517/0.962 | va 1.365/0.707 mF1 0.678
  Fold2 Ep12: tr 0.494/0.970 | va 1.299/0.707 mF1 0.666
  Fold2 Ep13: tr 0.463/0.986 | va 1.315/0.696 mF1 0.659
  Fold2 Ep14: tr 0.434/0.997 | va 1.326/0.685 mF1 0.648
  Fold2 Ep15: tr 0.418/1.000 | va 1.323/0.685 mF1 0.653
  Fold2 Ep16: tr 0.413/1.000 | va 1.309/0.707 mF1 0.672
  Fold2 Ep17: tr 0.405/1.000 | va 1.310/0.707 mF1 0.672
  Fold2 Ep18: tr 0.403/1.000 | va 1.313/0.707 mF1 0.672
  Fold2 Ep19: tr 0.402/1.000 | va 1.316/0.707 mF1 0.672
  Fold2 Ep20: tr 0.402/1.000 | va 1.319/0.707 mF1 0.673
  Fold2 Ep21: tr 0.403/1.000 | va 1.319/0.707 mF1 0.673
  Fold3 Ep1: tr 2.264/0.299 | va 2.065/0.489 mF1 0.395
  Fold3 Ep2: tr 1.904/0.511 | va 1.776/0.543 mF1 0.485
  Fold3 Ep3: tr 1.567/0.666 | va 1.498/0.576 mF1 0.504
  Fold3 Ep4: tr 1.289/0.690 | va 1.387/0.533 mF1 0.493
  Fold3 Ep5: tr 1.072/0.755 | va 1.302/0.609 mF1 0.542
  Fold3 Ep6: tr 0.899/0.810 | va 1.172/0.685 mF1 0.641
  Fold3 Ep7: tr 0.836/0.834 | va 1.210/0.641 mF1 0.576
  Fold3 Ep8: tr 0.726/0.880 | va 1.128/0.663 mF1 0.640
  Fold3 Ep9: tr 0.632/0.910 | va 1.181/0.696 mF1 0.660
  Fold3 Ep10: tr 0.569/0.946 | va 1.149/0.696 mF1 0.649
  Fold3 Ep11: tr 0.517/0.959 | va 1.166/0.652 mF1 0.618
  Fold3 Ep12: tr 0.485/0.984 | va 1.245/0.739 mF1 0.683
  Fold3 Ep13: tr 0.458/0.989 | va 1.228/0.728 mF1 0.677
  Fold3 Ep14: tr 0.451/0.986 | va 1.211/0.728 mF1 0.671
  Fold3 Ep15: tr 0.443/0.995 | va 1.204/0.717 mF1 0.650
  Fold3 Ep16: tr 0.443/0.997 | va 1.201/0.717 mF1 0.650
  Fold3 Ep17: tr 0.444/0.995 | va 1.201/0.717 mF1 0.650
  Fold4 Ep1: tr 2.269/0.215 | va 2.039/0.457 mF1 0.365
  Fold4 Ep2: tr 1.830/0.576 | va 1.669/0.641 mF1 0.546
  Fold4 Ep3: tr 1.427/0.698 | va 1.461/0.630 mF1 0.572
  Fold4 Ep4: tr 1.192/0.750 | va 1.337/0.685 mF1 0.627
  Fold4 Ep5: tr 1.103/0.785 | va 1.304/0.674 mF1 0.598
  Fold4 Ep6: tr 0.922/0.815 | va 1.209/0.674 mF1 0.630
  Fold4 Ep7: tr 0.810/0.845 | va 1.301/0.717 mF1 0.663
  Fold4 Ep8: tr 0.712/0.897 | va 1.204/0.685 mF1 0.627
  Fold4 Ep9: tr 0.632/0.916 | va 1.217/0.717 mF1 0.674
  Fold4 Ep10: tr 0.564/0.948 | va 1.242/0.739 mF1 0.687
  Fold4 Ep11: tr 0.495/0.970 | va 1.224/0.739 mF1 0.664
  Fold4 Ep12: tr 0.459/0.995 | va 1.208/0.739 mF1 0.683
  Fold4 Ep13: tr 0.437/1.000 | va 1.206/0.739 mF1 0.683
  Fold4 Ep14: tr 0.437/1.000 | va 1.202/0.728 mF1 0.671
  Fold4 Ep15: tr 0.435/1.000 | va 1.201/0.728 mF1 0.671
  Fold4 Ep16: tr 0.429/1.000 | va 1.201/0.728 mF1 0.671
  Fold4 Ep17: tr 0.424/1.000 | va 1.201/0.728 mF1 0.671
  Fold4 Ep18: tr 0.424/1.000 | va 1.198/0.739 mF1 0.686
  Fold4 Ep19: tr 0.421/1.000 | va 1.197/0.739 mF1 0.686
  Fold4 Ep20: tr 0.422/1.000 | va 1.197/0.739 mF1 0.686
  Fold4 Ep21: tr 0.421/1.000 | va 1.194/0.728 mF1 0.666
  Fold4 Ep22: tr 0.419/1.000 | va 1.194/0.717 mF1 0.643
  Fold4 Ep23: tr 0.419/1.000 | va 1.192/0.717 mF1 0.643
  Fold4 Ep24: tr 0.417/1.000 | va 1.190/0.717 mF1 0.643
  Fold4 Ep25: tr 0.416/1.000 | va 1.191/0.728 mF1 0.656
  Fold4 Ep26: tr 0.413/1.000 | va 1.190/0.728 mF1 0.656
  Fold4 Ep27: tr 0.414/1.000 | va 1.189/0.717 mF1 0.643
  Fold4 Ep28: tr 0.413/1.000 | va 1.185/0.728 mF1 0.656
  Fold4 Ep29: tr 0.410/1.000 | va 1.185/0.728 mF1 0.656
  Fold4 Ep30: tr 0.412/1.000 | va 1.183/0.728 mF1 0.656
  Fold5 Ep1: tr 2.219/0.234 | va 2.056/0.315 mF1 0.246
  Fold5 Ep2: tr 1.789/0.489 | va 1.772/0.478 mF1 0.359
  Fold5 Ep3: tr 1.366/0.693 | va 1.795/0.609 mF1 0.558
  Fold5 Ep4: tr 1.078/0.785 | va 1.490/0.609 mF1 0.552
  Fold5 Ep5: tr 0.889/0.823 | va 1.764/0.641 mF1 0.562
  Fold5 Ep6: tr 0.882/0.842 | va 1.365/0.674 mF1 0.636
  Fold5 Ep7: tr 0.718/0.891 | va 1.430/0.685 mF1 0.649
  Fold5 Ep8: tr 0.650/0.921 | va 1.297/0.739 mF1 0.707
  Fold5 Ep9: tr 0.596/0.921 | va 1.366/0.739 mF1 0.710
  Fold5 Ep10: tr 0.506/0.976 | va 1.416/0.739 mF1 0.699
  Fold5 Ep11: tr 0.476/0.984 | va 1.373/0.750 mF1 0.704
  Fold5 Ep12: tr 0.457/0.984 | va 1.443/0.728 mF1 0.678
  Fold5 Ep13: tr 0.467/0.989 | va 1.447/0.707 mF1 0.661
  Fold5 Ep14: tr 0.444/0.997 | va 1.410/0.739 mF1 0.695
  Fold5 Ep15: tr 0.433/1.000 | va 1.401/0.739 mF1 0.696
  Fold5 Ep16: tr 0.431/1.000 | va 1.394/0.739 mF1 0.696
  Fold5 Ep17: tr 0.429/1.000 | va 1.394/0.739 mF1 0.696

  Testing

  py
  Holdout Acc: 0.7739130434782608
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.78      0.70      0.74        20
           Cow       0.92      0.73      0.81        15
           Dog       1.00      0.75      0.86        20
        Donkey       0.83      1.00      0.91         5
          Frog       0.44      0.57      0.50         7
          Lion       0.54      0.78      0.64         9
        Maymun       0.25      0.20      0.22         5
         Sheep       0.75      0.75      0.75         8
         Tavuk       0.60      1.00      0.75         6

      accuracy                           0.77       115
     macro avg       0.71      0.75      0.72       115
  weighted avg       0.80      0.77      0.78       115
  """
  log_text_layer2 = """
  Fold1 Ep1: tr 1.961/0.215 | va 1.939/0.413 mF1 0.371
  Fold1 Ep2: tr 1.478/0.492 | va 1.286/0.696 mF1 0.681
  Fold1 Ep3: tr 1.073/0.685 | va 1.413/0.620 mF1 0.637
  Fold1 Ep4: tr 0.972/0.690 | va 1.207/0.707 mF1 0.631
  Fold1 Ep5: tr 0.770/0.799 | va 0.971/0.772 mF1 0.754
  Fold1 Ep6: tr 0.694/0.832 | va 0.926/0.750 mF1 0.731
  Fold1 Ep7: tr 0.672/0.802 | va 0.940/0.848 mF1 0.820
  Fold1 Ep8: tr 0.606/0.832 | va 0.944/0.750 mF1 0.719
  Fold1 Ep9: tr 0.569/0.870 | va 0.780/0.848 mF1 0.843
  Fold1 Ep10: tr 0.561/0.886 | va 0.853/0.870 mF1 0.850
  Fold1 Ep11: tr 0.469/0.918 | va 0.775/0.826 mF1 0.802
  Fold1 Ep12: tr 0.475/0.921 | va 0.761/0.891 mF1 0.895
  Fold1 Ep13: tr 0.481/0.924 | va 0.726/0.870 mF1 0.866
  Fold1 Ep14: tr 0.416/0.940 | va 0.702/0.880 mF1 0.876
  Fold1 Ep15: tr 0.418/0.924 | va 0.718/0.859 mF1 0.857
  Fold1 Ep16: tr 0.499/0.897 | va 0.746/0.859 mF1 0.850
  Fold1 Ep17: tr 0.412/0.940 | va 0.737/0.880 mF1 0.883
  Fold1 Ep18: tr 0.418/0.954 | va 0.715/0.870 mF1 0.863
  Fold1 Ep19: tr 0.427/0.948 | va 0.714/0.891 mF1 0.891
  Fold1 Ep20: tr 0.381/0.967 | va 0.711/0.902 mF1 0.909
  Fold1 Ep21: tr 0.405/0.957 | va 0.705/0.891 mF1 0.891
  Fold1 Ep22: tr 0.354/0.959 | va 0.708/0.891 mF1 0.891
  Fold1 Ep23: tr 0.389/0.967 | va 0.705/0.891 mF1 0.891
  Fold1 Ep24: tr 0.443/0.921 | va 0.702/0.913 mF1 0.912
  Fold2 Ep1: tr 1.949/0.226 | va 1.998/0.152 mF1 0.165
  Fold2 Ep2: tr 1.295/0.503 | va 1.636/0.478 mF1 0.451
  Fold2 Ep3: tr 1.001/0.630 | va 1.283/0.750 mF1 0.720
  Fold2 Ep4: tr 0.767/0.766 | va 1.214/0.674 mF1 0.646
  Fold2 Ep5: tr 0.630/0.826 | va 1.079/0.761 mF1 0.737
  Fold2 Ep6: tr 0.788/0.777 | va 1.414/0.630 mF1 0.603
  Fold2 Ep7: tr 0.588/0.807 | va 1.276/0.728 mF1 0.685
  Fold2 Ep8: tr 0.610/0.834 | va 1.102/0.793 mF1 0.743
  Fold2 Ep9: tr 0.519/0.880 | va 1.034/0.815 mF1 0.782
  Fold2 Ep10: tr 0.532/0.851 | va 1.130/0.783 mF1 0.761
  Fold2 Ep11: tr 0.540/0.880 | va 0.990/0.880 mF1 0.851
  Fold2 Ep12: tr 0.519/0.875 | va 0.956/0.848 mF1 0.832
  Fold2 Ep13: tr 0.421/0.924 | va 1.000/0.739 mF1 0.717
  Fold2 Ep14: tr 0.414/0.946 | va 1.004/0.826 mF1 0.799
  Fold2 Ep15: tr 0.435/0.918 | va 0.952/0.837 mF1 0.812
  Fold2 Ep16: tr 0.464/0.902 | va 0.949/0.848 mF1 0.827
  Fold2 Ep17: tr 0.477/0.940 | va 0.903/0.848 mF1 0.825
  Fold2 Ep18: tr 0.395/0.938 | va 0.897/0.848 mF1 0.821
  Fold2 Ep19: tr 0.430/0.946 | va 0.905/0.848 mF1 0.825
  Fold2 Ep20: tr 0.419/0.940 | va 0.889/0.859 mF1 0.834
  Fold2 Ep21: tr 0.411/0.940 | va 0.907/0.870 mF1 0.845
  Fold2 Ep22: tr 0.382/0.962 | va 0.888/0.848 mF1 0.825
  Fold2 Ep23: tr 0.439/0.940 | va 0.896/0.870 mF1 0.845
  Fold2 Ep24: tr 0.421/0.954 | va 0.893/0.870 mF1 0.845
  Fold2 Ep25: tr 0.383/0.940 | va 0.894/0.870 mF1 0.845
  Fold2 Ep26: tr 0.407/0.946 | va 0.895/0.859 mF1 0.832
  Fold2 Ep27: tr 0.401/0.946 | va 0.915/0.880 mF1 0.859
  Fold2 Ep28: tr 0.355/0.973 | va 0.928/0.880 mF1 0.860
  Fold2 Ep29: tr 0.401/0.943 | va 0.937/0.859 mF1 0.827
  Fold2 Ep30: tr 0.383/0.948 | va 0.863/0.848 mF1 0.814
  Fold3 Ep1: tr 2.059/0.163 | va 2.180/0.087 mF1 0.044
  Fold3 Ep2: tr 1.642/0.410 | va 1.618/0.413 mF1 0.395
  Fold3 Ep3: tr 1.223/0.554 | va 1.558/0.478 mF1 0.433
  Fold3 Ep4: tr 0.976/0.663 | va 1.343/0.500 mF1 0.511
  Fold3 Ep5: tr 0.866/0.755 | va 1.242/0.620 mF1 0.589
  Fold3 Ep6: tr 0.818/0.766 | va 1.207/0.685 mF1 0.625
  Fold3 Ep7: tr 0.688/0.851 | va 1.142/0.826 mF1 0.796
  Fold3 Ep8: tr 0.635/0.842 | va 1.165/0.707 mF1 0.670
  Fold3 Ep9: tr 0.574/0.864 | va 1.083/0.793 mF1 0.763
  Fold3 Ep10: tr 0.449/0.940 | va 1.058/0.761 mF1 0.747
  Fold3 Ep11: tr 0.448/0.932 | va 1.046/0.793 mF1 0.765
  Fold3 Ep12: tr 0.481/0.916 | va 1.091/0.804 mF1 0.788
  Fold3 Ep13: tr 0.421/0.957 | va 1.023/0.783 mF1 0.778
  Fold3 Ep14: tr 0.461/0.916 | va 1.024/0.793 mF1 0.770
  Fold3 Ep15: tr 0.415/0.962 | va 1.035/0.783 mF1 0.764
  Fold3 Ep16: tr 0.479/0.932 | va 0.999/0.793 mF1 0.768
  Fold3 Ep17: tr 0.395/0.970 | va 0.993/0.783 mF1 0.769
  Fold3 Ep18: tr 0.483/0.910 | va 0.984/0.783 mF1 0.769
  Fold3 Ep19: tr 0.411/0.943 | va 0.985/0.793 mF1 0.772
  Fold3 Ep20: tr 0.380/0.978 | va 0.990/0.793 mF1 0.776
  Fold3 Ep21: tr 0.370/0.965 | va 0.986/0.783 mF1 0.769
  Fold3 Ep22: tr 0.395/0.957 | va 0.982/0.783 mF1 0.769
  Fold3 Ep23: tr 0.392/0.967 | va 0.974/0.793 mF1 0.776
  Fold3 Ep24: tr 0.412/0.954 | va 0.980/0.804 mF1 0.778
  Fold3 Ep25: tr 0.409/0.957 | va 0.978/0.793 mF1 0.765
  Fold3 Ep26: tr 0.458/0.924 | va 0.999/0.793 mF1 0.765
  Fold3 Ep27: tr 0.374/0.957 | va 0.993/0.793 mF1 0.755
  Fold3 Ep28: tr 0.392/0.962 | va 0.953/0.815 mF1 0.787
  Fold3 Ep29: tr 0.426/0.954 | va 0.977/0.815 mF1 0.768
  Fold3 Ep30: tr 0.418/0.957 | va 0.984/0.804 mF1 0.765
  Fold4 Ep1: tr 2.059/0.190 | va 2.057/0.141 mF1 0.108
  Fold4 Ep2: tr 1.358/0.543 | va 1.380/0.543 mF1 0.492
  Fold4 Ep3: tr 0.978/0.698 | va 1.177/0.641 mF1 0.600
  Fold4 Ep4: tr 0.783/0.761 | va 1.019/0.750 mF1 0.700
  Fold4 Ep5: tr 0.688/0.812 | va 1.256/0.663 mF1 0.615
  Fold4 Ep6: tr 0.634/0.821 | va 0.902/0.793 mF1 0.754
  Fold4 Ep7: tr 0.595/0.848 | va 0.925/0.804 mF1 0.789
  Fold4 Ep8: tr 0.615/0.851 | va 0.888/0.815 mF1 0.800
  Fold4 Ep9: tr 0.478/0.913 | va 0.865/0.826 mF1 0.807
  Fold4 Ep10: tr 0.522/0.908 | va 0.892/0.848 mF1 0.822
  Fold4 Ep11: tr 0.429/0.927 | va 0.853/0.837 mF1 0.809
  Fold4 Ep12: tr 0.510/0.894 | va 0.886/0.804 mF1 0.793
  Fold4 Ep13: tr 0.417/0.943 | va 0.825/0.837 mF1 0.799
  Fold4 Ep14: tr 0.451/0.908 | va 0.904/0.815 mF1 0.802
  Fold4 Ep15: tr 0.442/0.916 | va 0.866/0.815 mF1 0.774
  Fold4 Ep16: tr 0.392/0.951 | va 0.828/0.815 mF1 0.782
  Fold4 Ep17: tr 0.388/0.948 | va 0.826/0.837 mF1 0.804
  Fold4 Ep18: tr 0.397/0.946 | va 0.792/0.848 mF1 0.824
  Fold4 Ep19: tr 0.390/0.954 | va 0.819/0.837 mF1 0.812
  Fold4 Ep20: tr 0.408/0.957 | va 0.804/0.837 mF1 0.806
  Fold4 Ep21: tr 0.369/0.954 | va 0.808/0.837 mF1 0.802
  Fold4 Ep22: tr 0.365/0.962 | va 0.803/0.837 mF1 0.808
  Fold4 Ep23: tr 0.399/0.954 | va 0.810/0.826 mF1 0.795
  Fold4 Ep24: tr 0.470/0.913 | va 0.802/0.826 mF1 0.801
  Fold4 Ep25: tr 0.361/0.973 | va 0.810/0.859 mF1 0.835
  Fold4 Ep26: tr 0.374/0.962 | va 0.805/0.848 mF1 0.821
  Fold4 Ep27: tr 0.420/0.962 | va 0.804/0.870 mF1 0.848
  Fold4 Ep28: tr 0.386/0.965 | va 0.836/0.870 mF1 0.832
  Fold5 Ep1: tr 1.988/0.193 | va 2.001/0.239 mF1 0.208
  Fold5 Ep2: tr 1.411/0.538 | va 1.650/0.446 mF1 0.410
  Fold5 Ep3: tr 1.017/0.620 | va 1.421/0.630 mF1 0.601
  Fold5 Ep4: tr 0.945/0.701 | va 1.587/0.598 mF1 0.542
  Fold5 Ep5: tr 0.801/0.766 | va 1.360/0.663 mF1 0.655
  Fold5 Ep6: tr 0.790/0.772 | va 1.361/0.717 mF1 0.674
  Fold5 Ep7: tr 0.635/0.840 | va 1.130/0.804 mF1 0.770
  Fold5 Ep8: tr 0.590/0.837 | va 1.408/0.728 mF1 0.635
  Fold5 Ep9: tr 0.488/0.905 | va 1.139/0.783 mF1 0.738
  Fold5 Ep10: tr 0.460/0.913 | va 1.141/0.837 mF1 0.782
  Fold5 Ep11: tr 0.546/0.908 | va 1.227/0.783 mF1 0.725
  Fold5 Ep12: tr 0.435/0.918 | va 1.217/0.804 mF1 0.743
  Fold5 Ep13: tr 0.440/0.935 | va 1.197/0.826 mF1 0.771
  Fold5 Ep14: tr 0.367/0.967 | va 1.188/0.826 mF1 0.766
  Fold5 Ep15: tr 0.375/0.962 | va 1.170/0.815 mF1 0.764
  Fold5 Ep16: tr 0.402/0.948 | va 1.095/0.804 mF1 0.749
  Fold5 Ep17: tr 0.384/0.959 | va 1.071/0.837 mF1 0.777
  Fold5 Ep18: tr 0.447/0.921 | va 1.084/0.826 mF1 0.766
  Fold5 Ep19: tr 0.389/0.965 | va 1.086/0.826 mF1 0.765
  Fold5 Ep20: tr 0.375/0.965 | va 1.073/0.826 mF1 0.766
  Fold5 Ep21: tr 0.348/0.981 | va 1.096/0.826 mF1 0.767
  Fold5 Ep22: tr 0.389/0.954 | va 1.092/0.826 mF1 0.765
  Fold5 Ep23: tr 0.416/0.946 | va 1.066/0.826 mF1 0.767
  Fold5 Ep24: tr 0.381/0.973 | va 1.069/0.826 mF1 0.770
  Fold5 Ep25: tr 0.385/0.965 | va 1.082/0.826 mF1 0.766
  Fold5 Ep26: tr 0.481/0.924 | va 1.090/0.848 mF1 0.816
  Fold5 Ep27: tr 0.373/0.957 | va 1.123/0.837 mF1 0.777
  Fold5 Ep28: tr 0.417/0.957 | va 1.160/0.826 mF1 0.800
  Fold5 Ep29: tr 0.398/0.951 | va 1.094/0.837 mF1 0.792
  Fold5 Ep30: tr 0.405/0.943 | va 1.080/0.837 mF1 0.777
  Testing output:
  py
  ===== Evaluating GRU-Seq ensemble on the hold-out set =====

  Final Test Set Accuracy (GRU-Seq Ensemble): 0.8609

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.84      0.80      0.82        20
           Cow       0.86      0.80      0.83        15
           Dog       0.90      0.95      0.93        20
        Donkey       1.00      1.00      1.00         5
          Frog       0.83      0.71      0.77         7
          Lion       0.90      1.00      0.95         9
        Maymun       0.00      0.00      0.00         5
         Sheep       0.70      0.88      0.78         8
         Tavuk       0.86      1.00      0.92         6

      accuracy                           0.86       115
     macro avg       0.79      0.81      0.80       115
  weighted avg       0.85      0.86      0.85       115
  """
  log_text_layer1_h128 = """
  Fold1 Ep1: tr 2.048/0.193 | va 2.154/0.098 mF1 0.114
  Fold1 Ep2: tr 1.676/0.405 | va 1.693/0.435 mF1 0.389
  Fold1 Ep3: tr 1.305/0.587 | va 1.435/0.511 mF1 0.487
  Fold1 Ep4: tr 1.018/0.658 | va 1.228/0.620 mF1 0.550
  Fold1 Ep5: tr 0.754/0.772 | va 1.143/0.707 mF1 0.649
  Fold1 Ep6: tr 0.737/0.802 | va 1.211/0.641 mF1 0.601
  Fold1 Ep7: tr 0.610/0.840 | va 1.083/0.761 mF1 0.723
  Fold1 Ep8: tr 0.544/0.878 | va 1.027/0.761 mF1 0.700
  Fold1 Ep9: tr 0.576/0.853 | va 0.959/0.793 mF1 0.750
  Fold1 Ep10: tr 0.492/0.910 | va 0.991/0.772 mF1 0.726
  Fold1 Ep11: tr 0.452/0.927 | va 1.007/0.783 mF1 0.751
  Fold1 Ep12: tr 0.469/0.921 | va 1.024/0.804 mF1 0.765
  Fold1 Ep13: tr 0.475/0.913 | va 0.982/0.826 mF1 0.780
  Fold1 Ep14: tr 0.513/0.910 | va 0.930/0.826 mF1 0.779
  Fold1 Ep15: tr 0.470/0.935 | va 0.909/0.826 mF1 0.797
  Fold1 Ep16: tr 0.517/0.899 | va 0.904/0.783 mF1 0.752
  Fold1 Ep17: tr 0.462/0.927 | va 0.941/0.783 mF1 0.738
  Fold1 Ep18: tr 0.430/0.929 | va 0.967/0.804 mF1 0.757
  Fold1 Ep19: tr 0.449/0.943 | va 0.944/0.793 mF1 0.748
  Fold1 Ep20: tr 0.464/0.935 | va 0.943/0.793 mF1 0.748
  Fold1 Ep21: tr 0.397/0.948 | va 0.941/0.783 mF1 0.738
  Fold1 Ep22: tr 0.449/0.918 | va 0.943/0.793 mF1 0.748
  Fold1 Ep23: tr 0.405/0.954 | va 0.916/0.793 mF1 0.748
  Fold1 Ep24: tr 0.417/0.959 | va 0.938/0.772 mF1 0.724
  Fold1 Ep25: tr 0.469/0.929 | va 0.940/0.793 mF1 0.751
  Fold1 Ep26: tr 0.443/0.957 | va 0.928/0.783 mF1 0.737
  Fold2 Ep1: tr 2.101/0.163 | va 2.215/0.098 mF1 0.081
  Fold2 Ep2: tr 1.717/0.359 | va 1.904/0.130 mF1 0.166
  Fold2 Ep3: tr 1.315/0.495 | va 1.513/0.478 mF1 0.437
  Fold2 Ep4: tr 1.066/0.668 | va 1.122/0.620 mF1 0.637
  Fold2 Ep5: tr 0.788/0.734 | va 0.944/0.717 mF1 0.713
  Fold2 Ep6: tr 0.756/0.780 | va 0.878/0.783 mF1 0.749
  Fold2 Ep7: tr 0.670/0.812 | va 0.811/0.815 mF1 0.816
  Fold2 Ep8: tr 0.584/0.840 | va 0.791/0.859 mF1 0.852
  Fold2 Ep9: tr 0.584/0.842 | va 0.786/0.783 mF1 0.765
  Fold2 Ep10: tr 0.538/0.889 | va 0.779/0.815 mF1 0.799
  Fold2 Ep11: tr 0.540/0.894 | va 0.773/0.848 mF1 0.849
  Fold2 Ep12: tr 0.487/0.913 | va 0.842/0.804 mF1 0.800
  Fold2 Ep13: tr 0.457/0.897 | va 0.741/0.837 mF1 0.835
  Fold2 Ep14: tr 0.498/0.908 | va 0.712/0.859 mF1 0.865
  Fold2 Ep15: tr 0.437/0.918 | va 0.734/0.848 mF1 0.848
  Fold2 Ep16: tr 0.434/0.938 | va 0.717/0.859 mF1 0.858
  Fold2 Ep17: tr 0.473/0.918 | va 0.705/0.880 mF1 0.884
  Fold2 Ep18: tr 0.442/0.910 | va 0.698/0.880 mF1 0.884
  Fold2 Ep19: tr 0.446/0.913 | va 0.708/0.859 mF1 0.853
  Fold2 Ep20: tr 0.415/0.951 | va 0.705/0.880 mF1 0.884
  Fold2 Ep21: tr 0.427/0.924 | va 0.706/0.880 mF1 0.884
  Fold2 Ep22: tr 0.480/0.913 | va 0.707/0.880 mF1 0.884
  Fold2 Ep23: tr 0.423/0.921 | va 0.703/0.880 mF1 0.884
  Fold2 Ep24: tr 0.495/0.902 | va 0.692/0.891 mF1 0.896
  Fold2 Ep25: tr 0.417/0.932 | va 0.710/0.870 mF1 0.871
  Fold2 Ep26: tr 0.435/0.929 | va 0.725/0.870 mF1 0.868
  Fold2 Ep27: tr 0.447/0.938 | va 0.720/0.880 mF1 0.884
  Fold2 Ep28: tr 0.403/0.957 | va 0.749/0.859 mF1 0.846
  Fold2 Ep29: tr 0.435/0.932 | va 0.753/0.848 mF1 0.843
  Fold2 Ep30: tr 0.421/0.962 | va 0.696/0.859 mF1 0.846
  Fold3 Ep1: tr 2.105/0.168 | va 2.194/0.054 mF1 0.023
  Fold3 Ep2: tr 1.762/0.340 | va 1.848/0.457 mF1 0.374
  Fold3 Ep3: tr 1.384/0.552 | va 1.660/0.489 mF1 0.420
  Fold3 Ep4: tr 0.993/0.701 | va 1.626/0.576 mF1 0.504
  Fold3 Ep5: tr 0.839/0.761 | va 1.511/0.652 mF1 0.615
  Fold3 Ep6: tr 0.678/0.815 | va 1.498/0.674 mF1 0.636
  Fold3 Ep7: tr 0.730/0.788 | va 1.536/0.674 mF1 0.632
  Fold3 Ep8: tr 0.599/0.842 | va 1.414/0.696 mF1 0.624
  Fold3 Ep9: tr 0.596/0.832 | va 1.431/0.728 mF1 0.684
  Fold3 Ep10: tr 0.605/0.853 | va 1.309/0.750 mF1 0.687
  Fold3 Ep11: tr 0.520/0.894 | va 1.290/0.739 mF1 0.686
  Fold3 Ep12: tr 0.471/0.921 | va 1.283/0.750 mF1 0.688
  Fold3 Ep13: tr 0.504/0.878 | va 1.252/0.750 mF1 0.706
  Fold3 Ep14: tr 0.459/0.913 | va 1.223/0.750 mF1 0.689
  Fold3 Ep15: tr 0.495/0.905 | va 1.235/0.728 mF1 0.665
  Fold3 Ep16: tr 0.461/0.902 | va 1.231/0.750 mF1 0.700
  Fold3 Ep17: tr 0.439/0.938 | va 1.236/0.761 mF1 0.714
  Fold3 Ep18: tr 0.458/0.905 | va 1.229/0.761 mF1 0.716
  Fold3 Ep19: tr 0.470/0.916 | va 1.217/0.761 mF1 0.718
  Fold3 Ep20: tr 0.421/0.951 | va 1.223/0.761 mF1 0.717
  Fold3 Ep21: tr 0.390/0.965 | va 1.212/0.761 mF1 0.716
  Fold3 Ep22: tr 0.407/0.948 | va 1.221/0.761 mF1 0.716
  Fold3 Ep23: tr 0.441/0.929 | va 1.236/0.761 mF1 0.718
  Fold3 Ep24: tr 0.389/0.951 | va 1.228/0.761 mF1 0.717
  Fold3 Ep25: tr 0.408/0.946 | va 1.221/0.761 mF1 0.701
  Fold3 Ep26: tr 0.432/0.921 | va 1.226/0.761 mF1 0.707
  Fold3 Ep27: tr 0.443/0.910 | va 1.208/0.772 mF1 0.723
  Fold3 Ep28: tr 0.401/0.946 | va 1.271/0.728 mF1 0.689
  Fold3 Ep29: tr 0.442/0.921 | va 1.298/0.772 mF1 0.725
  Fold3 Ep30: tr 0.409/0.935 | va 1.288/0.750 mF1 0.700
  Fold4 Ep1: tr 2.066/0.147 | va 2.278/0.130 mF1 0.097
  Fold4 Ep2: tr 1.698/0.348 | va 1.791/0.402 mF1 0.404
  Fold4 Ep3: tr 1.373/0.486 | va 1.513/0.413 mF1 0.418
  Fold4 Ep4: tr 0.976/0.663 | va 1.227/0.598 mF1 0.571
  Fold4 Ep5: tr 0.918/0.715 | va 1.182/0.685 mF1 0.626
  Fold4 Ep6: tr 0.794/0.745 | va 1.073/0.750 mF1 0.722
  Fold4 Ep7: tr 0.706/0.823 | va 1.018/0.761 mF1 0.730
  Fold4 Ep8: tr 0.627/0.837 | va 0.951/0.772 mF1 0.739
  Fold4 Ep9: tr 0.544/0.891 | va 0.952/0.804 mF1 0.769
  Fold4 Ep10: tr 0.481/0.929 | va 1.035/0.728 mF1 0.695
  Fold4 Ep11: tr 0.629/0.837 | va 1.081/0.739 mF1 0.711
  Fold4 Ep12: tr 0.502/0.891 | va 1.015/0.772 mF1 0.730
  Fold4 Ep13: tr 0.484/0.910 | va 0.989/0.804 mF1 0.777
  Fold4 Ep14: tr 0.447/0.929 | va 0.979/0.783 mF1 0.753
  Fold4 Ep15: tr 0.458/0.916 | va 0.955/0.783 mF1 0.753
  Fold4 Ep16: tr 0.471/0.889 | va 0.946/0.804 mF1 0.774
  Fold4 Ep17: tr 0.455/0.918 | va 0.926/0.804 mF1 0.773
  Fold4 Ep18: tr 0.490/0.902 | va 0.935/0.815 mF1 0.787
  Fold4 Ep19: tr 0.467/0.924 | va 0.943/0.804 mF1 0.774
  Fold4 Ep20: tr 0.417/0.943 | va 0.924/0.804 mF1 0.774
  Fold4 Ep21: tr 0.426/0.946 | va 0.931/0.815 mF1 0.785
  Fold4 Ep22: tr 0.446/0.929 | va 0.928/0.815 mF1 0.787
  Fold4 Ep23: tr 0.436/0.946 | va 0.938/0.804 mF1 0.774
  Fold4 Ep24: tr 0.473/0.910 | va 0.939/0.815 mF1 0.780
  Fold4 Ep25: tr 0.437/0.927 | va 0.948/0.815 mF1 0.779
  Fold4 Ep26: tr 0.419/0.924 | va 0.961/0.793 mF1 0.750
  Fold4 Ep27: tr 0.396/0.935 | va 0.980/0.815 mF1 0.781
  Fold4 Ep28: tr 0.470/0.918 | va 0.935/0.804 mF1 0.787
  Fold4 Ep29: tr 0.436/0.929 | va 0.972/0.815 mF1 0.786
  Fold4 Ep30: tr 0.428/0.921 | va 0.909/0.815 mF1 0.792
  Fold5 Ep1: tr 2.034/0.168 | va 2.069/0.304 mF1 0.265
  Fold5 Ep2: tr 1.477/0.443 | va 1.661/0.391 mF1 0.351
  Fold5 Ep3: tr 1.112/0.595 | va 1.387/0.554 mF1 0.484
  Fold5 Ep4: tr 0.940/0.712 | va 1.441/0.576 mF1 0.523
  Fold5 Ep5: tr 0.890/0.726 | va 1.142/0.707 mF1 0.668
  Fold5 Ep6: tr 0.742/0.832 | va 1.091/0.696 mF1 0.662
  Fold5 Ep7: tr 0.561/0.859 | va 1.026/0.717 mF1 0.683
  Fold5 Ep8: tr 0.525/0.878 | va 1.078/0.696 mF1 0.652
  Fold5 Ep9: tr 0.558/0.856 | va 1.046/0.717 mF1 0.682
  Fold5 Ep10: tr 0.472/0.918 | va 0.932/0.783 mF1 0.767
  Fold5 Ep11: tr 0.524/0.910 | va 0.908/0.804 mF1 0.794
  Fold5 Ep12: tr 0.453/0.935 | va 0.910/0.804 mF1 0.786
  Fold5 Ep13: tr 0.460/0.910 | va 0.919/0.815 mF1 0.789
  Fold5 Ep14: tr 0.468/0.905 | va 0.893/0.837 mF1 0.820
  Fold5 Ep15: tr 0.469/0.932 | va 0.888/0.804 mF1 0.790
  Fold5 Ep16: tr 0.399/0.962 | va 0.901/0.793 mF1 0.775
  Fold5 Ep17: tr 0.406/0.962 | va 0.883/0.815 mF1 0.807
  Fold5 Ep18: tr 0.445/0.927 | va 0.862/0.815 mF1 0.807
  Fold5 Ep19: tr 0.409/0.948 | va 0.873/0.815 mF1 0.807
  Fold5 Ep20: tr 0.430/0.916 | va 0.857/0.815 mF1 0.807
  Fold5 Ep21: tr 0.430/0.943 | va 0.850/0.815 mF1 0.807
  Fold5 Ep22: tr 0.408/0.935 | va 0.861/0.815 mF1 0.809
  Fold5 Ep23: tr 0.428/0.940 | va 0.851/0.826 mF1 0.814
  Fold5 Ep24: tr 0.422/0.965 | va 0.855/0.826 mF1 0.821
  Fold5 Ep25: tr 0.387/0.957 | va 0.850/0.837 mF1 0.838
  Fold5 Ep26: tr 0.390/0.959 | va 0.863/0.826 mF1 0.818
  Fold5 Ep27: tr 0.460/0.938 | va 0.844/0.826 mF1 0.826
  Fold5 Ep28: tr 0.400/0.973 | va 0.890/0.826 mF1 0.809
  Fold5 Ep29: tr 0.430/0.932 | va 0.892/0.804 mF1 0.792
  Fold5 Ep30: tr 0.434/0.943 | va 0.861/0.870 mF1 0.864

  ===== Evaluating GRU-Seq ensemble on the hold-out set =====
  Final Test Set Accuracy (GRU-Seq Ensemble): 0.8435

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.88      0.75      0.81        20
           Cow       0.93      0.87      0.90        15
           Dog       0.90      0.90      0.90        20
        Donkey       1.00      1.00      1.00         5
          Frog       0.75      0.43      0.55         7
          Lion       0.82      1.00      0.90         9
        Maymun       0.40      0.40      0.40         5
         Sheep       0.50      0.75      0.60         8
         Tavuk       0.86      1.00      0.92         6

      accuracy                           0.84       115
     macro avg       0.80      0.81      0.80       115
  weighted avg       0.86      0.84      0.84       115
  """
  log_text_layer1_h128_dr03 = """
  Fold1 Ep1: tr 2.093/0.204 | va 2.206/0.163 mF1 0.114
  Fold1 Ep2: tr 1.730/0.299 | va 1.938/0.304 mF1 0.285
  Fold1 Ep3: tr 1.339/0.530 | va 1.574/0.467 mF1 0.447
  Fold1 Ep4: tr 1.018/0.723 | va 1.326/0.576 mF1 0.517
  Fold1 Ep5: tr 0.795/0.783 | va 1.271/0.630 mF1 0.628
  Fold1 Ep6: tr 0.720/0.810 | va 1.138/0.674 mF1 0.649
  Fold1 Ep7: tr 0.618/0.848 | va 1.150/0.674 mF1 0.668
  Fold1 Ep8: tr 0.691/0.840 | va 1.183/0.728 mF1 0.691
  Fold1 Ep9: tr 0.621/0.793 | va 1.012/0.783 mF1 0.741
  Fold1 Ep10: tr 0.571/0.853 | va 1.011/0.772 mF1 0.731
  Fold1 Ep11: tr 0.568/0.878 | va 1.033/0.761 mF1 0.733
  Fold1 Ep12: tr 0.562/0.880 | va 0.996/0.815 mF1 0.789
  Fold1 Ep13: tr 0.497/0.913 | va 1.005/0.761 mF1 0.730
  Fold1 Ep14: tr 0.479/0.902 | va 0.935/0.804 mF1 0.778
  Fold1 Ep15: tr 0.476/0.924 | va 0.919/0.826 mF1 0.806
  Fold1 Ep16: tr 0.495/0.880 | va 0.917/0.815 mF1 0.784
  Fold1 Ep17: tr 0.513/0.889 | va 0.935/0.837 mF1 0.808
  Fold1 Ep18: tr 0.445/0.921 | va 0.936/0.837 mF1 0.808
  Fold1 Ep19: tr 0.449/0.924 | va 0.923/0.826 mF1 0.792
  Fold1 Ep20: tr 0.468/0.894 | va 0.926/0.826 mF1 0.792
  Fold1 Ep21: tr 0.478/0.899 | va 0.933/0.815 mF1 0.783
  Fold1 Ep22: tr 0.476/0.916 | va 0.918/0.826 mF1 0.792
  Fold1 Ep23: tr 0.483/0.910 | va 0.923/0.815 mF1 0.783
  Fold1 Ep24: tr 0.459/0.916 | va 0.910/0.815 mF1 0.783
  Fold1 Ep25: tr 0.490/0.902 | va 0.906/0.804 mF1 0.782
  Fold1 Ep26: tr 0.440/0.927 | va 0.915/0.804 mF1 0.784
  Fold1 Ep27: tr 0.515/0.908 | va 0.894/0.826 mF1 0.809
  Fold1 Ep28: tr 0.450/0.929 | va 0.897/0.815 mF1 0.783
  Fold1 Ep29: tr 0.457/0.921 | va 0.903/0.859 mF1 0.830
  Fold1 Ep30: tr 0.429/0.946 | va 0.953/0.783 mF1 0.767
  Fold2 Ep1: tr 2.063/0.198 | va 2.100/0.239 mF1 0.215
  Fold2 Ep2: tr 1.574/0.443 | va 1.741/0.359 mF1 0.375
  Fold2 Ep3: tr 1.256/0.503 | va 1.327/0.489 mF1 0.453
  Fold2 Ep4: tr 0.901/0.723 | va 1.132/0.663 mF1 0.651
  Fold2 Ep5: tr 0.803/0.734 | va 1.173/0.543 mF1 0.534
  Fold2 Ep6: tr 0.815/0.742 | va 1.071/0.717 mF1 0.706
  Fold2 Ep7: tr 0.771/0.793 | va 0.942/0.761 mF1 0.735
  Fold2 Ep8: tr 0.603/0.848 | va 0.968/0.674 mF1 0.647
  Fold2 Ep9: tr 0.639/0.834 | va 0.863/0.750 mF1 0.729
  Fold2 Ep10: tr 0.600/0.842 | va 0.905/0.804 mF1 0.778
  Fold2 Ep11: tr 0.472/0.913 | va 0.905/0.815 mF1 0.798
  Fold2 Ep12: tr 0.507/0.897 | va 0.925/0.728 mF1 0.746
  Fold2 Ep13: tr 0.505/0.867 | va 0.835/0.804 mF1 0.776
  Fold2 Ep14: tr 0.491/0.910 | va 0.833/0.826 mF1 0.810
  Fold2 Ep15: tr 0.490/0.905 | va 0.926/0.837 mF1 0.816
  Fold2 Ep16: tr 0.469/0.902 | va 0.957/0.815 mF1 0.778
  Fold2 Ep17: tr 0.510/0.867 | va 0.944/0.859 mF1 0.821
  Fold2 Ep18: tr 0.507/0.902 | va 0.916/0.837 mF1 0.802
  Fold2 Ep19: tr 0.450/0.940 | va 0.922/0.837 mF1 0.800
  Fold2 Ep20: tr 0.429/0.943 | va 0.927/0.815 mF1 0.785
  Fold2 Ep21: tr 0.481/0.908 | va 0.950/0.826 mF1 0.789
  Fold2 Ep22: tr 0.457/0.921 | va 0.936/0.837 mF1 0.809
  Fold2 Ep23: tr 0.436/0.924 | va 0.927/0.848 mF1 0.824
  Fold2 Ep24: tr 0.452/0.921 | va 0.923/0.837 mF1 0.812
  Fold3 Ep1: tr 2.056/0.168 | va 2.209/0.098 mF1 0.072
  Fold3 Ep2: tr 1.537/0.462 | va 1.801/0.370 mF1 0.326
  Fold3 Ep3: tr 1.143/0.595 | va 1.668/0.554 mF1 0.531
  Fold3 Ep4: tr 0.840/0.788 | va 1.629/0.609 mF1 0.561
  Fold3 Ep5: tr 0.680/0.826 | va 1.738/0.500 mF1 0.472
  Fold3 Ep6: tr 0.764/0.753 | va 1.623/0.620 mF1 0.585
  Fold3 Ep7: tr 0.650/0.823 | va 1.537/0.652 mF1 0.613
  Fold3 Ep8: tr 0.564/0.880 | va 1.571/0.641 mF1 0.603
  Fold3 Ep9: tr 0.532/0.853 | va 1.504/0.685 mF1 0.630
  Fold3 Ep10: tr 0.512/0.897 | va 1.518/0.685 mF1 0.629
  Fold3 Ep11: tr 0.524/0.902 | va 1.502/0.696 mF1 0.646
  Fold3 Ep12: tr 0.445/0.946 | va 1.494/0.674 mF1 0.618
  Fold3 Ep13: tr 0.441/0.927 | va 1.547/0.696 mF1 0.648
  Fold3 Ep14: tr 0.463/0.918 | va 1.505/0.728 mF1 0.674
  Fold3 Ep15: tr 0.414/0.943 | va 1.421/0.707 mF1 0.643
  Fold3 Ep16: tr 0.420/0.946 | va 1.440/0.685 mF1 0.620
  Fold3 Ep17: tr 0.446/0.927 | va 1.428/0.696 mF1 0.638
  Fold3 Ep18: tr 0.395/0.957 | va 1.430/0.696 mF1 0.648
  Fold3 Ep19: tr 0.405/0.951 | va 1.441/0.707 mF1 0.660
  Fold3 Ep20: tr 0.457/0.946 | va 1.440/0.707 mF1 0.660
  Fold3 Ep21: tr 0.414/0.962 | va 1.439/0.707 mF1 0.665
  Fold3 Ep22: tr 0.390/0.959 | va 1.435/0.707 mF1 0.659
  Fold3 Ep23: tr 0.444/0.927 | va 1.448/0.696 mF1 0.645
  Fold3 Ep24: tr 0.441/0.921 | va 1.424/0.717 mF1 0.677
  Fold3 Ep25: tr 0.418/0.954 | va 1.430/0.707 mF1 0.655
  Fold4 Ep1: tr 2.005/0.231 | va 2.157/0.152 mF1 0.121
  Fold4 Ep2: tr 1.562/0.402 | va 1.719/0.293 mF1 0.289
  Fold4 Ep3: tr 1.189/0.549 | va 1.391/0.435 mF1 0.433
  Fold4 Ep4: tr 1.017/0.682 | va 1.292/0.522 mF1 0.507
  Fold4 Ep5: tr 0.865/0.742 | va 1.053/0.728 mF1 0.706
  Fold4 Ep6: tr 0.667/0.845 | va 1.055/0.707 mF1 0.679
  Fold4 Ep7: tr 0.541/0.894 | va 1.022/0.761 mF1 0.740
  Fold4 Ep8: tr 0.626/0.821 | va 0.980/0.761 mF1 0.745
  Fold4 Ep9: tr 0.558/0.878 | va 0.916/0.772 mF1 0.763
  Fold4 Ep10: tr 0.525/0.878 | va 1.044/0.739 mF1 0.717
  Fold4 Ep11: tr 0.507/0.889 | va 0.943/0.772 mF1 0.760
  Fold4 Ep12: tr 0.465/0.910 | va 0.866/0.783 mF1 0.788
  Fold4 Ep13: tr 0.453/0.932 | va 0.833/0.793 mF1 0.785
  Fold4 Ep14: tr 0.445/0.929 | va 0.881/0.793 mF1 0.782
  Fold4 Ep15: tr 0.410/0.935 | va 0.878/0.783 mF1 0.760
  Fold4 Ep16: tr 0.427/0.927 | va 0.888/0.772 mF1 0.744
  Fold4 Ep17: tr 0.418/0.946 | va 0.892/0.772 mF1 0.748
  Fold4 Ep18: tr 0.406/0.951 | va 0.887/0.761 mF1 0.741
  Fold4 Ep19: tr 0.420/0.921 | va 0.873/0.761 mF1 0.741
  Fold4 Ep20: tr 0.461/0.918 | va 0.869/0.783 mF1 0.772
  Fold4 Ep21: tr 0.412/0.929 | va 0.855/0.815 mF1 0.803
  Fold4 Ep22: tr 0.436/0.921 | va 0.868/0.793 mF1 0.784
  Fold4 Ep23: tr 0.396/0.954 | va 0.872/0.783 mF1 0.759
  Fold5 Ep1: tr 2.110/0.152 | va 2.087/0.250 mF1 0.118
  Fold5 Ep2: tr 1.664/0.375 | va 1.743/0.337 mF1 0.270
  Fold5 Ep3: tr 1.310/0.554 | va 1.487/0.424 mF1 0.369
  Fold5 Ep4: tr 1.053/0.671 | va 1.258/0.598 mF1 0.523
  Fold5 Ep5: tr 0.950/0.720 | va 1.049/0.707 mF1 0.656
  Fold5 Ep6: tr 0.814/0.777 | va 0.984/0.761 mF1 0.715
  Fold5 Ep7: tr 0.706/0.818 | va 0.975/0.783 mF1 0.745
  Fold5 Ep8: tr 0.594/0.870 | va 0.924/0.783 mF1 0.747
  Fold5 Ep9: tr 0.679/0.845 | va 0.979/0.772 mF1 0.739
  Fold5 Ep10: tr 0.519/0.867 | va 0.873/0.804 mF1 0.792
  Fold5 Ep11: tr 0.499/0.894 | va 0.850/0.826 mF1 0.811
  Fold5 Ep12: tr 0.500/0.924 | va 0.882/0.793 mF1 0.768
  Fold5 Ep13: tr 0.442/0.935 | va 0.870/0.804 mF1 0.780
  Fold5 Ep14: tr 0.478/0.929 | va 0.846/0.826 mF1 0.801
  Fold5 Ep15: tr 0.463/0.927 | va 0.847/0.815 mF1 0.791
  Fold5 Ep16: tr 0.446/0.943 | va 0.843/0.793 mF1 0.753
  Fold5 Ep17: tr 0.412/0.927 | va 0.845/0.826 mF1 0.802
  Fold5 Ep18: tr 0.389/0.951 | va 0.834/0.804 mF1 0.767
  Fold5 Ep19: tr 0.412/0.954 | va 0.832/0.815 mF1 0.793
  Fold5 Ep20: tr 0.416/0.938 | va 0.835/0.815 mF1 0.793
  Fold5 Ep21: tr 0.412/0.959 | va 0.829/0.815 mF1 0.793
  Fold5 Ep22: tr 0.378/0.967 | va 0.821/0.837 mF1 0.813
  Fold5 Ep23: tr 0.404/0.965 | va 0.823/0.815 mF1 0.793
  Fold5 Ep24: tr 0.377/0.967 | va 0.818/0.815 mF1 0.791
  Fold5 Ep25: tr 0.503/0.902 | va 0.830/0.815 mF1 0.808
  Fold5 Ep26: tr 0.422/0.938 | va 0.850/0.815 mF1 0.805
  Fold5 Ep27: tr 0.414/0.954 | va 0.815/0.804 mF1 0.778
  Fold5 Ep28: tr 0.373/0.965 | va 0.821/0.815 mF1 0.778
  Fold5 Ep29: tr 0.415/0.951 | va 0.807/0.815 mF1 0.790
  Fold5 Ep30: tr 0.414/0.943 | va 0.808/0.826 mF1 0.798

  ===== Evaluating GRU-Seq ensemble on the hold-out set =====

  Final Test Set Accuracy (GRU-Seq Ensemble): 0.8435

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.88      0.75      0.81        20
           Cow       1.00      0.93      0.97        15
           Dog       1.00      0.90      0.95        20
        Donkey       1.00      1.00      1.00         5
          Frog       0.60      0.43      0.50         7
          Lion       0.64      1.00      0.78         9
        Maymun       0.50      0.40      0.44         5
         Sheep       0.55      0.75      0.63         8
         Tavuk       0.71      0.83      0.77         6

      accuracy                           0.84       115
     macro avg       0.79      0.80      0.79       115
  weighted avg       0.86      0.84      0.84       115
  """
  log_text_layer1_h256_dr02_global = """
  Fold1 Ep1: tr 1.924/0.283 | va 2.044/0.228 mF1 0.201
  Fold1 Ep2: tr 1.269/0.584 | va 1.604/0.489 mF1 0.459
  Fold1 Ep3: tr 0.992/0.698 | va 1.327/0.620 mF1 0.564
  Fold1 Ep4: tr 0.810/0.783 | va 1.137/0.728 mF1 0.714
  Fold1 Ep5: tr 0.686/0.829 | va 1.119/0.739 mF1 0.727
  Fold1 Ep6: tr 0.686/0.826 | va 1.215/0.685 mF1 0.642
  Fold1 Ep7: tr 0.601/0.829 | va 1.130/0.772 mF1 0.753
  Fold1 Ep8: tr 0.568/0.861 | va 1.079/0.728 mF1 0.681
  Fold1 Ep9: tr 0.605/0.842 | va 1.036/0.783 mF1 0.765
  Fold1 Ep10: tr 0.492/0.899 | va 1.038/0.761 mF1 0.729
  Fold1 Ep11: tr 0.535/0.886 | va 0.966/0.826 mF1 0.795
  Fold1 Ep12: tr 0.428/0.948 | va 0.890/0.826 mF1 0.793
  Fold1 Ep13: tr 0.420/0.938 | va 0.923/0.804 mF1 0.773
  Fold1 Ep14: tr 0.466/0.894 | va 0.923/0.837 mF1 0.806
  Fold1 Ep15: tr 0.465/0.927 | va 0.880/0.837 mF1 0.800
  Fold1 Ep16: tr 0.399/0.954 | va 0.871/0.859 mF1 0.822
  Fold1 Ep17: tr 0.418/0.951 | va 0.857/0.859 mF1 0.831
  Fold1 Ep18: tr 0.411/0.946 | va 0.853/0.880 mF1 0.849
  Fold1 Ep19: tr 0.413/0.943 | va 0.853/0.870 mF1 0.837
  Fold1 Ep20: tr 0.402/0.948 | va 0.853/0.859 mF1 0.830
  Fold1 Ep21: tr 0.390/0.954 | va 0.856/0.859 mF1 0.830
  Fold1 Ep22: tr 0.393/0.948 | va 0.834/0.870 mF1 0.846
  Fold1 Ep23: tr 0.395/0.959 | va 0.827/0.870 mF1 0.847
  Fold1 Ep24: tr 0.384/0.957 | va 0.843/0.859 mF1 0.835
  Fold1 Ep25: tr 0.405/0.943 | va 0.845/0.848 mF1 0.817
  Fold1 Ep26: tr 0.376/0.954 | va 0.849/0.870 mF1 0.835
  Fold1 Ep27: tr 0.410/0.954 | va 0.878/0.870 mF1 0.833
  Fold1 Ep28: tr 0.362/0.970 | va 0.847/0.880 mF1 0.853
  Fold1 Ep29: tr 0.368/0.959 | va 0.837/0.891 mF1 0.861
  Fold1 Ep30: tr 0.429/0.946 | va 0.785/0.913 mF1 0.874
  Fold2 Ep1: tr 2.030/0.212 | va 2.071/0.185 mF1 0.147
  Fold2 Ep2: tr 1.454/0.429 | va 1.621/0.391 mF1 0.430
  Fold2 Ep3: tr 1.105/0.628 | va 1.162/0.620 mF1 0.611
  Fold2 Ep4: tr 0.773/0.758 | va 1.019/0.772 mF1 0.758
  Fold2 Ep5: tr 0.696/0.834 | va 0.989/0.793 mF1 0.781
  Fold2 Ep6: tr 0.727/0.802 | va 1.103/0.750 mF1 0.677
  Fold2 Ep7: tr 0.624/0.861 | va 0.979/0.772 mF1 0.773
  Fold2 Ep8: tr 0.552/0.864 | va 0.892/0.804 mF1 0.835
  Fold2 Ep9: tr 0.465/0.905 | va 0.908/0.826 mF1 0.799
  Fold2 Ep10: tr 0.592/0.853 | va 0.830/0.848 mF1 0.835
  Fold2 Ep11: tr 0.547/0.880 | va 0.807/0.826 mF1 0.836
  Fold2 Ep12: tr 0.499/0.908 | va 0.748/0.870 mF1 0.882
  Fold2 Ep13: tr 0.405/0.951 | va 0.790/0.859 mF1 0.873
  Fold2 Ep14: tr 0.401/0.954 | va 0.748/0.837 mF1 0.843
  Fold2 Ep15: tr 0.409/0.943 | va 0.739/0.848 mF1 0.854
  Fold2 Ep16: tr 0.423/0.932 | va 0.764/0.859 mF1 0.864
  Fold2 Ep17: tr 0.403/0.957 | va 0.750/0.870 mF1 0.875
  Fold2 Ep18: tr 0.385/0.962 | va 0.754/0.859 mF1 0.873
  Fold2 Ep19: tr 0.374/0.976 | va 0.746/0.870 mF1 0.887
  Fold2 Ep20: tr 0.422/0.954 | va 0.749/0.880 mF1 0.889
  Fold2 Ep21: tr 0.400/0.957 | va 0.769/0.859 mF1 0.873
  Fold2 Ep22: tr 0.393/0.965 | va 0.742/0.870 mF1 0.882
  Fold2 Ep23: tr 0.465/0.957 | va 0.726/0.870 mF1 0.879
  Fold2 Ep24: tr 0.389/0.973 | va 0.750/0.870 mF1 0.879
  Fold2 Ep25: tr 0.422/0.967 | va 0.727/0.870 mF1 0.876
  Fold2 Ep26: tr 0.399/0.946 | va 0.731/0.859 mF1 0.864
  Fold2 Ep27: tr 0.414/0.962 | va 0.725/0.902 mF1 0.901
  Fold2 Ep28: tr 0.407/0.951 | va 0.730/0.891 mF1 0.894
  Fold2 Ep29: tr 0.416/0.973 | va 0.687/0.859 mF1 0.865
  Fold2 Ep30: tr 0.367/0.967 | va 0.697/0.859 mF1 0.857
  Fold3 Ep1: tr 1.991/0.185 | va 2.125/0.076 mF1 0.083
  Fold3 Ep2: tr 1.521/0.389 | va 1.767/0.543 mF1 0.507
  Fold3 Ep3: tr 1.145/0.644 | va 1.575/0.543 mF1 0.474
  Fold3 Ep4: tr 0.884/0.707 | va 1.383/0.685 mF1 0.647
  Fold3 Ep5: tr 0.662/0.823 | va 1.331/0.652 mF1 0.579
  Fold3 Ep6: tr 0.637/0.821 | va 1.296/0.707 mF1 0.651
  Fold3 Ep7: tr 0.693/0.821 | va 1.404/0.630 mF1 0.583
  Fold3 Ep8: tr 0.602/0.861 | va 1.266/0.761 mF1 0.695
  Fold3 Ep9: tr 0.559/0.883 | va 1.149/0.739 mF1 0.698
  Fold3 Ep10: tr 0.482/0.886 | va 1.283/0.707 mF1 0.634
  Fold3 Ep11: tr 0.512/0.905 | va 1.163/0.772 mF1 0.705
  Fold3 Ep12: tr 0.518/0.908 | va 1.146/0.772 mF1 0.716
  Fold3 Ep13: tr 0.408/0.962 | va 1.181/0.750 mF1 0.683
  Fold3 Ep14: tr 0.480/0.910 | va 1.120/0.750 mF1 0.681
  Fold3 Ep15: tr 0.382/0.965 | va 1.135/0.761 mF1 0.696
  Fold3 Ep16: tr 0.395/0.962 | va 1.158/0.750 mF1 0.682
  Fold3 Ep17: tr 0.375/0.962 | va 1.119/0.739 mF1 0.671
  Fold3 Ep18: tr 0.415/0.973 | va 1.110/0.750 mF1 0.684
  Fold3 Ep19: tr 0.367/0.989 | va 1.116/0.750 mF1 0.684
  Fold3 Ep20: tr 0.388/0.976 | va 1.129/0.739 mF1 0.668
  Fold3 Ep21: tr 0.425/0.957 | va 1.142/0.739 mF1 0.668
  Fold3 Ep22: tr 0.390/0.957 | va 1.134/0.717 mF1 0.641
  Fold3 Ep23: tr 0.356/0.989 | va 1.110/0.750 mF1 0.684
  Fold3 Ep24: tr 0.387/0.967 | va 1.114/0.739 mF1 0.668
  Fold3 Ep25: tr 0.386/0.967 | va 1.138/0.750 mF1 0.680
  Fold3 Ep26: tr 0.446/0.957 | va 1.123/0.750 mF1 0.688
  Fold3 Ep27: tr 0.375/0.962 | va 1.090/0.761 mF1 0.705
  Fold3 Ep28: tr 0.402/0.959 | va 1.106/0.761 mF1 0.687
  Fold3 Ep29: tr 0.386/0.976 | va 1.017/0.837 mF1 0.809
  Fold3 Ep30: tr 0.350/0.970 | va 1.001/0.815 mF1 0.779
  Fold4 Ep1: tr 2.014/0.217 | va 2.008/0.217 mF1 0.218
  Fold4 Ep2: tr 1.321/0.546 | va 1.426/0.609 mF1 0.567
  Fold4 Ep3: tr 1.001/0.739 | va 1.278/0.652 mF1 0.633
  Fold4 Ep4: tr 0.716/0.783 | va 1.228/0.696 mF1 0.644
  Fold4 Ep5: tr 0.688/0.807 | va 1.083/0.772 mF1 0.747
  Fold4 Ep6: tr 0.552/0.889 | va 1.058/0.772 mF1 0.747
  Fold4 Ep7: tr 0.509/0.872 | va 1.028/0.750 mF1 0.696
  Fold4 Ep8: tr 0.542/0.886 | va 0.962/0.804 mF1 0.767
  Fold4 Ep9: tr 0.477/0.883 | va 1.087/0.750 mF1 0.717
  Fold4 Ep10: tr 0.471/0.927 | va 1.000/0.761 mF1 0.729
  Fold4 Ep11: tr 0.525/0.910 | va 1.104/0.793 mF1 0.745
  Fold4 Ep12: tr 0.424/0.954 | va 0.940/0.826 mF1 0.787
  Fold4 Ep13: tr 0.389/0.965 | va 0.887/0.826 mF1 0.786
  Fold4 Ep14: tr 0.423/0.946 | va 0.907/0.783 mF1 0.762
  Fold4 Ep15: tr 0.381/0.957 | va 0.892/0.804 mF1 0.774
  Fold4 Ep16: tr 0.383/0.965 | va 0.895/0.826 mF1 0.791
  Fold4 Ep17: tr 0.389/0.967 | va 0.889/0.848 mF1 0.811
  Fold4 Ep18: tr 0.392/0.959 | va 0.881/0.826 mF1 0.794
  Fold4 Ep19: tr 0.398/0.962 | va 0.886/0.815 mF1 0.784
  Fold4 Ep20: tr 0.383/0.965 | va 0.877/0.826 mF1 0.794
  Fold4 Ep21: tr 0.413/0.962 | va 0.886/0.826 mF1 0.794
  Fold4 Ep22: tr 0.370/0.978 | va 0.872/0.826 mF1 0.794
  Fold4 Ep23: tr 0.401/0.967 | va 0.885/0.826 mF1 0.794
  Fold4 Ep24: tr 0.367/0.976 | va 0.892/0.815 mF1 0.784
  Fold4 Ep25: tr 0.357/0.981 | va 0.891/0.793 mF1 0.753
  Fold4 Ep26: tr 0.381/0.965 | va 0.874/0.815 mF1 0.775
  Fold4 Ep27: tr 0.414/0.943 | va 0.871/0.826 mF1 0.790
  Fold4 Ep28: tr 0.383/0.962 | va 0.826/0.837 mF1 0.794
  Fold4 Ep29: tr 0.377/0.970 | va 0.990/0.793 mF1 0.748
  Fold4 Ep30: tr 0.410/0.948 | va 0.947/0.772 mF1 0.725
  Fold5 Ep1: tr 2.041/0.188 | va 2.064/0.109 mF1 0.099
  Fold5 Ep2: tr 1.536/0.370 | va 1.549/0.489 mF1 0.470
  Fold5 Ep3: tr 1.195/0.622 | va 1.270/0.630 mF1 0.616
  Fold5 Ep4: tr 0.848/0.755 | va 1.199/0.663 mF1 0.623
  Fold5 Ep5: tr 0.711/0.774 | va 1.209/0.674 mF1 0.655
  Fold5 Ep6: tr 0.749/0.796 | va 0.941/0.728 mF1 0.714
  Fold5 Ep7: tr 0.695/0.818 | va 1.128/0.696 mF1 0.664
  Fold5 Ep8: tr 0.580/0.845 | va 1.055/0.717 mF1 0.699
  Fold5 Ep9: tr 0.571/0.845 | va 0.901/0.826 mF1 0.799
  Fold5 Ep10: tr 0.561/0.875 | va 0.822/0.815 mF1 0.797
  Fold5 Ep11: tr 0.502/0.894 | va 0.871/0.783 mF1 0.772
  Fold5 Ep12: tr 0.489/0.899 | va 0.873/0.815 mF1 0.797
  Fold5 Ep13: tr 0.422/0.929 | va 0.844/0.815 mF1 0.789
  Fold5 Ep14: tr 0.407/0.946 | va 0.875/0.815 mF1 0.791
  Fold5 Ep15: tr 0.414/0.965 | va 0.901/0.848 mF1 0.815
  Fold5 Ep16: tr 0.418/0.948 | va 0.882/0.870 mF1 0.834
  Fold5 Ep17: tr 0.443/0.943 | va 0.883/0.837 mF1 0.803
  Fold5 Ep18: tr 0.429/0.938 | va 0.878/0.837 mF1 0.803
  Fold5 Ep19: tr 0.426/0.938 | va 0.859/0.859 mF1 0.836
  Fold5 Ep20: tr 0.371/0.965 | va 0.855/0.848 mF1 0.825

  ===== Evaluating GRU-Seq ensemble on the hold-out set =====

  Final Test Set Accuracy (GRU-Seq Ensemble): 0.8957

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.94      0.85      0.89        20
           Cow       0.88      1.00      0.94        15
           Dog       0.95      0.90      0.92        20
        Donkey       1.00      1.00      1.00         5
          Frog       1.00      0.43      0.60         7
          Lion       0.82      1.00      0.90         9
        Maymun       0.50      0.60      0.55         5
         Sheep       0.70      0.88      0.78         8
         Tavuk       1.00      1.00      1.00         6

      accuracy                           0.90       115
     macro avg       0.88      0.87      0.86       115
  weighted avg       0.91      0.90      0.89       115
  """
  log_text_cnn_first = """
  ========== Fold 1/5 ==========
  Epoch 1/20 -> Train Loss: 1.8719, Train Acc: 0.5245 | Val Loss: 144.6801, Val Acc: 0.2065
  ✨ New best model saved with accuracy: 0.2065
  Epoch 2/20 -> Train Loss: 0.8127, Train Acc: 0.7772 | Val Loss: 0.7333, Val Acc: 0.8370
  ✨ New best model saved with accuracy: 0.8370
  Epoch 3/20 -> Train Loss: 0.6013, Train Acc: 0.8370 | Val Loss: 0.5235, Val Acc: 0.8261
  Epoch 4/20 -> Train Loss: 0.4195, Train Acc: 0.8723 | Val Loss: 0.5878, Val Acc: 0.8804
  ✨ New best model saved with accuracy: 0.8804
  Epoch 5/20 -> Train Loss: 0.3873, Train Acc: 0.9022 | Val Loss: 0.5083, Val Acc: 0.9130
  ✨ New best model saved with accuracy: 0.9130
  Epoch 6/20 -> Train Loss: 0.3350, Train Acc: 0.9076 | Val Loss: 0.2208, Val Acc: 0.9239
  ✨ New best model saved with accuracy: 0.9239
  Epoch 7/20 -> Train Loss: 0.1760, Train Acc: 0.9538 | Val Loss: 0.2489, Val Acc: 0.9457
  ✨ New best model saved with accuracy: 0.9457
  Epoch 8/20 -> Train Loss: 0.2321, Train Acc: 0.9402 | Val Loss: 0.4567, Val Acc: 0.9022
  Epoch 9/20 -> Train Loss: 0.2068, Train Acc: 0.9321 | Val Loss: 0.3468, Val Acc: 0.9239
  Epoch 10/20 -> Train Loss: 0.1183, Train Acc: 0.9620 | Val Loss: 0.5940, Val Acc: 0.9022
  Epoch 11/20 -> Train Loss: 0.1253, Train Acc: 0.9674 | Val Loss: 0.2247, Val Acc: 0.9457
  Epoch 12/20 -> Train Loss: 0.0994, Train Acc: 0.9755 | Val Loss: 0.2526, Val Acc: 0.9130
  Epoch 13/20 -> Train Loss: 0.0514, Train Acc: 0.9891 | Val Loss: 0.2621, Val Acc: 0.9022
  Epoch 14/20 -> Train Loss: 0.0802, Train Acc: 0.9783 | Val Loss: 0.2522, Val Acc: 0.9348
  Epoch 15/20 -> Train Loss: 0.0399, Train Acc: 0.9837 | Val Loss: 0.2091, Val Acc: 0.9457
  Epoch 16/20 -> Train Loss: 0.0221, Train Acc: 0.9973 | Val Loss: 0.1837, Val Acc: 0.9457
  Epoch 17/20 -> Train Loss: 0.0363, Train Acc: 0.9864 | Val Loss: 0.1815, Val Acc: 0.9457
  Epoch 18/20 -> Train Loss: 0.0239, Train Acc: 0.9918 | Val Loss: 0.1723, Val Acc: 0.9565
  ✨ New best model saved with accuracy: 0.9565
  Epoch 19/20 -> Train Loss: 0.0325, Train Acc: 0.9946 | Val Loss: 0.1698, Val Acc: 0.9565
  Epoch 20/20 -> Train Loss: 0.0237, Train Acc: 0.9891 | Val Loss: 0.1754, Val Acc: 0.9457

  Fold 1 finished. Best validation accuracy: 0.9565

  ========== Fold 2/5 ==========
  Epoch 1/20 -> Train Loss: 2.1156, Train Acc: 0.4348 | Val Loss: 445.9111, Val Acc: 0.1522
  ✨ New best model saved with accuracy: 0.1522
  Epoch 2/20 -> Train Loss: 1.1557, Train Acc: 0.7228 | Val Loss: 0.9123, Val Acc: 0.6848
  ✨ New best model saved with accuracy: 0.6848
  Epoch 3/20 -> Train Loss: 0.7185, Train Acc: 0.8179 | Val Loss: 0.6819, Val Acc: 0.8696
  ✨ New best model saved with accuracy: 0.8696
  Epoch 4/20 -> Train Loss: 0.3629, Train Acc: 0.8668 | Val Loss: 0.4123, Val Acc: 0.9348
  ✨ New best model saved with accuracy: 0.9348
  Epoch 5/20 -> Train Loss: 0.3825, Train Acc: 0.8967 | Val Loss: 0.3578, Val Acc: 0.9239
  Epoch 6/20 -> Train Loss: 0.2112, Train Acc: 0.9402 | Val Loss: 0.3495, Val Acc: 0.9239
  Epoch 7/20 -> Train Loss: 0.2925, Train Acc: 0.9185 | Val Loss: 0.3858, Val Acc: 0.8913
  Epoch 8/20 -> Train Loss: 0.2136, Train Acc: 0.9158 | Val Loss: 0.4040, Val Acc: 0.9022
  Epoch 9/20 -> Train Loss: 0.1781, Train Acc: 0.9348 | Val Loss: 0.1244, Val Acc: 0.9783
  ✨ New best model saved with accuracy: 0.9783
  Epoch 10/20 -> Train Loss: 0.1051, Train Acc: 0.9674 | Val Loss: 0.2054, Val Acc: 0.9674
  Epoch 11/20 -> Train Loss: 0.1498, Train Acc: 0.9457 | Val Loss: 0.2089, Val Acc: 0.9457
  Epoch 12/20 -> Train Loss: 0.0929, Train Acc: 0.9701 | Val Loss: 0.2165, Val Acc: 0.9239
  Epoch 13/20 -> Train Loss: 0.1340, Train Acc: 0.9484 | Val Loss: 0.2413, Val Acc: 0.9348
  Epoch 14/20 -> Train Loss: 0.0426, Train Acc: 0.9837 | Val Loss: 0.1836, Val Acc: 0.9457
  Epoch 15/20 -> Train Loss: 0.0956, Train Acc: 0.9701 | Val Loss: 0.1621, Val Acc: 0.9457
  Epoch 16/20 -> Train Loss: 0.0660, Train Acc: 0.9810 | Val Loss: 0.1360, Val Acc: 0.9457
  Epoch 17/20 -> Train Loss: 0.0579, Train Acc: 0.9837 | Val Loss: 0.1222, Val Acc: 0.9348
  Epoch 18/20 -> Train Loss: 0.0498, Train Acc: 0.9837 | Val Loss: 0.1304, Val Acc: 0.9565
  Epoch 19/20 -> Train Loss: 0.0422, Train Acc: 0.9864 | Val Loss: 0.1311, Val Acc: 0.9674
  Epoch 20/20 -> Train Loss: 0.0966, Train Acc: 0.9647 | Val Loss: 0.1545, Val Acc: 0.9457

  Fold 2 finished. Best validation accuracy: 0.9783

  ========== Fold 3/5 ==========
  Epoch 1/20 -> Train Loss: 2.0682, Train Acc: 0.4918 | Val Loss: 600.8102, Val Acc: 0.1522
  ✨ New best model saved with accuracy: 0.1522
  Epoch 2/20 -> Train Loss: 0.9376, Train Acc: 0.7364 | Val Loss: 1.1400, Val Acc: 0.7391
  ✨ New best model saved with accuracy: 0.7391
  Epoch 3/20 -> Train Loss: 0.5339, Train Acc: 0.8533 | Val Loss: 0.6987, Val Acc: 0.8152
  ✨ New best model saved with accuracy: 0.8152
  Epoch 4/20 -> Train Loss: 0.4288, Train Acc: 0.8560 | Val Loss: 0.9134, Val Acc: 0.8152
  Epoch 5/20 -> Train Loss: 0.4681, Train Acc: 0.8913 | Val Loss: 0.3362, Val Acc: 0.9348
  ✨ New best model saved with accuracy: 0.9348
  Epoch 6/20 -> Train Loss: 0.2620, Train Acc: 0.9185 | Val Loss: 0.3565, Val Acc: 0.9239
  Epoch 7/20 -> Train Loss: 0.1519, Train Acc: 0.9457 | Val Loss: 0.3271, Val Acc: 0.9239
  Epoch 8/20 -> Train Loss: 0.2583, Train Acc: 0.9185 | Val Loss: 0.2823, Val Acc: 0.9348
  Epoch 9/20 -> Train Loss: 0.2528, Train Acc: 0.9321 | Val Loss: 0.3157, Val Acc: 0.9239
  Epoch 10/20 -> Train Loss: 0.2065, Train Acc: 0.9402 | Val Loss: 0.3497, Val Acc: 0.9130
  Epoch 11/20 -> Train Loss: 0.1061, Train Acc: 0.9620 | Val Loss: 0.4297, Val Acc: 0.9022
  Epoch 12/20 -> Train Loss: 0.1672, Train Acc: 0.9565 | Val Loss: 0.2936, Val Acc: 0.9457
  ✨ New best model saved with accuracy: 0.9457
  Epoch 13/20 -> Train Loss: 0.0902, Train Acc: 0.9755 | Val Loss: 0.4752, Val Acc: 0.9130
  Epoch 14/20 -> Train Loss: 0.0715, Train Acc: 0.9810 | Val Loss: 0.4520, Val Acc: 0.9239
  Epoch 15/20 -> Train Loss: 0.0237, Train Acc: 0.9918 | Val Loss: 0.3908, Val Acc: 0.9239
  Epoch 16/20 -> Train Loss: 0.0568, Train Acc: 0.9783 | Val Loss: 0.3817, Val Acc: 0.9239
  Epoch 17/20 -> Train Loss: 0.0757, Train Acc: 0.9783 | Val Loss: 0.3239, Val Acc: 0.9239
  Epoch 18/20 -> Train Loss: 0.0289, Train Acc: 0.9918 | Val Loss: 0.2997, Val Acc: 0.9239
  Epoch 19/20 -> Train Loss: 0.0569, Train Acc: 0.9810 | Val Loss: 0.3189, Val Acc: 0.9239
  Epoch 20/20 -> Train Loss: 0.0258, Train Acc: 0.9946 | Val Loss: 0.3123, Val Acc: 0.9239

  Fold 3 finished. Best validation accuracy: 0.9457

  ========== Fold 4/5 ==========
  Epoch 1/20 -> Train Loss: 2.1823, Train Acc: 0.4918 | Val Loss: 1751.4473, Val Acc: 0.0978
  ✨ New best model saved with accuracy: 0.0978
  Epoch 2/20 -> Train Loss: 0.9139, Train Acc: 0.7717 | Val Loss: 0.8842, Val Acc: 0.7391
  ✨ New best model saved with accuracy: 0.7391
  Epoch 3/20 -> Train Loss: 0.5984, Train Acc: 0.8179 | Val Loss: 0.3441, Val Acc: 0.8696
  ✨ New best model saved with accuracy: 0.8696
  Epoch 4/20 -> Train Loss: 0.5103, Train Acc: 0.8505 | Val Loss: 0.7629, Val Acc: 0.8152
  Epoch 5/20 -> Train Loss: 0.4830, Train Acc: 0.8859 | Val Loss: 0.6479, Val Acc: 0.8696
  Epoch 6/20 -> Train Loss: 0.2029, Train Acc: 0.9293 | Val Loss: 0.5691, Val Acc: 0.8587
  Epoch 7/20 -> Train Loss: 0.1783, Train Acc: 0.9375 | Val Loss: 1.2099, Val Acc: 0.8587
  Epoch 8/20 -> Train Loss: 0.1633, Train Acc: 0.9348 | Val Loss: 0.8830, Val Acc: 0.8587
  Epoch 9/20 -> Train Loss: 0.1122, Train Acc: 0.9647 | Val Loss: 0.6947, Val Acc: 0.8913
  ✨ New best model saved with accuracy: 0.8913
  Epoch 10/20 -> Train Loss: 0.1353, Train Acc: 0.9565 | Val Loss: 0.6405, Val Acc: 0.8913
  Epoch 11/20 -> Train Loss: 0.1060, Train Acc: 0.9728 | Val Loss: 0.6064, Val Acc: 0.8913
  Epoch 12/20 -> Train Loss: 0.0334, Train Acc: 0.9864 | Val Loss: 0.4890, Val Acc: 0.9130
  ✨ New best model saved with accuracy: 0.9130
  Epoch 13/20 -> Train Loss: 0.1281, Train Acc: 0.9620 | Val Loss: 0.4511, Val Acc: 0.9130
  Epoch 14/20 -> Train Loss: 0.0874, Train Acc: 0.9701 | Val Loss: 0.4700, Val Acc: 0.9239
  ✨ New best model saved with accuracy: 0.9239
  Epoch 15/20 -> Train Loss: 0.0851, Train Acc: 0.9701 | Val Loss: 0.5165, Val Acc: 0.9022
  Epoch 16/20 -> Train Loss: 0.0459, Train Acc: 0.9755 | Val Loss: 0.5325, Val Acc: 0.8913
  Epoch 17/20 -> Train Loss: 0.0768, Train Acc: 0.9701 | Val Loss: 0.5343, Val Acc: 0.8913
  Epoch 18/20 -> Train Loss: 0.0685, Train Acc: 0.9783 | Val Loss: 0.4863, Val Acc: 0.8913
  Epoch 19/20 -> Train Loss: 0.0679, Train Acc: 0.9783 | Val Loss: 0.5296, Val Acc: 0.8913
  Epoch 20/20 -> Train Loss: 0.0492, Train Acc: 0.9864 | Val Loss: 0.5126, Val Acc: 0.8913

  Fold 4 finished. Best validation accuracy: 0.9239

  ========== Fold 5/5 ==========
  Epoch 1/20 -> Train Loss: 1.8638, Train Acc: 0.5380 | Val Loss: 297.4087, Val Acc: 0.1630
  ✨ New best model saved with accuracy: 0.1630
  Epoch 2/20 -> Train Loss: 1.0052, Train Acc: 0.7418 | Val Loss: 1.8822, Val Acc: 0.7391
  ✨ New best model saved with accuracy: 0.7391
  Epoch 3/20 -> Train Loss: 0.6592, Train Acc: 0.8179 | Val Loss: 0.7010, Val Acc: 0.8261
  ✨ New best model saved with accuracy: 0.8261
  Epoch 4/20 -> Train Loss: 0.4940, Train Acc: 0.8505 | Val Loss: 0.8481, Val Acc: 0.8043
  Epoch 5/20 -> Train Loss: 0.2846, Train Acc: 0.9076 | Val Loss: 0.5824, Val Acc: 0.8804
  ✨ New best model saved with accuracy: 0.8804
  Epoch 6/20 -> Train Loss: 0.1823, Train Acc: 0.9484 | Val Loss: 0.5369, Val Acc: 0.9022
  ✨ New best model saved with accuracy: 0.9022
  Epoch 7/20 -> Train Loss: 0.2858, Train Acc: 0.9402 | Val Loss: 0.4076, Val Acc: 0.9022
  Epoch 8/20 -> Train Loss: 0.2017, Train Acc: 0.9402 | Val Loss: 0.4695, Val Acc: 0.8804
  Epoch 9/20 -> Train Loss: 0.1716, Train Acc: 0.9484 | Val Loss: 0.3468, Val Acc: 0.8913
  Epoch 10/20 -> Train Loss: 0.1554, Train Acc: 0.9511 | Val Loss: 0.6073, Val Acc: 0.8913
  Epoch 11/20 -> Train Loss: 0.0754, Train Acc: 0.9783 | Val Loss: 0.6041, Val Acc: 0.9239
  ✨ New best model saved with accuracy: 0.9239
  Epoch 12/20 -> Train Loss: 0.0588, Train Acc: 0.9837 | Val Loss: 0.5736, Val Acc: 0.9130
  Epoch 13/20 -> Train Loss: 0.1104, Train Acc: 0.9620 | Val Loss: 0.5856, Val Acc: 0.9130
  Epoch 14/20 -> Train Loss: 0.0522, Train Acc: 0.9810 | Val Loss: 0.5531, Val Acc: 0.9022
  Epoch 15/20 -> Train Loss: 0.0616, Train Acc: 0.9755 | Val Loss: 0.5100, Val Acc: 0.9130
  Epoch 16/20 -> Train Loss: 0.0473, Train Acc: 0.9837 | Val Loss: 0.4586, Val Acc: 0.9239
  Epoch 17/20 -> Train Loss: 0.0751, Train Acc: 0.9783 | Val Loss: 0.5039, Val Acc: 0.9130
  Epoch 18/20 -> Train Loss: 0.0857, Train Acc: 0.9728 | Val Loss: 0.4429, Val Acc: 0.9130
  Epoch 19/20 -> Train Loss: 0.0842, Train Acc: 0.9728 | Val Loss: 0.4772, Val Acc: 0.9130
  Epoch 20/20 -> Train Loss: 0.0518, Train Acc: 0.9783 | Val Loss: 0.5215, Val Acc: 0.9022

  Fold 5 finished. Best validation accuracy: 0.9239

  ===== Evaluating on the final hold-out test set =====

  Final Test Set Accuracy (Ensemble): 0.9478

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       1.00      1.00      1.00        20
           Cow       1.00      1.00      1.00        15
           Dog       0.87      1.00      0.93        20
        Donkey       1.00      1.00      1.00         5
          Frog       0.78      1.00      0.88         7
          Lion       1.00      1.00      1.00         9
        Maymun       0.67      0.40      0.50         5
         Sheep       1.00      1.00      1.00         8
         Tavuk       1.00      0.50      0.67         6

      accuracy                           0.95       115
     macro avg       0.93      0.89      0.90       115
  weighted avg       0.95      0.95      0.94       115
  """
  log_text_cnn_weighted = """
  ========== Fold 1/5 ==========
  Class Weights for Fold 1: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.2690, 2.3000, 1.4720,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.2985, Train Acc: 0.4837 | Val Loss: 3.5312, Val Acc: 0.1957
  ✨ New best model saved with loss: 3.5312, accuracy: 0.1957
  Epoch 2/20 -> Train Loss: 1.0839, Train Acc: 0.7120 | Val Loss: 2.2743, Val Acc: 0.7826
  ✨ New best model saved with loss: 2.2743, accuracy: 0.7826
  Epoch 3/20 -> Train Loss: 0.9047, Train Acc: 0.7663 | Val Loss: 0.8344, Val Acc: 0.8261
  ✨ New best model saved with loss: 0.8344, accuracy: 0.8261
  Epoch 4/20 -> Train Loss: 0.5354, Train Acc: 0.8370 | Val Loss: 0.5996, Val Acc: 0.8587
  ✨ New best model saved with loss: 0.5996, accuracy: 0.8587
  Epoch 5/20 -> Train Loss: 0.3317, Train Acc: 0.8832 | Val Loss: 0.3778, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.3778, accuracy: 0.8913
  Epoch 6/20 -> Train Loss: 0.4317, Train Acc: 0.8777 | Val Loss: 0.4554, Val Acc: 0.8696
  Epoch 7/20 -> Train Loss: 0.2826, Train Acc: 0.9130 | Val Loss: 0.2352, Val Acc: 0.9022
  ✨ New best model saved with loss: 0.2352, accuracy: 0.9022
  Epoch 8/20 -> Train Loss: 0.2482, Train Acc: 0.9321 | Val Loss: 0.3554, Val Acc: 0.9130
  Epoch 9/20 -> Train Loss: 0.1259, Train Acc: 0.9701 | Val Loss: 0.2881, Val Acc: 0.8913
  Epoch 10/20 -> Train Loss: 0.0981, Train Acc: 0.9647 | Val Loss: 0.2549, Val Acc: 0.9130
  Epoch 11/20 -> Train Loss: 0.0965, Train Acc: 0.9674 | Val Loss: 0.1813, Val Acc: 0.9130
  ✨ New best model saved with loss: 0.1813, accuracy: 0.9130
  Epoch 12/20 -> Train Loss: 0.1567, Train Acc: 0.9592 | Val Loss: 0.3044, Val Acc: 0.9348
  Epoch 13/20 -> Train Loss: 0.1417, Train Acc: 0.9783 | Val Loss: 0.1707, Val Acc: 0.9457
  ✨ New best model saved with loss: 0.1707, accuracy: 0.9457
  Epoch 14/20 -> Train Loss: 0.1070, Train Acc: 0.9755 | Val Loss: 0.1830, Val Acc: 0.9348
  Epoch 15/20 -> Train Loss: 0.0462, Train Acc: 0.9810 | Val Loss: 0.2151, Val Acc: 0.9130
  Epoch 16/20 -> Train Loss: 0.0690, Train Acc: 0.9674 | Val Loss: 0.2074, Val Acc: 0.9130
  Epoch 17/20 -> Train Loss: 0.0988, Train Acc: 0.9565 | Val Loss: 0.1854, Val Acc: 0.9457
  Epoch 18/20 -> Train Loss: 0.0793, Train Acc: 0.9647 | Val Loss: 0.1778, Val Acc: 0.9348
  Epoch 19/20 -> Train Loss: 0.0316, Train Acc: 0.9810 | Val Loss: 0.1673, Val Acc: 0.9457
  ✨ New best model saved with loss: 0.1673, accuracy: 0.9457
  Epoch 20/20 -> Train Loss: 0.0525, Train Acc: 0.9864 | Val Loss: 0.1787, Val Acc: 0.9348

  Fold 1 finished. Best validation loss: 0.1673, accuracy: 0.9457

  ========== Fold 2/5 ==========
  Class Weights for Fold 2: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4720,
          1.8400], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.7622, Train Acc: 0.3478 | Val Loss: 181.2737, Val Acc: 0.1087
  ✨ New best model saved with loss: 181.2737, accuracy: 0.1087
  Epoch 2/20 -> Train Loss: 1.3197, Train Acc: 0.6766 | Val Loss: 1.3497, Val Acc: 0.6848
  ✨ New best model saved with loss: 1.3497, accuracy: 0.6848
  Epoch 3/20 -> Train Loss: 0.7807, Train Acc: 0.7799 | Val Loss: 0.5696, Val Acc: 0.8043
  ✨ New best model saved with loss: 0.5696, accuracy: 0.8043
  Epoch 4/20 -> Train Loss: 0.4634, Train Acc: 0.8668 | Val Loss: 0.5505, Val Acc: 0.8370
  ✨ New best model saved with loss: 0.5505, accuracy: 0.8370
  Epoch 5/20 -> Train Loss: 0.5555, Train Acc: 0.8641 | Val Loss: 0.8972, Val Acc: 0.9130
  Epoch 6/20 -> Train Loss: 0.4002, Train Acc: 0.8804 | Val Loss: 0.5222, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.5222, accuracy: 0.9239
  Epoch 7/20 -> Train Loss: 0.3576, Train Acc: 0.9076 | Val Loss: 0.5785, Val Acc: 0.9130
  Epoch 8/20 -> Train Loss: 0.2994, Train Acc: 0.9158 | Val Loss: 0.3199, Val Acc: 0.8804
  Epoch 9/20 -> Train Loss: 0.2395, Train Acc: 0.9266 | Val Loss: 0.2662, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.2662, accuracy: 0.9239
  Epoch 10/20 -> Train Loss: 0.1778, Train Acc: 0.9375 | Val Loss: 0.2120, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.2120, accuracy: 0.9239
  Epoch 11/20 -> Train Loss: 0.1143, Train Acc: 0.9620 | Val Loss: 0.2629, Val Acc: 0.9239
  Epoch 12/20 -> Train Loss: 0.1617, Train Acc: 0.9538 | Val Loss: 0.3409, Val Acc: 0.8587
  Epoch 13/20 -> Train Loss: 0.1765, Train Acc: 0.9538 | Val Loss: 0.2443, Val Acc: 0.9239
  Epoch 14/20 -> Train Loss: 0.0904, Train Acc: 0.9783 | Val Loss: 0.1804, Val Acc: 0.9457
  ✨ New best model saved with loss: 0.1804, accuracy: 0.9457
  Epoch 15/20 -> Train Loss: 0.1004, Train Acc: 0.9701 | Val Loss: 0.1398, Val Acc: 0.9457
  ✨ New best model saved with loss: 0.1398, accuracy: 0.9457
  Epoch 16/20 -> Train Loss: 0.0583, Train Acc: 0.9755 | Val Loss: 0.1274, Val Acc: 0.9565
  ✨ New best model saved with loss: 0.1274, accuracy: 0.9565
  Epoch 17/20 -> Train Loss: 0.0287, Train Acc: 0.9891 | Val Loss: 0.1205, Val Acc: 0.9565
  ✨ New best model saved with loss: 0.1205, accuracy: 0.9565
  Epoch 18/20 -> Train Loss: 0.0543, Train Acc: 0.9837 | Val Loss: 0.1129, Val Acc: 0.9565
  ✨ New best model saved with loss: 0.1129, accuracy: 0.9565
  Epoch 19/20 -> Train Loss: 0.1159, Train Acc: 0.9755 | Val Loss: 0.1067, Val Acc: 0.9783
  ✨ New best model saved with loss: 0.1067, accuracy: 0.9783
  Epoch 20/20 -> Train Loss: 0.0644, Train Acc: 0.9810 | Val Loss: 0.1163, Val Acc: 0.9457

  Fold 2 finished. Best validation loss: 0.1067, accuracy: 0.9783

  ========== Fold 3/5 ==========
  Class Weights for Fold 3: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.3074, Train Acc: 0.4348 | Val Loss: 439.6310, Val Acc: 0.0978
  ✨ New best model saved with loss: 439.6310, accuracy: 0.0978
  Epoch 2/20 -> Train Loss: 1.1739, Train Acc: 0.7364 | Val Loss: 1.7443, Val Acc: 0.6630
  ✨ New best model saved with loss: 1.7443, accuracy: 0.6630
  Epoch 3/20 -> Train Loss: 0.6073, Train Acc: 0.8179 | Val Loss: 0.8777, Val Acc: 0.7717
  ✨ New best model saved with loss: 0.8777, accuracy: 0.7717
  Epoch 4/20 -> Train Loss: 0.5486, Train Acc: 0.8750 | Val Loss: 0.9804, Val Acc: 0.8261
  Epoch 5/20 -> Train Loss: 0.3569, Train Acc: 0.8967 | Val Loss: 1.0430, Val Acc: 0.7935
  Epoch 6/20 -> Train Loss: 0.3710, Train Acc: 0.8913 | Val Loss: 0.6690, Val Acc: 0.8587
  ✨ New best model saved with loss: 0.6690, accuracy: 0.8587
  Epoch 7/20 -> Train Loss: 0.2981, Train Acc: 0.9076 | Val Loss: 0.5093, Val Acc: 0.8804
  ✨ New best model saved with loss: 0.5093, accuracy: 0.8804
  Epoch 8/20 -> Train Loss: 0.3764, Train Acc: 0.8886 | Val Loss: 0.5186, Val Acc: 0.8478
  Epoch 9/20 -> Train Loss: 0.2729, Train Acc: 0.9212 | Val Loss: 0.5368, Val Acc: 0.8152
  Epoch 10/20 -> Train Loss: 0.2746, Train Acc: 0.9239 | Val Loss: 0.4627, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.4627, accuracy: 0.8913
  Epoch 11/20 -> Train Loss: 0.1310, Train Acc: 0.9620 | Val Loss: 0.4538, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.4538, accuracy: 0.8913
  Epoch 12/20 -> Train Loss: 0.0999, Train Acc: 0.9647 | Val Loss: 0.4591, Val Acc: 0.8913
  Epoch 13/20 -> Train Loss: 0.0850, Train Acc: 0.9701 | Val Loss: 0.4217, Val Acc: 0.9022
  ✨ New best model saved with loss: 0.4217, accuracy: 0.9022
  Epoch 14/20 -> Train Loss: 0.0745, Train Acc: 0.9837 | Val Loss: 0.3750, Val Acc: 0.9022
  ✨ New best model saved with loss: 0.3750, accuracy: 0.9022
  Epoch 15/20 -> Train Loss: 0.0459, Train Acc: 0.9837 | Val Loss: 0.3174, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.3174, accuracy: 0.9239
  Epoch 16/20 -> Train Loss: 0.1189, Train Acc: 0.9620 | Val Loss: 0.3197, Val Acc: 0.9130
  Epoch 17/20 -> Train Loss: 0.0552, Train Acc: 0.9620 | Val Loss: 0.3194, Val Acc: 0.9130
  Epoch 18/20 -> Train Loss: 0.0371, Train Acc: 0.9783 | Val Loss: 0.3203, Val Acc: 0.9022
  Epoch 19/20 -> Train Loss: 0.0747, Train Acc: 0.9837 | Val Loss: 0.3037, Val Acc: 0.9130
  Epoch 20/20 -> Train Loss: 0.0534, Train Acc: 0.9837 | Val Loss: 0.2947, Val Acc: 0.9130

  Fold 3 finished. Best validation loss: 0.3174, accuracy: 0.9239

  ========== Fold 4/5 ==========
  Class Weights for Fold 4: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.6711, Train Acc: 0.3967 | Val Loss: 3.4483, Val Acc: 0.2283
  ✨ New best model saved with loss: 3.4483, accuracy: 0.2283
  Epoch 2/20 -> Train Loss: 1.1051, Train Acc: 0.6929 | Val Loss: 1.1014, Val Acc: 0.8043
  ✨ New best model saved with loss: 1.1014, accuracy: 0.8043
  Epoch 3/20 -> Train Loss: 0.7023, Train Acc: 0.7772 | Val Loss: 1.5168, Val Acc: 0.7609
  Epoch 4/20 -> Train Loss: 0.5549, Train Acc: 0.8288 | Val Loss: 0.5408, Val Acc: 0.9022
  ✨ New best model saved with loss: 0.5408, accuracy: 0.9022
  Epoch 5/20 -> Train Loss: 0.3994, Train Acc: 0.8832 | Val Loss: 1.0980, Val Acc: 0.8913
  Epoch 6/20 -> Train Loss: 0.3299, Train Acc: 0.9049 | Val Loss: 0.7042, Val Acc: 0.8913
  Epoch 7/20 -> Train Loss: 0.2015, Train Acc: 0.9266 | Val Loss: 0.9334, Val Acc: 0.9022
  Epoch 8/20 -> Train Loss: 0.2345, Train Acc: 0.9239 | Val Loss: 0.7072, Val Acc: 0.9130
  Epoch 9/20 -> Train Loss: 0.1894, Train Acc: 0.9484 | Val Loss: 0.2030, Val Acc: 0.9348
  ✨ New best model saved with loss: 0.2030, accuracy: 0.9348
  Epoch 10/20 -> Train Loss: 0.1652, Train Acc: 0.9321 | Val Loss: 0.2155, Val Acc: 0.9565
  Epoch 11/20 -> Train Loss: 0.0720, Train Acc: 0.9755 | Val Loss: 0.3782, Val Acc: 0.9348
  Epoch 12/20 -> Train Loss: 0.0807, Train Acc: 0.9701 | Val Loss: 0.2814, Val Acc: 0.9348
  Epoch 13/20 -> Train Loss: 0.0621, Train Acc: 0.9810 | Val Loss: 0.2485, Val Acc: 0.9457
  Epoch 14/20 -> Train Loss: 0.0695, Train Acc: 0.9810 | Val Loss: 0.2725, Val Acc: 0.9674
  Epoch 15/20 -> Train Loss: 0.0580, Train Acc: 0.9755 | Val Loss: 0.2305, Val Acc: 0.9674
  Epoch 16/20 -> Train Loss: 0.0631, Train Acc: 0.9864 | Val Loss: 0.2685, Val Acc: 0.9565
  Epoch 17/20 -> Train Loss: 0.0734, Train Acc: 0.9701 | Val Loss: 0.2604, Val Acc: 0.9565
  Epoch 18/20 -> Train Loss: 0.0881, Train Acc: 0.9701 | Val Loss: 0.2829, Val Acc: 0.9674
  Epoch 19/20 -> Train Loss: 0.0675, Train Acc: 0.9755 | Val Loss: 0.2848, Val Acc: 0.9674
  Epoch 20/20 -> Train Loss: 0.0671, Train Acc: 0.9837 | Val Loss: 0.2777, Val Acc: 0.9674

  Fold 4 finished. Best validation loss: 0.2030, accuracy: 0.9348

  ========== Fold 5/5 ==========
  Class Weights for Fold 5: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.3143, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.5959, Train Acc: 0.4076 | Val Loss: 6042.5123, Val Acc: 0.1522
  ✨ New best model saved with loss: 6042.5123, accuracy: 0.1522
  Epoch 2/20 -> Train Loss: 1.1115, Train Acc: 0.7255 | Val Loss: 1.3653, Val Acc: 0.7065
  ✨ New best model saved with loss: 1.3653, accuracy: 0.7065
  Epoch 3/20 -> Train Loss: 0.8327, Train Acc: 0.7772 | Val Loss: 1.7396, Val Acc: 0.7717
  Epoch 4/20 -> Train Loss: 0.6501, Train Acc: 0.8370 | Val Loss: 1.0289, Val Acc: 0.8370
  ✨ New best model saved with loss: 1.0289, accuracy: 0.8370
  Epoch 5/20 -> Train Loss: 0.4153, Train Acc: 0.8668 | Val Loss: 1.5086, Val Acc: 0.8478
  Epoch 6/20 -> Train Loss: 0.3020, Train Acc: 0.9022 | Val Loss: 1.2343, Val Acc: 0.8370
  Epoch 7/20 -> Train Loss: 0.2200, Train Acc: 0.9375 | Val Loss: 1.0858, Val Acc: 0.8913
  Epoch 8/20 -> Train Loss: 0.2671, Train Acc: 0.9239 | Val Loss: 0.8827, Val Acc: 0.8696
  ✨ New best model saved with loss: 0.8827, accuracy: 0.8696
  Epoch 9/20 -> Train Loss: 0.1858, Train Acc: 0.9429 | Val Loss: 0.5201, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.5201, accuracy: 0.9239
  Epoch 10/20 -> Train Loss: 0.2247, Train Acc: 0.9348 | Val Loss: 0.7111, Val Acc: 0.8696
  Epoch 11/20 -> Train Loss: 0.1134, Train Acc: 0.9348 | Val Loss: 0.7991, Val Acc: 0.8913
  Epoch 12/20 -> Train Loss: 0.0888, Train Acc: 0.9728 | Val Loss: 0.8604, Val Acc: 0.8804
  Epoch 13/20 -> Train Loss: 0.1061, Train Acc: 0.9592 | Val Loss: 0.7462, Val Acc: 0.9022
  Epoch 14/20 -> Train Loss: 0.0542, Train Acc: 0.9837 | Val Loss: 0.7010, Val Acc: 0.9130
  Epoch 15/20 -> Train Loss: 0.0773, Train Acc: 0.9810 | Val Loss: 0.5346, Val Acc: 0.9348
  Epoch 16/20 -> Train Loss: 0.0364, Train Acc: 0.9918 | Val Loss: 0.4950, Val Acc: 0.9348
  ✨ New best model saved with loss: 0.4950, accuracy: 0.9348
  Epoch 17/20 -> Train Loss: 0.0431, Train Acc: 0.9864 | Val Loss: 0.5364, Val Acc: 0.9348
  Epoch 18/20 -> Train Loss: 0.0386, Train Acc: 0.9891 | Val Loss: 0.5025, Val Acc: 0.9348

  ===== Evaluating on the final hold-out test set =====

  Final Test Set Accuracy (Ensemble): 0.9478

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.86      0.90      0.88        20
           Cow       1.00      1.00      1.00        15
           Dog       1.00      0.95      0.97        20
        Donkey       1.00      1.00      1.00         5
          Frog       1.00      1.00      1.00         7
          Lion       1.00      1.00      1.00         9
        Maymun       0.67      0.40      0.50         5
         Sheep       0.89      1.00      0.94         8
         Tavuk       0.86      1.00      0.92         6

      accuracy                           0.95       115
     macro avg       0.93      0.93      0.92       115
  weighted avg       0.95      0.95      0.94       115
  """
  log_text_harder = """
  ========== Fold 1/5 ==========
  Class Weights for Fold 1: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.2690, 2.3000, 1.4720,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.6546, Train Acc: 0.3614 | Val Loss: 76.6831, Val Acc: 0.1522
  Epoch 2/20 -> Train Loss: 1.3905, Train Acc: 0.6576 | Val Loss: 1.0761, Val Acc: 0.7391
  Epoch 3/20 -> Train Loss: 0.6740, Train Acc: 0.7935 | Val Loss: 0.8836, Val Acc: 0.8804
  Epoch 4/20 -> Train Loss: 0.6586, Train Acc: 0.8397 | Val Loss: 0.5449, Val Acc: 0.8370
  Epoch 5/20 -> Train Loss: 0.5126, Train Acc: 0.8505 | Val Loss: 0.3821, Val Acc: 0.9239
  Epoch 6/20 -> Train Loss: 0.2967, Train Acc: 0.8859 | Val Loss: 0.3258, Val Acc: 0.9239
  Epoch 7/20 -> Train Loss: 0.2080, Train Acc: 0.9321 | Val Loss: 0.3249, Val Acc: 0.9239
  Epoch 8/20 -> Train Loss: 0.2821, Train Acc: 0.9348 | Val Loss: 0.3631, Val Acc: 0.9130
  Epoch 9/20 -> Train Loss: 0.2573, Train Acc: 0.9158 | Val Loss: 0.2908, Val Acc: 0.8804
  Epoch 10/20 -> Train Loss: 0.1759, Train Acc: 0.9484 | Val Loss: 0.3265, Val Acc: 0.9022
  Epoch 11/20 -> Train Loss: 0.1339, Train Acc: 0.9565 | Val Loss: 0.2276, Val Acc: 0.9130
  Epoch 12/20 -> Train Loss: 0.1288, Train Acc: 0.9620 | Val Loss: 0.3524, Val Acc: 0.9239
  Epoch 13/20 -> Train Loss: 0.0985, Train Acc: 0.9728 | Val Loss: 0.3056, Val Acc: 0.9348
  Epoch 14/20 -> Train Loss: 0.0543, Train Acc: 0.9728 | Val Loss: 0.2964, Val Acc: 0.9239
  Epoch 15/20 -> Train Loss: 0.1017, Train Acc: 0.9620 | Val Loss: 0.2946, Val Acc: 0.9130
  Epoch 16/20 -> Train Loss: 0.0545, Train Acc: 0.9755 | Val Loss: 0.3106, Val Acc: 0.9239
  Epoch 17/20 -> Train Loss: 0.0484, Train Acc: 0.9837 | Val Loss: 0.3248, Val Acc: 0.9239
  Epoch 18/20 -> Train Loss: 0.1051, Train Acc: 0.9647 | Val Loss: 0.3026, Val Acc: 0.9239
  Epoch 19/20 -> Train Loss: 0.0648, Train Acc: 0.9783 | Val Loss: 0.2914, Val Acc: 0.9239
  Epoch 20/20 -> Train Loss: 0.0481, Train Acc: 0.9918 | Val Loss: 0.2885, Val Acc: 0.9239

  Fold 1 finished. Best validation loss: 0.2885, accuracy: 0.9239

  ========== Fold 2/5 ==========
  Class Weights for Fold 2: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4720,
          1.8400], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.5114, Train Acc: 0.4076 | Val Loss: 4.3966, Val Acc: 0.2717
  Epoch 2/20 -> Train Loss: 1.3182, Train Acc: 0.6549 | Val Loss: 2.0315, Val Acc: 0.6196
  Epoch 3/20 -> Train Loss: 0.6432, Train Acc: 0.7663 | Val Loss: 1.0965, Val Acc: 0.7283
  Epoch 4/20 -> Train Loss: 0.5177, Train Acc: 0.8641 | Val Loss: 0.6798, Val Acc: 0.8043
  Epoch 5/20 -> Train Loss: 0.3543, Train Acc: 0.9022 | Val Loss: 0.4141, Val Acc: 0.8587
  Epoch 6/20 -> Train Loss: 0.4701, Train Acc: 0.9022 | Val Loss: 0.4592, Val Acc: 0.8478
  Epoch 7/20 -> Train Loss: 0.3544, Train Acc: 0.8777 | Val Loss: 0.5764, Val Acc: 0.8804
  Epoch 8/20 -> Train Loss: 0.2716, Train Acc: 0.9266 | Val Loss: 0.6403, Val Acc: 0.8370
  Epoch 9/20 -> Train Loss: 0.3860, Train Acc: 0.9130 | Val Loss: 0.6310, Val Acc: 0.8370
  Epoch 10/20 -> Train Loss: 0.1876, Train Acc: 0.9158 | Val Loss: 0.6370, Val Acc: 0.8804
  Epoch 11/20 -> Train Loss: 0.2042, Train Acc: 0.9402 | Val Loss: 0.8895, Val Acc: 0.9130
  Epoch 12/20 -> Train Loss: 0.1587, Train Acc: 0.9429 | Val Loss: 0.9946, Val Acc: 0.8261
  Epoch 13/20 -> Train Loss: 0.1443, Train Acc: 0.9429 | Val Loss: 0.4160, Val Acc: 0.8913
  Epoch 14/20 -> Train Loss: 0.2051, Train Acc: 0.9457 | Val Loss: 0.3589, Val Acc: 0.8804
  Epoch 15/20 -> Train Loss: 0.1732, Train Acc: 0.9402 | Val Loss: 0.7066, Val Acc: 0.8804
  Epoch 16/20 -> Train Loss: 0.1186, Train Acc: 0.9511 | Val Loss: 0.5537, Val Acc: 0.8696
  Epoch 17/20 -> Train Loss: 0.0754, Train Acc: 0.9674 | Val Loss: 0.4715, Val Acc: 0.8913
  Epoch 18/20 -> Train Loss: 0.1074, Train Acc: 0.9755 | Val Loss: 0.4522, Val Acc: 0.8913
  Epoch 19/20 -> Train Loss: 0.0532, Train Acc: 0.9755 | Val Loss: 0.4473, Val Acc: 0.8804
  Epoch 20/20 -> Train Loss: 0.1572, Train Acc: 0.9728 | Val Loss: 0.4566, Val Acc: 0.8804

  Fold 2 finished. Best validation loss: 0.4566, accuracy: 0.8804

  ========== Fold 3/5 ==========
  Class Weights for Fold 3: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.4935, Train Acc: 0.3913 | Val Loss: 2.4977, Val Acc: 0.2609
  Epoch 2/20 -> Train Loss: 1.0603, Train Acc: 0.6984 | Val Loss: 1.5842, Val Acc: 0.6087
  Epoch 3/20 -> Train Loss: 0.8555, Train Acc: 0.7908 | Val Loss: 0.9396, Val Acc: 0.7717
  Epoch 4/20 -> Train Loss: 0.7022, Train Acc: 0.8016 | Val Loss: 0.6286, Val Acc: 0.8043
  Epoch 5/20 -> Train Loss: 0.4439, Train Acc: 0.8832 | Val Loss: 0.6163, Val Acc: 0.8804
  Epoch 6/20 -> Train Loss: 0.3214, Train Acc: 0.9158 | Val Loss: 0.6135, Val Acc: 0.8370
  Epoch 7/20 -> Train Loss: 0.2487, Train Acc: 0.9293 | Val Loss: 0.5874, Val Acc: 0.8261
  Epoch 8/20 -> Train Loss: 0.3239, Train Acc: 0.8913 | Val Loss: 0.4984, Val Acc: 0.8478
  Epoch 9/20 -> Train Loss: 0.1976, Train Acc: 0.9375 | Val Loss: 0.4717, Val Acc: 0.8478
  Epoch 10/20 -> Train Loss: 0.1896, Train Acc: 0.9402 | Val Loss: 0.5092, Val Acc: 0.9022
  Epoch 11/20 -> Train Loss: 0.1083, Train Acc: 0.9620 | Val Loss: 0.6074, Val Acc: 0.9130
  Epoch 12/20 -> Train Loss: 0.1843, Train Acc: 0.9511 | Val Loss: 0.5069, Val Acc: 0.8913
  Epoch 13/20 -> Train Loss: 0.1230, Train Acc: 0.9647 | Val Loss: 0.4934, Val Acc: 0.8913
  Epoch 14/20 -> Train Loss: 0.0765, Train Acc: 0.9755 | Val Loss: 0.5084, Val Acc: 0.8587
  Epoch 15/20 -> Train Loss: 0.0936, Train Acc: 0.9538 | Val Loss: 0.5124, Val Acc: 0.8804
  Epoch 16/20 -> Train Loss: 0.1434, Train Acc: 0.9565 | Val Loss: 0.5031, Val Acc: 0.9022
  Epoch 17/20 -> Train Loss: 0.0660, Train Acc: 0.9755 | Val Loss: 0.4579, Val Acc: 0.9022
  Epoch 18/20 -> Train Loss: 0.0799, Train Acc: 0.9810 | Val Loss: 0.4523, Val Acc: 0.8913
  Epoch 19/20 -> Train Loss: 0.1010, Train Acc: 0.9647 | Val Loss: 0.4949, Val Acc: 0.8913
  Epoch 20/20 -> Train Loss: 0.0543, Train Acc: 0.9810 | Val Loss: 0.4855, Val Acc: 0.9130

  Fold 3 finished. Best validation loss: 0.4855, accuracy: 0.9130

  ========== Fold 4/5 ==========
  Class Weights for Fold 4: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.2399, Train Acc: 0.4022 | Val Loss: 3.9872, Val Acc: 0.3370
  Epoch 2/20 -> Train Loss: 1.0881, Train Acc: 0.7092 | Val Loss: 1.3496, Val Acc: 0.6848
  Epoch 3/20 -> Train Loss: 0.8301, Train Acc: 0.7717 | Val Loss: 1.2783, Val Acc: 0.8152
  Epoch 4/20 -> Train Loss: 0.8228, Train Acc: 0.7935 | Val Loss: 1.5261, Val Acc: 0.8043
  Epoch 5/20 -> Train Loss: 0.4961, Train Acc: 0.8397 | Val Loss: 0.8699, Val Acc: 0.8261
  Epoch 6/20 -> Train Loss: 0.3614, Train Acc: 0.8804 | Val Loss: 0.4402, Val Acc: 0.8804
  Epoch 7/20 -> Train Loss: 0.2669, Train Acc: 0.9130 | Val Loss: 0.5175, Val Acc: 0.8804
  Epoch 8/20 -> Train Loss: 0.2467, Train Acc: 0.9185 | Val Loss: 0.4030, Val Acc: 0.9022
  Epoch 9/20 -> Train Loss: 0.2207, Train Acc: 0.9212 | Val Loss: 0.6062, Val Acc: 0.8804
  Epoch 10/20 -> Train Loss: 0.3093, Train Acc: 0.9321 | Val Loss: 0.5274, Val Acc: 0.8913
  Epoch 11/20 -> Train Loss: 0.2361, Train Acc: 0.9212 | Val Loss: 0.4018, Val Acc: 0.8804
  Epoch 12/20 -> Train Loss: 0.1413, Train Acc: 0.9565 | Val Loss: 0.5596, Val Acc: 0.8804
  Epoch 13/20 -> Train Loss: 0.1426, Train Acc: 0.9592 | Val Loss: 0.3643, Val Acc: 0.8804
  Epoch 14/20 -> Train Loss: 0.1377, Train Acc: 0.9647 | Val Loss: 0.3975, Val Acc: 0.8696
  Epoch 15/20 -> Train Loss: 0.0623, Train Acc: 0.9783 | Val Loss: 0.3571, Val Acc: 0.8913
  Epoch 16/20 -> Train Loss: 0.0580, Train Acc: 0.9755 | Val Loss: 0.3929, Val Acc: 0.9022
  Epoch 17/20 -> Train Loss: 0.0641, Train Acc: 0.9728 | Val Loss: 0.3656, Val Acc: 0.9022
  Epoch 18/20 -> Train Loss: 0.0590, Train Acc: 0.9864 | Val Loss: 0.3581, Val Acc: 0.9022
  Epoch 19/20 -> Train Loss: 0.0819, Train Acc: 0.9674 | Val Loss: 0.3821, Val Acc: 0.9130
  Epoch 20/20 -> Train Loss: 0.0655, Train Acc: 0.9783 | Val Loss: 0.3941, Val Acc: 0.9022

  Fold 4 finished. Best validation loss: 0.3821, accuracy: 0.9130

  ========== Fold 5/5 ==========
  Class Weights for Fold 5: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.3143, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.7504, Train Acc: 0.3723 | Val Loss: 65.5953, Val Acc: 0.1304
  Epoch 2/20 -> Train Loss: 1.3579, Train Acc: 0.6413 | Val Loss: 1.7366, Val Acc: 0.4783
  Epoch 3/20 -> Train Loss: 0.8728, Train Acc: 0.7609 | Val Loss: 0.5132, Val Acc: 0.8696
  Epoch 4/20 -> Train Loss: 0.5445, Train Acc: 0.8234 | Val Loss: 0.5198, Val Acc: 0.8804
  Epoch 5/20 -> Train Loss: 0.3655, Train Acc: 0.8832 | Val Loss: 0.6431, Val Acc: 0.8804
  Epoch 6/20 -> Train Loss: 0.3779, Train Acc: 0.8859 | Val Loss: 0.6282, Val Acc: 0.8913
  Epoch 7/20 -> Train Loss: 0.2687, Train Acc: 0.9212 | Val Loss: 0.1654, Val Acc: 0.9674
  Epoch 8/20 -> Train Loss: 0.2625, Train Acc: 0.9049 | Val Loss: 0.2733, Val Acc: 0.9348
  Epoch 9/20 -> Train Loss: 0.3077, Train Acc: 0.9130 | Val Loss: 0.3774, Val Acc: 0.8804
  Epoch 10/20 -> Train Loss: 0.2602, Train Acc: 0.9022 | Val Loss: 0.3731, Val Acc: 0.9239
  Epoch 11/20 -> Train Loss: 0.1719, Train Acc: 0.9484 | Val Loss: 0.2975, Val Acc: 0.9239
  Epoch 12/20 -> Train Loss: 0.1383, Train Acc: 0.9620 | Val Loss: 0.2212, Val Acc: 0.9565
  Epoch 13/20 -> Train Loss: 0.1908, Train Acc: 0.9484 | Val Loss: 0.2920, Val Acc: 0.9130
  Epoch 14/20 -> Train Loss: 0.1239, Train Acc: 0.9565 | Val Loss: 0.4050, Val Acc: 0.9130
  Epoch 15/20 -> Train Loss: 0.0801, Train Acc: 0.9674 | Val Loss: 0.3725, Val Acc: 0.9239
  Epoch 16/20 -> Train Loss: 0.0703, Train Acc: 0.9755 | Val Loss: 0.3222, Val Acc: 0.9457
  Epoch 17/20 -> Train Loss: 0.0980, Train Acc: 0.9620 | Val Loss: 0.2982, Val Acc: 0.9674
  Epoch 18/20 -> Train Loss: 0.0601, Train Acc: 0.9755 | Val Loss: 0.3131, Val Acc: 0.9565
  Epoch 19/20 -> Train Loss: 0.1269, Train Acc: 0.9592 | Val Loss: 0.2999, Val Acc: 0.9457
  Epoch 20/20 -> Train Loss: 0.0639, Train Acc: 0.9728 | Val Loss: 0.2960, Val Acc: 0.9565

  Fold 5 finished. Best validation loss: 0.2960, accuracy: 0.9565

  ===== Evaluating on the final hold-out test set =====

  Final Test Set Accuracy (Ensemble): 0.9391

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.86      0.90      0.88        20
           Cow       1.00      1.00      1.00        15
           Dog       0.95      0.95      0.95        20
        Donkey       1.00      1.00      1.00         5
          Frog       1.00      0.86      0.92         7
          Lion       1.00      1.00      1.00         9
        Maymun       0.67      0.40      0.50         5
         Sheep       0.89      1.00      0.94         8
         Tavuk       0.86      1.00      0.92         6

      accuracy                           0.94       115
     macro avg       0.92      0.91      0.91       115
  weighted avg       0.94      0.94      0.94       115
  """
  log_text_lighter = """
  ========== Fold 1/5 ==========
  Class Weights for Fold 1: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.2690, 2.3000, 1.4720,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.8001, Train Acc: 0.3723 | Val Loss: 254.2116, Val Acc: 0.1087
  ✨ New best model saved with loss: 254.2116, acc: 0.1087
  Epoch 2/20 -> Train Loss: 1.0475, Train Acc: 0.7011 | Val Loss: 1.2219, Val Acc: 0.6957
  ✨ New best model saved with loss: 1.2219, acc: 0.6957
  Epoch 3/20 -> Train Loss: 0.8960, Train Acc: 0.7527 | Val Loss: 0.7807, Val Acc: 0.8587
  ✨ New best model saved with loss: 0.7807, acc: 0.8587
  Epoch 4/20 -> Train Loss: 0.5473, Train Acc: 0.8370 | Val Loss: 0.8224, Val Acc: 0.8478
  EarlyStopping counter: 1/10
  Epoch 5/20 -> Train Loss: 0.4203, Train Acc: 0.8804 | Val Loss: 0.4463, Val Acc: 0.8696
  ✨ New best model saved with loss: 0.4463, acc: 0.8696
  Epoch 6/20 -> Train Loss: 0.3045, Train Acc: 0.8777 | Val Loss: 0.5498, Val Acc: 0.8913
  EarlyStopping counter: 1/10
  Epoch 7/20 -> Train Loss: 0.1658, Train Acc: 0.9620 | Val Loss: 0.2654, Val Acc: 0.9348
  ✨ New best model saved with loss: 0.2654, acc: 0.9348
  Epoch 8/20 -> Train Loss: 0.2041, Train Acc: 0.9266 | Val Loss: 0.3297, Val Acc: 0.9022
  EarlyStopping counter: 1/10
  Epoch 9/20 -> Train Loss: 0.2442, Train Acc: 0.9620 | Val Loss: 0.3907, Val Acc: 0.8913
  EarlyStopping counter: 2/10
  Epoch 10/20 -> Train Loss: 0.2045, Train Acc: 0.9511 | Val Loss: 0.4240, Val Acc: 0.8804
  EarlyStopping counter: 3/10
  Epoch 11/20 -> Train Loss: 0.1547, Train Acc: 0.9592 | Val Loss: 0.4156, Val Acc: 0.8804
  EarlyStopping counter: 4/10
  Epoch 12/20 -> Train Loss: 0.1139, Train Acc: 0.9538 | Val Loss: 0.3001, Val Acc: 0.8913
  EarlyStopping counter: 5/10
  Epoch 13/20 -> Train Loss: 0.1198, Train Acc: 0.9592 | Val Loss: 0.1951, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.1951, acc: 0.9239
  Epoch 14/20 -> Train Loss: 0.0742, Train Acc: 0.9755 | Val Loss: 0.1890, Val Acc: 0.9348
  ✨ New best model saved with loss: 0.1890, acc: 0.9348
  Epoch 15/20 -> Train Loss: 0.0871, Train Acc: 0.9701 | Val Loss: 0.1950, Val Acc: 0.9239
  EarlyStopping counter: 1/10
  Epoch 16/20 -> Train Loss: 0.0812, Train Acc: 0.9728 | Val Loss: 0.2086, Val Acc: 0.9239
  EarlyStopping counter: 2/10
  Epoch 17/20 -> Train Loss: 0.0429, Train Acc: 0.9891 | Val Loss: 0.2004, Val Acc: 0.9239
  EarlyStopping counter: 3/10
  Epoch 18/20 -> Train Loss: 0.0312, Train Acc: 0.9891 | Val Loss: 0.1930, Val Acc: 0.9239
  EarlyStopping counter: 4/10
  Epoch 19/20 -> Train Loss: 0.0470, Train Acc: 0.9783 | Val Loss: 0.1946, Val Acc: 0.9348
  EarlyStopping counter: 5/10
  Epoch 20/20 -> Train Loss: 0.0870, Train Acc: 0.9755 | Val Loss: 0.1982, Val Acc: 0.9348
  EarlyStopping counter: 6/10

  Fold 1 finished. Best validation loss: 0.1890, accuracy: 0.9348

  ========== Fold 2/5 ==========
  Class Weights for Fold 2: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4720,
          1.8400], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.3655, Train Acc: 0.4348 | Val Loss: 555.7816, Val Acc: 0.1196
  ✨ New best model saved with loss: 555.7816, acc: 0.1196
  Epoch 2/20 -> Train Loss: 1.0493, Train Acc: 0.7120 | Val Loss: 1.1417, Val Acc: 0.7174
  ✨ New best model saved with loss: 1.1417, acc: 0.7174
  Epoch 3/20 -> Train Loss: 0.5290, Train Acc: 0.8370 | Val Loss: 1.3633, Val Acc: 0.7391
  EarlyStopping counter: 1/10
  Epoch 4/20 -> Train Loss: 0.5405, Train Acc: 0.8723 | Val Loss: 1.0362, Val Acc: 0.8043
  ✨ New best model saved with loss: 1.0362, acc: 0.8043
  Epoch 5/20 -> Train Loss: 0.4904, Train Acc: 0.8750 | Val Loss: 0.6408, Val Acc: 0.8370
  ✨ New best model saved with loss: 0.6408, acc: 0.8370
  Epoch 6/20 -> Train Loss: 0.3637, Train Acc: 0.8940 | Val Loss: 0.5260, Val Acc: 0.8804
  ✨ New best model saved with loss: 0.5260, acc: 0.8804
  Epoch 7/20 -> Train Loss: 0.2571, Train Acc: 0.9321 | Val Loss: 0.4025, Val Acc: 0.8696
  ✨ New best model saved with loss: 0.4025, acc: 0.8696
  Epoch 8/20 -> Train Loss: 0.1431, Train Acc: 0.9511 | Val Loss: 0.4252, Val Acc: 0.8913
  EarlyStopping counter: 1/10
  Epoch 9/20 -> Train Loss: 0.1130, Train Acc: 0.9728 | Val Loss: 0.8209, Val Acc: 0.8587
  EarlyStopping counter: 2/10
  Epoch 10/20 -> Train Loss: 0.2034, Train Acc: 0.9402 | Val Loss: 0.5756, Val Acc: 0.9022
  EarlyStopping counter: 3/10
  Epoch 11/20 -> Train Loss: 0.0921, Train Acc: 0.9674 | Val Loss: 0.6022, Val Acc: 0.8913
  EarlyStopping counter: 4/10
  Epoch 12/20 -> Train Loss: 0.1671, Train Acc: 0.9620 | Val Loss: 0.5278, Val Acc: 0.8913
  EarlyStopping counter: 5/10
  Epoch 13/20 -> Train Loss: 0.1170, Train Acc: 0.9728 | Val Loss: 0.4793, Val Acc: 0.9022
  EarlyStopping counter: 6/10
  Epoch 14/20 -> Train Loss: 0.0541, Train Acc: 0.9810 | Val Loss: 0.5304, Val Acc: 0.8913
  EarlyStopping counter: 7/10
  Epoch 15/20 -> Train Loss: 0.0824, Train Acc: 0.9728 | Val Loss: 0.4358, Val Acc: 0.9130
  EarlyStopping counter: 8/10
  Epoch 16/20 -> Train Loss: 0.0600, Train Acc: 0.9783 | Val Loss: 0.4026, Val Acc: 0.9239
  EarlyStopping counter: 9/10
  Epoch 17/20 -> Train Loss: 0.0340, Train Acc: 0.9837 | Val Loss: 0.3809, Val Acc: 0.9348
  ✨ New best model saved with loss: 0.3809, acc: 0.9348
  Epoch 18/20 -> Train Loss: 0.0232, Train Acc: 0.9946 | Val Loss: 0.3593, Val Acc: 0.9348
  ✨ New best model saved with loss: 0.3593, acc: 0.9348
  Epoch 19/20 -> Train Loss: 0.0389, Train Acc: 0.9891 | Val Loss: 0.3709, Val Acc: 0.9348
  EarlyStopping counter: 1/10
  Epoch 20/20 -> Train Loss: 0.0521, Train Acc: 0.9837 | Val Loss: 0.3806, Val Acc: 0.9348
  EarlyStopping counter: 2/10

  Fold 2 finished. Best validation loss: 0.3593, accuracy: 0.9348

  ========== Fold 3/5 ==========
  Class Weights for Fold 3: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.5009, Train Acc: 0.4266 | Val Loss: 110.9266, Val Acc: 0.0978
  ✨ New best model saved with loss: 110.9266, acc: 0.0978
  Epoch 2/20 -> Train Loss: 1.4291, Train Acc: 0.6712 | Val Loss: 1.6223, Val Acc: 0.7065
  ✨ New best model saved with loss: 1.6223, acc: 0.7065
  Epoch 3/20 -> Train Loss: 1.0657, Train Acc: 0.7582 | Val Loss: 0.8942, Val Acc: 0.7174
  ✨ New best model saved with loss: 0.8942, acc: 0.7174
  Epoch 4/20 -> Train Loss: 0.7277, Train Acc: 0.7962 | Val Loss: 0.8646, Val Acc: 0.8261
  ✨ New best model saved with loss: 0.8646, acc: 0.8261
  Epoch 5/20 -> Train Loss: 0.4334, Train Acc: 0.8478 | Val Loss: 0.6532, Val Acc: 0.8152
  ✨ New best model saved with loss: 0.6532, acc: 0.8152
  Epoch 6/20 -> Train Loss: 0.3843, Train Acc: 0.9076 | Val Loss: 0.6138, Val Acc: 0.8696
  ✨ New best model saved with loss: 0.6138, acc: 0.8696
  Epoch 7/20 -> Train Loss: 0.2931, Train Acc: 0.8886 | Val Loss: 0.5973, Val Acc: 0.8478
  ✨ New best model saved with loss: 0.5973, acc: 0.8478
  Epoch 8/20 -> Train Loss: 0.2211, Train Acc: 0.9293 | Val Loss: 0.3628, Val Acc: 0.8804
  ✨ New best model saved with loss: 0.3628, acc: 0.8804
  Epoch 9/20 -> Train Loss: 0.2179, Train Acc: 0.9321 | Val Loss: 0.3610, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.3610, acc: 0.8913
  Epoch 10/20 -> Train Loss: 0.1060, Train Acc: 0.9674 | Val Loss: 0.2822, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.2822, acc: 0.8913
  Epoch 11/20 -> Train Loss: 0.1398, Train Acc: 0.9592 | Val Loss: 0.1893, Val Acc: 0.9239
  ✨ New best model saved with loss: 0.1893, acc: 0.9239
  Epoch 12/20 -> Train Loss: 0.0863, Train Acc: 0.9674 | Val Loss: 0.1347, Val Acc: 0.9457
  ✨ New best model saved with loss: 0.1347, acc: 0.9457
  Epoch 13/20 -> Train Loss: 0.0574, Train Acc: 0.9783 | Val Loss: 0.1647, Val Acc: 0.9239
  EarlyStopping counter: 1/10
  Epoch 14/20 -> Train Loss: 0.0742, Train Acc: 0.9783 | Val Loss: 0.1704, Val Acc: 0.9348
  EarlyStopping counter: 2/10
  Epoch 15/20 -> Train Loss: 0.0396, Train Acc: 0.9837 | Val Loss: 0.1964, Val Acc: 0.9239
  EarlyStopping counter: 3/10
  Epoch 16/20 -> Train Loss: 0.0808, Train Acc: 0.9755 | Val Loss: 0.1745, Val Acc: 0.9239
  EarlyStopping counter: 4/10
  Epoch 17/20 -> Train Loss: 0.0527, Train Acc: 0.9837 | Val Loss: 0.1767, Val Acc: 0.9348
  EarlyStopping counter: 5/10
  Epoch 18/20 -> Train Loss: 0.0787, Train Acc: 0.9810 | Val Loss: 0.1757, Val Acc: 0.9239
  EarlyStopping counter: 6/10
  Epoch 19/20 -> Train Loss: 0.0544, Train Acc: 0.9783 | Val Loss: 0.1522, Val Acc: 0.9239
  EarlyStopping counter: 7/10
  Epoch 20/20 -> Train Loss: 0.0427, Train Acc: 0.9755 | Val Loss: 0.1700, Val Acc: 0.9239
  EarlyStopping counter: 8/10

  Fold 3 finished. Best validation loss: 0.1347, accuracy: 0.9457

  ========== Fold 4/5 ==========
  Class Weights for Fold 4: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.4497, Train Acc: 0.4103 | Val Loss: 81.7385, Val Acc: 0.1196
  ✨ New best model saved with loss: 81.7385, acc: 0.1196
  Epoch 2/20 -> Train Loss: 1.2055, Train Acc: 0.7038 | Val Loss: 1.2208, Val Acc: 0.7500
  ✨ New best model saved with loss: 1.2208, acc: 0.7500
  Epoch 3/20 -> Train Loss: 0.8753, Train Acc: 0.7745 | Val Loss: 0.8197, Val Acc: 0.7935
  ✨ New best model saved with loss: 0.8197, acc: 0.7935
  Epoch 4/20 -> Train Loss: 0.4464, Train Acc: 0.8614 | Val Loss: 0.6715, Val Acc: 0.8152
  ✨ New best model saved with loss: 0.6715, acc: 0.8152
  Epoch 5/20 -> Train Loss: 0.2161, Train Acc: 0.9103 | Val Loss: 0.6560, Val Acc: 0.8804
  ✨ New best model saved with loss: 0.6560, acc: 0.8804
  Epoch 6/20 -> Train Loss: 0.2324, Train Acc: 0.9266 | Val Loss: 0.8592, Val Acc: 0.8696
  EarlyStopping counter: 1/10
  Epoch 7/20 -> Train Loss: 0.2689, Train Acc: 0.9185 | Val Loss: 0.6415, Val Acc: 0.8696
  ✨ New best model saved with loss: 0.6415, acc: 0.8696
  Epoch 8/20 -> Train Loss: 0.2951, Train Acc: 0.9239 | Val Loss: 0.7903, Val Acc: 0.8804
  EarlyStopping counter: 1/10
  Epoch 9/20 -> Train Loss: 0.1402, Train Acc: 0.9375 | Val Loss: 0.6075, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.6075, acc: 0.8913
  Epoch 10/20 -> Train Loss: 0.1287, Train Acc: 0.9592 | Val Loss: 0.5206, Val Acc: 0.8913
  ✨ New best model saved with loss: 0.5206, acc: 0.8913
  Epoch 11/20 -> Train Loss: 0.1031, Train Acc: 0.9674 | Val Loss: 0.6074, Val Acc: 0.9239
  EarlyStopping counter: 1/10
  Epoch 12/20 -> Train Loss: 0.1178, Train Acc: 0.9674 | Val Loss: 0.5848, Val Acc: 0.8804
  EarlyStopping counter: 2/10
  Epoch 13/20 -> Train Loss: 0.0414, Train Acc: 0.9891 | Val Loss: 0.6591, Val Acc: 0.8913
  EarlyStopping counter: 3/10
  Epoch 14/20 -> Train Loss: 0.0426, Train Acc: 0.9891 | Val Loss: 0.6356, Val Acc: 0.8913
  EarlyStopping counter: 4/10
  Epoch 15/20 -> Train Loss: 0.0543, Train Acc: 0.9891 | Val Loss: 0.6353, Val Acc: 0.8913
  EarlyStopping counter: 5/10
  Epoch 16/20 -> Train Loss: 0.0378, Train Acc: 0.9837 | Val Loss: 0.6350, Val Acc: 0.8913
  EarlyStopping counter: 6/10
  Epoch 17/20 -> Train Loss: 0.0595, Train Acc: 0.9755 | Val Loss: 0.6019, Val Acc: 0.8913
  EarlyStopping counter: 7/10
  Epoch 18/20 -> Train Loss: 0.0528, Train Acc: 0.9728 | Val Loss: 0.6078, Val Acc: 0.8804
  EarlyStopping counter: 8/10
  Epoch 19/20 -> Train Loss: 0.0917, Train Acc: 0.9837 | Val Loss: 0.6370, Val Acc: 0.8913
  EarlyStopping counter: 9/10
  Epoch 20/20 -> Train Loss: 0.0451, Train Acc: 0.9701 | Val Loss: 0.6171, Val Acc: 0.8913
  EarlyStopping counter: 10/10
  Early stopping triggered!

  Fold 4 finished. Best validation loss: 0.5206, accuracy: 0.8913

  ========== Fold 5/5 ==========
  Class Weights for Fold 5: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.3143, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.7650, Train Acc: 0.3886 | Val Loss: 3.2560, Val Acc: 0.2826
  ✨ New best model saved with loss: 3.2560, acc: 0.2826
  Epoch 2/20 -> Train Loss: 1.0413, Train Acc: 0.7120 | Val Loss: 2.2060, Val Acc: 0.6522
  ✨ New best model saved with loss: 2.2060, acc: 0.6522
  Epoch 3/20 -> Train Loss: 0.6811, Train Acc: 0.7935 | Val Loss: 1.6280, Val Acc: 0.7500
  ✨ New best model saved with loss: 1.6280, acc: 0.7500
  Epoch 4/20 -> Train Loss: 0.4691, Train Acc: 0.8424 | Val Loss: 0.6418, Val Acc: 0.8370
  ✨ New best model saved with loss: 0.6418, acc: 0.8370
  Epoch 5/20 -> Train Loss: 0.5419, Train Acc: 0.8668 | Val Loss: 0.3893, Val Acc: 0.9022
  ✨ New best model saved with loss: 0.3893, acc: 0.9022
  Epoch 6/20 -> Train Loss: 0.4643, Train Acc: 0.8750 | Val Loss: 0.4319, Val Acc: 0.9130
  EarlyStopping counter: 1/10
  Epoch 7/20 -> Train Loss: 0.3031, Train Acc: 0.9293 | Val Loss: 0.2369, Val Acc: 0.9022
  ✨ New best model saved with loss: 0.2369, acc: 0.9022
  Epoch 8/20 -> Train Loss: 0.3310, Train Acc: 0.9130 | Val Loss: 0.3623, Val Acc: 0.9348
  EarlyStopping counter: 1/10
  Epoch 9/20 -> Train Loss: 0.2821, Train Acc: 0.9239 | Val Loss: 0.3955, Val Acc: 0.8804
  EarlyStopping counter: 2/10
  Epoch 10/20 -> Train Loss: 0.1667, Train Acc: 0.9375 | Val Loss: 0.4216, Val Acc: 0.8804
  EarlyStopping counter: 3/10
  Epoch 11/20 -> Train Loss: 0.1607, Train Acc: 0.9484 | Val Loss: 0.5168, Val Acc: 0.8696
  EarlyStopping counter: 4/10
  Epoch 12/20 -> Train Loss: 0.1538, Train Acc: 0.9321 | Val Loss: 0.6942, Val Acc: 0.8696
  EarlyStopping counter: 5/10
  Epoch 13/20 -> Train Loss: 0.1077, Train Acc: 0.9538 | Val Loss: 0.5485, Val Acc: 0.9130
  EarlyStopping counter: 6/10
  Epoch 14/20 -> Train Loss: 0.1128, Train Acc: 0.9647 | Val Loss: 0.4341, Val Acc: 0.9348
  EarlyStopping counter: 7/10
  Epoch 15/20 -> Train Loss: 0.0605, Train Acc: 0.9783 | Val Loss: 0.4081, Val Acc: 0.9130
  EarlyStopping counter: 8/10
  Epoch 16/20 -> Train Loss: 0.1002, Train Acc: 0.9674 | Val Loss: 0.4248, Val Acc: 0.9457
  EarlyStopping counter: 9/10
  Epoch 17/20 -> Train Loss: 0.0685, Train Acc: 0.9837 | Val Loss: 0.4241, Val Acc: 0.9348
  EarlyStopping counter: 10/10
  Early stopping triggered!

  Fold 5 finished. Best validation loss: 0.2369, accuracy: 0.9022

  ===== Evaluating on the final hold-out test set =====

  Final Test Set Accuracy (Ensemble): 0.9565

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.86      0.95      0.90        20
           Cow       1.00      1.00      1.00        15
           Dog       1.00      1.00      1.00        20
        Donkey       1.00      1.00      1.00         5
          Frog       1.00      0.86      0.92         7
          Lion       1.00      1.00      1.00         9
        Maymun       1.00      0.40      0.57         5
         Sheep       0.89      1.00      0.94         8
         Tavuk       0.86      1.00      0.92         6

      accuracy                           0.96       115
     macro avg       0.96      0.92      0.93       115
  weighted avg       0.96      0.96      0.95       115
  """
  log_text_final = """
  ========== Fold 1/5 ==========
  Class Weights for Fold 1: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.2690, 2.3000, 1.4720,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.0664, Train Acc: 0.3152 | Val Loss: 1.6003, Val Acc: 0.6630, Val mF1: 0.6210
  ✨ New best model saved with loss: 1.6003, acc: 0.6630
  Epoch 2/20 -> Train Loss: 1.0317, Train Acc: 0.7745 | Val Loss: 0.9160, Val Acc: 0.8478, Val mF1: 0.8059
  ✨ New best model saved with loss: 0.9160, acc: 0.8478
  Epoch 3/20 -> Train Loss: 0.7829, Train Acc: 0.8668 | Val Loss: 0.7803, Val Acc: 0.8370, Val mF1: 0.8297
  ✨ New best model saved with loss: 0.7803, acc: 0.8370
  Epoch 4/20 -> Train Loss: 0.7251, Train Acc: 0.8723 | Val Loss: 0.9547, Val Acc: 0.8696, Val mF1: 0.8329
  EarlyStopping counter: 1/10
  Epoch 5/20 -> Train Loss: 0.6759, Train Acc: 0.8913 | Val Loss: 1.0545, Val Acc: 0.7609, Val mF1: 0.6933
  EarlyStopping counter: 2/10
  Epoch 6/20 -> Train Loss: 0.6358, Train Acc: 0.9049 | Val Loss: 0.6464, Val Acc: 0.9130, Val mF1: 0.8734
  ✨ New best model saved with loss: 0.6464, acc: 0.9130
  Epoch 7/20 -> Train Loss: 0.5864, Train Acc: 0.9348 | Val Loss: 0.6338, Val Acc: 0.9130, Val mF1: 0.9062
  ✨ New best model saved with loss: 0.6338, acc: 0.9130
  Epoch 8/20 -> Train Loss: 0.5822, Train Acc: 0.9402 | Val Loss: 0.6783, Val Acc: 0.9022, Val mF1: 0.8771
  EarlyStopping counter: 1/10
  Epoch 9/20 -> Train Loss: 0.6145, Train Acc: 0.9158 | Val Loss: 0.6069, Val Acc: 0.9130, Val mF1: 0.9000
  ✨ New best model saved with loss: 0.6069, acc: 0.9130
  Epoch 10/20 -> Train Loss: 0.5543, Train Acc: 0.9158 | Val Loss: 0.6148, Val Acc: 0.8913, Val mF1: 0.8810
  EarlyStopping counter: 1/10
  Epoch 11/20 -> Train Loss: 0.5624, Train Acc: 0.9402 | Val Loss: 0.6841, Val Acc: 0.8478, Val mF1: 0.8643
  EarlyStopping counter: 2/10
  Epoch 12/20 -> Train Loss: 0.5703, Train Acc: 0.9348 | Val Loss: 0.5050, Val Acc: 0.9565, Val mF1: 0.9549
  ✨ New best model saved with loss: 0.5050, acc: 0.9565
  Epoch 13/20 -> Train Loss: 0.5451, Train Acc: 0.9321 | Val Loss: 0.4891, Val Acc: 0.9565, Val mF1: 0.9577
  ✨ New best model saved with loss: 0.4891, acc: 0.9565
  Epoch 14/20 -> Train Loss: 0.5162, Train Acc: 0.9538 | Val Loss: 0.5584, Val Acc: 0.9348, Val mF1: 0.9293
  EarlyStopping counter: 1/10
  Epoch 15/20 -> Train Loss: 0.5231, Train Acc: 0.9429 | Val Loss: 0.6473, Val Acc: 0.9239, Val mF1: 0.9296
  EarlyStopping counter: 2/10
  Epoch 16/20 -> Train Loss: 0.4915, Train Acc: 0.9674 | Val Loss: 0.5188, Val Acc: 0.9348, Val mF1: 0.9065
  EarlyStopping counter: 3/10
  Epoch 17/20 -> Train Loss: 0.4615, Train Acc: 0.9728 | Val Loss: 0.5817, Val Acc: 0.9239, Val mF1: 0.9160
  EarlyStopping counter: 4/10
  Epoch 18/20 -> Train Loss: 0.4504, Train Acc: 0.9701 | Val Loss: 0.5283, Val Acc: 0.9348, Val mF1: 0.9336
  EarlyStopping counter: 5/10
  Epoch 19/20 -> Train Loss: 0.4688, Train Acc: 0.9620 | Val Loss: 0.5162, Val Acc: 0.9239, Val mF1: 0.9275
  EarlyStopping counter: 6/10
  Epoch 20/20 -> Train Loss: 0.4473, Train Acc: 0.9647 | Val Loss: 0.4888, Val Acc: 0.9348, Val mF1: 0.9389
  ✨ New best model saved with loss: 0.4888, acc: 0.9348

  Fold 1 finished. Best validation loss: 0.4888, accuracy: 0.9348

  ========== Fold 2/5 ==========
  Class Weights for Fold 2: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4720,
          1.8400], device='cuda:0')
  Epoch 1/20 -> Train Loss: 1.9238, Train Acc: 0.4130 | Val Loss: 1.8672, Val Acc: 0.4565, Val mF1: 0.3753
  ✨ New best model saved with loss: 1.8672, acc: 0.4565
  Epoch 2/20 -> Train Loss: 1.0803, Train Acc: 0.7418 | Val Loss: 0.8812, Val Acc: 0.7935, Val mF1: 0.7389
  ✨ New best model saved with loss: 0.8812, acc: 0.7935
  Epoch 3/20 -> Train Loss: 0.7723, Train Acc: 0.8560 | Val Loss: 1.5838, Val Acc: 0.7500, Val mF1: 0.6000
  EarlyStopping counter: 1/10
  Epoch 4/20 -> Train Loss: 0.8156, Train Acc: 0.8424 | Val Loss: 0.7355, Val Acc: 0.8261, Val mF1: 0.8182
  ✨ New best model saved with loss: 0.7355, acc: 0.8261
  Epoch 5/20 -> Train Loss: 0.6564, Train Acc: 0.8967 | Val Loss: 0.8361, Val Acc: 0.8043, Val mF1: 0.7664
  EarlyStopping counter: 1/10
  Epoch 6/20 -> Train Loss: 0.6488, Train Acc: 0.8995 | Val Loss: 0.6884, Val Acc: 0.9022, Val mF1: 0.8980
  ✨ New best model saved with loss: 0.6884, acc: 0.9022
  Epoch 7/20 -> Train Loss: 0.5527, Train Acc: 0.9429 | Val Loss: 0.9144, Val Acc: 0.8696, Val mF1: 0.8400
  EarlyStopping counter: 1/10
  Epoch 8/20 -> Train Loss: 0.6092, Train Acc: 0.9321 | Val Loss: 0.6118, Val Acc: 0.8913, Val mF1: 0.8684
  ✨ New best model saved with loss: 0.6118, acc: 0.8913
  Epoch 9/20 -> Train Loss: 0.5179, Train Acc: 0.9565 | Val Loss: 0.8131, Val Acc: 0.8370, Val mF1: 0.7736
  EarlyStopping counter: 1/10
  Epoch 10/20 -> Train Loss: 0.5451, Train Acc: 0.9239 | Val Loss: 0.7192, Val Acc: 0.9022, Val mF1: 0.8727
  EarlyStopping counter: 2/10
  Epoch 11/20 -> Train Loss: 0.5016, Train Acc: 0.9538 | Val Loss: 0.7208, Val Acc: 0.9022, Val mF1: 0.8726
  EarlyStopping counter: 3/10
  Epoch 12/20 -> Train Loss: 0.4817, Train Acc: 0.9592 | Val Loss: 0.5854, Val Acc: 0.8913, Val mF1: 0.8990
  ✨ New best model saved with loss: 0.5854, acc: 0.8913
  Epoch 13/20 -> Train Loss: 0.5026, Train Acc: 0.9565 | Val Loss: 0.6280, Val Acc: 0.9130, Val mF1: 0.9298
  EarlyStopping counter: 1/10
  Epoch 14/20 -> Train Loss: 0.5186, Train Acc: 0.9511 | Val Loss: 0.8409, Val Acc: 0.8261, Val mF1: 0.8054
  EarlyStopping counter: 2/10
  Epoch 15/20 -> Train Loss: 0.4519, Train Acc: 0.9674 | Val Loss: 0.7297, Val Acc: 0.8587, Val mF1: 0.8225
  EarlyStopping counter: 3/10
  Epoch 16/20 -> Train Loss: 0.5824, Train Acc: 0.9375 | Val Loss: 0.6040, Val Acc: 0.8913, Val mF1: 0.8739
  EarlyStopping counter: 4/10
  Epoch 17/20 -> Train Loss: 0.4599, Train Acc: 0.9511 | Val Loss: 0.5843, Val Acc: 0.9022, Val mF1: 0.8812
  ✨ New best model saved with loss: 0.5843, acc: 0.9022
  Epoch 18/20 -> Train Loss: 0.4183, Train Acc: 0.9755 | Val Loss: 0.6089, Val Acc: 0.9130, Val mF1: 0.8824
  EarlyStopping counter: 1/10
  Epoch 19/20 -> Train Loss: 0.4348, Train Acc: 0.9837 | Val Loss: 0.5833, Val Acc: 0.9239, Val mF1: 0.9072
  ✨ New best model saved with loss: 0.5833, acc: 0.9239
  Epoch 20/20 -> Train Loss: 0.3820, Train Acc: 0.9918 | Val Loss: 0.6092, Val Acc: 0.9130, Val mF1: 0.8824
  EarlyStopping counter: 1/10

  Fold 2 finished. Best validation loss: 0.5833, accuracy: 0.9239

  ========== Fold 3/5 ==========
  Class Weights for Fold 3: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 1.9464, Train Acc: 0.3614 | Val Loss: 1.6075, Val Acc: 0.4674, Val mF1: 0.4544
  ✨ New best model saved with loss: 1.6075, acc: 0.4674
  Epoch 2/20 -> Train Loss: 0.9576, Train Acc: 0.8179 | Val Loss: 0.9254, Val Acc: 0.7283, Val mF1: 0.6873
  ✨ New best model saved with loss: 0.9254, acc: 0.7283
  Epoch 3/20 -> Train Loss: 0.7288, Train Acc: 0.8696 | Val Loss: 0.7849, Val Acc: 0.8587, Val mF1: 0.8448
  ✨ New best model saved with loss: 0.7849, acc: 0.8587
  Epoch 4/20 -> Train Loss: 0.7156, Train Acc: 0.8777 | Val Loss: 0.7554, Val Acc: 0.8370, Val mF1: 0.8306
  ✨ New best model saved with loss: 0.7554, acc: 0.8370
  Epoch 5/20 -> Train Loss: 0.6302, Train Acc: 0.9076 | Val Loss: 0.8732, Val Acc: 0.8587, Val mF1: 0.8384
  EarlyStopping counter: 1/10
  Epoch 6/20 -> Train Loss: 0.5436, Train Acc: 0.9321 | Val Loss: 0.8162, Val Acc: 0.8478, Val mF1: 0.8393
  EarlyStopping counter: 2/10
  Epoch 7/20 -> Train Loss: 0.6150, Train Acc: 0.9130 | Val Loss: 0.8468, Val Acc: 0.8478, Val mF1: 0.8187
  EarlyStopping counter: 3/10
  Epoch 8/20 -> Train Loss: 0.5254, Train Acc: 0.9239 | Val Loss: 0.7863, Val Acc: 0.8913, Val mF1: 0.8648
  EarlyStopping counter: 4/10
  Epoch 9/20 -> Train Loss: 0.5201, Train Acc: 0.9266 | Val Loss: 0.7419, Val Acc: 0.9130, Val mF1: 0.8970
  ✨ New best model saved with loss: 0.7419, acc: 0.9130
  Epoch 10/20 -> Train Loss: 0.4698, Train Acc: 0.9592 | Val Loss: 0.6869, Val Acc: 0.9239, Val mF1: 0.9126
  ✨ New best model saved with loss: 0.6869, acc: 0.9239
  Epoch 11/20 -> Train Loss: 0.4752, Train Acc: 0.9728 | Val Loss: 0.6911, Val Acc: 0.9022, Val mF1: 0.8980
  EarlyStopping counter: 1/10
  Epoch 12/20 -> Train Loss: 0.4203, Train Acc: 0.9755 | Val Loss: 0.6615, Val Acc: 0.9348, Val mF1: 0.9298
  ✨ New best model saved with loss: 0.6615, acc: 0.9348
  Epoch 13/20 -> Train Loss: 0.4254, Train Acc: 0.9728 | Val Loss: 0.6533, Val Acc: 0.9130, Val mF1: 0.9139
  ✨ New best model saved with loss: 0.6533, acc: 0.9130
  Epoch 14/20 -> Train Loss: 0.4369, Train Acc: 0.9728 | Val Loss: 0.6822, Val Acc: 0.8804, Val mF1: 0.8813
  EarlyStopping counter: 1/10
  Epoch 15/20 -> Train Loss: 0.4323, Train Acc: 0.9755 | Val Loss: 0.6595, Val Acc: 0.9130, Val mF1: 0.9099
  EarlyStopping counter: 2/10
  Epoch 16/20 -> Train Loss: 0.4178, Train Acc: 0.9837 | Val Loss: 0.6749, Val Acc: 0.9022, Val mF1: 0.8939
  EarlyStopping counter: 3/10
  Epoch 17/20 -> Train Loss: 0.3848, Train Acc: 0.9891 | Val Loss: 0.6392, Val Acc: 0.9457, Val mF1: 0.9317
  ✨ New best model saved with loss: 0.6392, acc: 0.9457
  Epoch 18/20 -> Train Loss: 0.4016, Train Acc: 0.9891 | Val Loss: 0.6312, Val Acc: 0.9457, Val mF1: 0.9317
  ✨ New best model saved with loss: 0.6312, acc: 0.9457
  Epoch 19/20 -> Train Loss: 0.3980, Train Acc: 0.9864 | Val Loss: 0.6120, Val Acc: 0.9457, Val mF1: 0.9352
  ✨ New best model saved with loss: 0.6120, acc: 0.9457
  Epoch 20/20 -> Train Loss: 0.3869, Train Acc: 0.9946 | Val Loss: 0.5847, Val Acc: 0.9457, Val mF1: 0.9352
  ✨ New best model saved with loss: 0.5847, acc: 0.9457

  Fold 3 finished. Best validation loss: 0.5847, accuracy: 0.9457

  ========== Fold 4/5 ==========
  Class Weights for Fold 4: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6727, 1.2690, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 2.0215, Train Acc: 0.3505 | Val Loss: 2.2260, Val Acc: 0.2065, Val mF1: 0.2260
  ✨ New best model saved with loss: 2.2260, acc: 0.2065
  Epoch 2/20 -> Train Loss: 1.0891, Train Acc: 0.7500 | Val Loss: 1.0145, Val Acc: 0.7826, Val mF1: 0.7356
  ✨ New best model saved with loss: 1.0145, acc: 0.7826
  Epoch 3/20 -> Train Loss: 0.8418, Train Acc: 0.8261 | Val Loss: 0.8272, Val Acc: 0.8804, Val mF1: 0.8451
  ✨ New best model saved with loss: 0.8272, acc: 0.8804
  Epoch 4/20 -> Train Loss: 0.7438, Train Acc: 0.8668 | Val Loss: 1.3540, Val Acc: 0.7717, Val mF1: 0.7344
  EarlyStopping counter: 1/10
  Epoch 5/20 -> Train Loss: 0.7783, Train Acc: 0.8560 | Val Loss: 1.0107, Val Acc: 0.8152, Val mF1: 0.7611
  EarlyStopping counter: 2/10
  Epoch 6/20 -> Train Loss: 0.7142, Train Acc: 0.8886 | Val Loss: 0.8557, Val Acc: 0.8696, Val mF1: 0.8469
  EarlyStopping counter: 3/10
  Epoch 7/20 -> Train Loss: 0.6556, Train Acc: 0.8886 | Val Loss: 1.2285, Val Acc: 0.7935, Val mF1: 0.7650
  EarlyStopping counter: 4/10
  Epoch 8/20 -> Train Loss: 0.5914, Train Acc: 0.9158 | Val Loss: 1.0055, Val Acc: 0.8370, Val mF1: 0.8081
  EarlyStopping counter: 5/10
  Epoch 9/20 -> Train Loss: 0.5741, Train Acc: 0.9130 | Val Loss: 0.8642, Val Acc: 0.8804, Val mF1: 0.8538
  EarlyStopping counter: 6/10
  Epoch 10/20 -> Train Loss: 0.4845, Train Acc: 0.9565 | Val Loss: 0.7645, Val Acc: 0.8913, Val mF1: 0.8567
  ✨ New best model saved with loss: 0.7645, acc: 0.8913
  Epoch 11/20 -> Train Loss: 0.5000, Train Acc: 0.9592 | Val Loss: 0.6585, Val Acc: 0.8913, Val mF1: 0.8553
  ✨ New best model saved with loss: 0.6585, acc: 0.8913
  Epoch 12/20 -> Train Loss: 0.4759, Train Acc: 0.9511 | Val Loss: 0.6179, Val Acc: 0.9239, Val mF1: 0.8945
  ✨ New best model saved with loss: 0.6179, acc: 0.9239
  Epoch 13/20 -> Train Loss: 0.4933, Train Acc: 0.9592 | Val Loss: 0.6888, Val Acc: 0.9022, Val mF1: 0.8611
  EarlyStopping counter: 1/10
  Epoch 14/20 -> Train Loss: 0.4581, Train Acc: 0.9701 | Val Loss: 0.6795, Val Acc: 0.9130, Val mF1: 0.8708
  EarlyStopping counter: 2/10
  Epoch 15/20 -> Train Loss: 0.4193, Train Acc: 0.9810 | Val Loss: 0.6847, Val Acc: 0.9130, Val mF1: 0.8708
  EarlyStopping counter: 3/10
  Epoch 16/20 -> Train Loss: 0.4093, Train Acc: 0.9837 | Val Loss: 0.6714, Val Acc: 0.8913, Val mF1: 0.8447
  EarlyStopping counter: 4/10
  Epoch 17/20 -> Train Loss: 0.4332, Train Acc: 0.9728 | Val Loss: 0.6720, Val Acc: 0.9130, Val mF1: 0.8705
  EarlyStopping counter: 5/10
  Epoch 18/20 -> Train Loss: 0.4131, Train Acc: 0.9783 | Val Loss: 0.6926, Val Acc: 0.9022, Val mF1: 0.8569
  EarlyStopping counter: 6/10
  Epoch 19/20 -> Train Loss: 0.4197, Train Acc: 0.9783 | Val Loss: 0.7087, Val Acc: 0.8913, Val mF1: 0.8466
  EarlyStopping counter: 7/10
  Epoch 20/20 -> Train Loss: 0.4255, Train Acc: 0.9755 | Val Loss: 0.6817, Val Acc: 0.9022, Val mF1: 0.8569
  EarlyStopping counter: 8/10

  Fold 4 finished. Best validation loss: 0.6179, accuracy: 0.9239

  ========== Fold 5/5 ==========
  Class Weights for Fold 5: tensor([0.5750, 0.5750, 0.7667, 0.5750, 2.3000, 1.6000, 1.3143, 2.3000, 1.4154,
          1.9368], device='cuda:0')
  Epoch 1/20 -> Train Loss: 1.9261, Train Acc: 0.3913 | Val Loss: 1.6016, Val Acc: 0.5109, Val mF1: 0.4154
  ✨ New best model saved with loss: 1.6016, acc: 0.5109
  Epoch 2/20 -> Train Loss: 0.8614, Train Acc: 0.8288 | Val Loss: 0.9935, Val Acc: 0.7174, Val mF1: 0.7060
  ✨ New best model saved with loss: 0.9935, acc: 0.7174
  Epoch 3/20 -> Train Loss: 0.7278, Train Acc: 0.8424 | Val Loss: 0.9271, Val Acc: 0.8478, Val mF1: 0.7708
  ✨ New best model saved with loss: 0.9271, acc: 0.8478
  Epoch 4/20 -> Train Loss: 0.6658, Train Acc: 0.8913 | Val Loss: 0.5650, Val Acc: 0.9457, Val mF1: 0.9208
  ✨ New best model saved with loss: 0.5650, acc: 0.9457
  Epoch 5/20 -> Train Loss: 0.7051, Train Acc: 0.8560 | Val Loss: 0.8618, Val Acc: 0.8696, Val mF1: 0.8086
  EarlyStopping counter: 1/10
  Epoch 6/20 -> Train Loss: 0.6256, Train Acc: 0.9158 | Val Loss: 0.7094, Val Acc: 0.8587, Val mF1: 0.8526
  EarlyStopping counter: 2/10
  Epoch 7/20 -> Train Loss: 0.6296, Train Acc: 0.8940 | Val Loss: 0.6272, Val Acc: 0.9457, Val mF1: 0.9139
  EarlyStopping counter: 3/10
  Epoch 8/20 -> Train Loss: 0.5282, Train Acc: 0.9266 | Val Loss: 0.8897, Val Acc: 0.8696, Val mF1: 0.8218
  EarlyStopping counter: 4/10
  Epoch 9/20 -> Train Loss: 0.4929, Train Acc: 0.9565 | Val Loss: 0.8146, Val Acc: 0.9022, Val mF1: 0.8531
  EarlyStopping counter: 5/10
  Epoch 10/20 -> Train Loss: 0.4749, Train Acc: 0.9565 | Val Loss: 0.7250, Val Acc: 0.9130, Val mF1: 0.8615
  EarlyStopping counter: 6/10
  Epoch 11/20 -> Train Loss: 0.4080, Train Acc: 0.9864 | Val Loss: 0.6817, Val Acc: 0.9130, Val mF1: 0.8642
  EarlyStopping counter: 7/10
  Epoch 12/20 -> Train Loss: 0.4492, Train Acc: 0.9810 | Val Loss: 0.6643, Val Acc: 0.9130, Val mF1: 0.8642
  EarlyStopping counter: 8/10
  Epoch 13/20 -> Train Loss: 0.4424, Train Acc: 0.9565 | Val Loss: 0.6722, Val Acc: 0.9130, Val mF1: 0.8642
  EarlyStopping counter: 9/10
  Epoch 14/20 -> Train Loss: 0.4097, Train Acc: 0.9864 | Val Loss: 0.6620, Val Acc: 0.9239, Val mF1: 0.8781
  EarlyStopping counter: 10/10
  Early stopping triggered!

  Fold 5 finished. Best validation loss: 0.5650, accuracy: 0.9457
  final testing result
  py
  ===== Evaluating on the final hold-out test set =====

  Final Test Set Accuracy (Ensemble): 0.9565

  Classification Report:
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.87      1.00      0.93        20
           Cow       1.00      1.00      1.00        15
           Dog       1.00      1.00      1.00        20
        Donkey       1.00      1.00      1.00         5
          Frog       1.00      0.71      0.83         7
          Lion       1.00      1.00      1.00         9
        Maymun       1.00      0.40      0.57         5
         Sheep       1.00      1.00      1.00         8
         Tavuk       0.75      1.00      0.86         6

      accuracy                           0.96       115
     macro avg       0.96      0.91      0.92       115
  weighted avg       0.96      0.96      0.95       115
  """
  ```
## Wav2vec2
  W2V2+SVM: CV Acc=0.8913  mF1=0.8476
  W2V2+LogReg: CV Acc=0.8739  mF1=0.8384

  Holdout (W2V2+SVM) Acc=0.7565
                precision    recall  f1-score   support

          Bird       0.95      1.00      0.98        20
           Cat       0.71      0.60      0.65        20
           Cow       0.83      0.67      0.74        15
           Dog       0.88      0.75      0.81        20
        Donkey       0.71      1.00      0.83         5
          Frog       0.62      0.71      0.67         7
          Lion       0.56      1.00      0.72         9
        Maymun       0.00      0.00      0.00         5
         Sheep       0.88      0.88      0.88         8
         Tavuk       0.67      0.67      0.67         6

      accuracy                           0.76       115
     macro avg       0.68      0.73      0.69       115
  weighted avg       0.76      0.76      0.75       115
## YAMNet
  [Compute] /content/drive/MyDrive/Datasets/Audio_Cache/yamnet_train.npz
  Extracting features: 100%|████████████████████████████████████████| 460/460 [00:24<00:00, 18.74it/s]
  [Compute] /content/drive/MyDrive/Datasets/Audio_Cache/yamnet_test.npz
  Extracting features: 100%|████████████████████████████████████████| 115/115 [00:04<00:00, 28.14it/s]
  YAMNet+SVM: CV Acc=0.8913  mF1=0.8476
  YAMNet+LogReg: CV Acc=0.8717  mF1=0.8362

  Holdout (YAMNet+SVM) Acc=0.9043
                precision    recall  f1-score   support

          Bird       1.00      1.00      1.00        20
           Cat       0.82      0.90      0.86        20
           Cow       0.93      0.93      0.93        15
           Dog       0.90      0.95      0.93        20
        Donkey       0.80      0.80      0.80         5
          Frog       1.00      0.86      0.92         7
          Lion       0.90      1.00      0.95         9
        Maymun       0.67      0.40      0.50         5
         Sheep       1.00      0.75      0.86         8
         Tavuk       0.86      1.00      0.92         6

      accuracy                           0.90       115
     macro avg       0.89      0.86      0.87       115
  weighted avg       0.90      0.90      0.90       115

  Adding Max feature doesn't help much.
## ploting
  ```py
  import re
  import pandas as pd
  import matplotlib.pyplot as plt

  # Raw training log text


  # Parse lines like: Fold1 Ep1: tr 2.269/0.269 | va 2.009/0.467 mF1 0.352

  def plot_log(log_text, axes, label_prefix="", color="b"):
      pattern = re.compile(r"Ep(\d+): tr ([\d.]+)/([\d.]+) \| va ([\d.]+)/([\d.]+) mF1 ([\d.]+)")
      data = []
      for match in pattern.finditer(log_text):
          ep, tr_loss, tr_acc, va_loss, va_acc, f1 = match.groups()
          data.append({
              "epoch": int(ep),
              "train_loss": float(tr_loss),
              "train_acc": float(tr_acc),
              "val_loss": float(va_loss),
              "val_acc": float(va_acc),
              "val_f1": float(f1)
          })

      df = pd.DataFrame(data)

      # Average across folds (group by epoch)
      df_mean = df.groupby("epoch").mean(numeric_only=True)

      # Plot accuracy and loss
      ax1, ax2 = axes
      ax1.plot(df_mean.index, df_mean["train_acc"], "--", color=color, label=f"[{label_prefix}] Train Acc")
      ax1.plot(df_mean.index, df_mean["val_acc"], "-", color=color, label=f"[{label_prefix}] Val Acc")
      ax1.set_xlabel("Epoch")
      ax1.set_ylabel("Accuracy")
      ax1.legend(loc="upper left")
      ax1.grid(True)

      # ax2 = ax1.twinx()
      ax2.plot(df_mean.index, df_mean["train_loss"], "--", color=color, label=f"[{label_prefix}] Train Loss")
      ax2.plot(df_mean.index, df_mean["val_loss"], "-", color=color, label=f"[{label_prefix}] Val Loss")
      ax2.set_xlabel("Epoch")
      ax2.set_ylabel("Loss")
      ax2.legend(loc="upper left")
      ax2.grid(True)

      # lines, labels = ax1.get_legend_handles_labels()
      # lines2, labels2 = ax2.get_legend_handles_labels()
      ax1.legend(loc="lower right")
      ax2.legend(loc="upper right")
      return fig

  import re
  import pandas as pd
  import matplotlib.pyplot as plt

  IMG_EPOCH_RE = re.compile(
      r"Epoch\s+(\d+)\s*/\s*\d+\s*->\s*Train Loss:\s*([\d.]+),\s*Train Acc:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+),\s*Val Acc:\s*([\d.]+)",
      re.IGNORECASE
  )

  def parse_image_log(text: str, skip_first_n: int = 0) -> pd.DataFrame:
      rows = []
      fold = None
      for line in text.splitlines():
          lf = re.search(r"Fold\s+(\d+)\s*/\s*\d+", line)
          if lf:
              fold = int(lf.group(1))
              continue
          m = IMG_EPOCH_RE.search(line)
          if m:
              ep, tr_loss, tr_acc, va_loss, va_acc = m.groups()
              rows.append({
                  "fold": fold,
                  "epoch": int(ep),
                  "train_loss": float(tr_loss),
                  "train_acc": float(tr_acc),
                  "val_loss": float(va_loss),
                  "val_acc": float(va_acc)
              })

      df = pd.DataFrame(rows)
      if df.empty:
          raise ValueError("No epoch lines parsed.")

      # 按 fold 跳过前 N 条
      if skip_first_n > 0:
          df = (
              df.groupby("fold", group_keys=False)
                .apply(lambda g: g.iloc[skip_first_n:])
                .reset_index(drop=True)
          )

      return df

  def plot_image_log(log_text: str, axes, label_prefix: str = "", skip_first_n: int = 0, color="b") -> pd.DataFrame:
      df = parse_image_log(log_text, skip_first_n)
      # print(df)
      # average across folds per epoch
      df_mean = df.groupby("epoch", as_index=True).mean(numeric_only=True).sort_index()

      ax_acc, ax_loss = axes
      # Accuracy subplot
      ax_acc.plot(df_mean.index, df_mean["train_acc"], "--", color=color, label=f"[{label_prefix}] Train Acc")
      ax_acc.plot(df_mean.index, df_mean["val_acc"], "-", color=color, label=f"[{label_prefix}] Val Acc")
      ax_acc.set_xlabel("Epoch")
      ax_acc.set_ylabel("Accuracy")
      ax_acc.grid(True)
      ax_acc.legend(loc="lower right")

      # Loss subplot
      ax_loss.plot(df_mean.index, df_mean["train_loss"], "--", color=color, label=f"[{label_prefix}] Train Loss")
      ax_loss.plot(df_mean.index, df_mean["val_loss"], "-", color=color, label=f"[{label_prefix}] Val Loss")
      ax_loss.set_xlabel("Epoch")
      ax_loss.set_ylabel("Loss")
      ax_loss.grid(True)
      ax_loss.legend(loc="upper right")

      return df_mean


  fig, axes = plt.subplots(2, 1, figsize=(8, 12))
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  fig = plot_log(log_text_layer1, axes, "1-layer GRU + dropout 0.2", color=colors.pop(0))
  fig = plot_log(log_text_layer1_h128, axes, "+ global CMVN", color=colors.pop(0))
  fig = plot_log(log_text_layer1_h128_dr03, axes, "++ dropout 0.3", color=colors.pop(0))
  fig = plot_log(log_text_layer1_h256_dr02_global, axes, "+++ hidden 256", color=colors.pop(0))
  fig = plot_log(log_text_layer2, axes, "+++ 2-layers GRU", color=colors.pop(0))
  fig.suptitle("GRU Training Curve (Avg. over Folds)", fontsize=14)
  plt.tight_layout(rect=[0, 0, 1, 0.95])  # 给标题留出上边距

  fig, axes = plt.subplots(2, 1, figsize=(8, 12))
  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  plot_image_log(log_text_cnn_first, axes, label_prefix="EffNet + cos + AdamW + data aug", skip_first_n=1, color=colors.pop(0))
  plot_image_log(log_text_cnn_weighted, axes, label_prefix="+ Weighted Loss", skip_first_n=1, color=colors.pop(0))
  plot_image_log(log_text_harder, axes, label_prefix="++ harder data aug", skip_first_n=1, color=colors.pop(0))
  # plot_image_log(log_text_lighter, axes, label_prefix="++ lighter data aug", skip_first_n=1, color=colors.pop(0))
  plot_image_log(log_text_final, axes, label_prefix="+++ GRU output", skip_first_n=1, color=colors.pop(0))
  fig.suptitle("EfficientNet Image Model — Training vs Validation (Avg across folds)", fontsize=13)
  plt.tight_layout(rect=[0,0,1,0.95])
  plt.show()
  ```

**My Contribution:**
* I focused on the **deep learning experiments**, including **GRU with sequential features**, **transfer learning with YAMNet and Wav2Vec2**, and the **EfficientNet-based image model (CRNN)**.
* I also helped analyze results, compare performance across methods, and prepare the **final slides and presentation**.

**Challenges Faced:**
* The main challenges were the **small and imbalanced dataset**, which made deep models overfit easily, and the **integration of multiple training pipelines** across tasks.
* For deployment, I also faced an **AWS server HTTPS issue**, where microphone access was blocked over HTTP.

**How I Solved Them:**
* I learned from similar Kaggle audio classification tasks, adapting them to stabilize our model training.
* I restructured the code for consistent preprocessing and evaluation, ensuring fair comparison across all models.
* For the HTTPS issue, I **configured an Nginx reverse proxy with SSL certificates from Let’s Encrypt**, enabling secure microphone access for the deployed web app.

**What I Gained:**
* I learned how to handle audio classification by **bridging traditional and deep learning approaches**.
* I developed a stronger understanding of **feature representation and transfer learning**.
* I gained hands-on experience in **model optimization and deployment using FastAPI with a live web demo**.
