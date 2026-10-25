# Backbones & ImageNet Training

Porting PyTorch models to Keras (keras_cv_attention_models), ImageNet training recipes, new architectures.

Back to [all indexes](README.md).

- **[PyTorch to Keras](../public/2021/08-20_pytorch_to_keras.md)** (2021-08-20) — Porting timm/PyTorch backbones to Keras (keras_cv_attention_models) — dozens of models and weight conversions.
  - [Timm](../public/2021/08-20_pytorch_to_keras.md#timm) · [Resnest](../public/2021/08-20_pytorch_to_keras.md#resnest) · [BotNet](../public/2021/08-20_pytorch_to_keras.md#BotNet) · [VOLO](../public/2021/08-20_pytorch_to_keras.md#volo) · [CotNet](../public/2021/08-20_pytorch_to_keras.md#cotnet) · [Attetion models](../public/2021/08-20_pytorch_to_keras.md#Attetion%20models) · [Noisy Student EfficientNet](../public/2021/08-20_pytorch_to_keras.md#Noisy%20Student%20EfficientNet) · [CMT](../public/2021/08-20_pytorch_to_keras.md#CMT)
- **[ImageNet Basic](../public/2021/10-14_imagenet_basic.md)** (2021-10-14) — ImageNet basics — dataset init, timm training and validation, TF training, pretrained model evaluation.
  - [Init dataset](../public/2021/10-14_imagenet_basic.md#Init%20dataset) · [Timm Imagenet](../public/2021/10-14_imagenet_basic.md#Timm%20Imagenet) · [TF Imagenet Basic](../public/2021/10-14_imagenet_basic.md#TF%20Imagenet%20Basic)
- **[ImageNet Train](../public/2022/01-09_imagenet_train.md)** (2022-01-09) — ImageNet training with kecam — FLOPs, training logs (AotNet, CoAtNet, CMT, SwinV2), distillation.
  - [Flops](../public/2022/01-09_imagenet_train.md#flops) · [Training logs](../public/2022/01-09_imagenet_train.md#Training%20logs) · [CMT](../public/2022/01-09_imagenet_train.md#cmt) · [SwinTransformerV2Tiny_ns](../public/2022/01-09_imagenet_train.md#swintransformerv2tiny_ns) · [Freezing backbone](../public/2022/01-09_imagenet_train.md#Freezing%20backbone)
- **[advance training](../public/2022/05-18_advance_training.md)** (2022-05-18) — Advanced training tricks — token labeling, ROI align/crop, distillation, distribution and DTensor.
  - [Token label](../public/2022/05-18_advance_training.md#Token%20label) · [Distill](../public/2022/05-18_advance_training.md#distill) · [Distribution](../public/2022/05-18_advance_training.md#distribution) · [DTensor](../public/2022/05-18_advance_training.md#dtensor)
- **[eva02](../public/2023/04-25_eva02.md)** (2023-04-25) — EVA02 rotary embedding port from timm to Keras with output checks.
- **[deformable conv](../public/2023/04-29_deformable_conv.md)** (2023-04-29) — Deformable convolution and attention (DCNv2, MSDeformAttn) implemented in TF/Keras.
  - [DCNv2 Deformable Convolution Network](../public/2023/04-29_deformable_conv.md#DCNv2%20Deformable%20Convolution%20Network) · [Python failback of MSDeformAttnFunction](../public/2023/04-29_deformable_conv.md#Python%20failback%20of%20MSDeformAttnFunction) · [DeformableAttention2D](../public/2023/04-29_deformable_conv.md#deformableattention2d) · [deformable_attention_core_func](../public/2023/04-29_deformable_conv.md#deformable_attention_core_func) · [DCNv2](../public/2023/04-29_deformable_conv.md#dcnv2) · [Grid sample](../public/2023/04-29_deformable_conv.md#Grid%20sample) · [Torchvision](../public/2023/04-29_deformable_conv.md#torchvision)
- **[deepmad](../public/2023/07-09_deepmad.md)** (2023-07-09) — DeepMAD model port attempt (unfinished — weights not right).
- **[mamba](../public/2024/09-2_mamba.md)** (2024-09-02) — Mamba model implementation in keras_cv_attention_models.

## See also

- [Model Conversion, Compression & Inference](model_deployment.md)
- [Deep Learning Frameworks](dl_frameworks.md)
