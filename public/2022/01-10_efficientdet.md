# EfficientDet
```py
import hparams_config
from tf2 import efficientdet_keras
config = hparams_config.get_efficientdet_config("efficientdet-d0")
model = efficientdet_keras.EfficientDetNet(config=config)
feats = keras.layers.Input([512, 512, 3])
cls_out_list, box_out_list = model(feats, training=False)
model.summary()
```
```py
model = efficientdet_keras.EfficientDetNet(model_name="efficientdet-d0")
print(f"{model.backbone.name = }")
# model.backbone.name = 'efficientnet-b0'
print({ii.name: ii.shape for ii in model.backbone.output})
model.backbone.summary()

inputs = keras.layers.Input([512, 512, 3])
bb = keras.models.Model(inputs, model.call(inputs, training=False))
bb.summary()
```
```py
from keras_cv_attention_models.efficientnet import efficientnet
mm = efficientnet.EfficientNetV1B0()

""" Pick all stack output layers """
dd = {}
for ii in mm.layers:
    match = re.match("^stack_?(\\d+)_block_?(\\d+)_output$", ii.name)
    if match is not None:
        cur_stack = "stack_" + match[1] + "_output"
        dd.update({cur_stack: ii})

""" Filter those have same downsample rate """
ee = {str(vv.output_shape[1]): vv for kk, vv in dd.items()}
{ii.name: ii.output_shape for ii in ee.values()}

""" Selected features """
features = list(ee.values())[1:]
```
```py
import tensorflow as tf
from keras import backend as K


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        #   y_true [batch_size, num_anchor, num_classes+1]
        #   y_pred [batch_size, num_anchor, num_classes]
        labels         = y_true[:, :, :-1]
        #   -1 是需要忽略的, 0 是背景, 1 是存在目标
        anchor_state   = y_true[:, :, -1]  
        classification = y_pred

        # 找出存在目标的先验框
        indices        = tf.where(K.not_equal(anchor_state, -1))
        labels         = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # 计算每一个先验框应该有的权重
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        # 将权重乘上所求得的交叉熵
        cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

        # 标准化，实际上是正样本的数量
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        # 将所获得的loss除上正样本的数量
        loss = K.sum(cls_loss) / normalizer
        return loss
    return _focal

def smooth_l1(sigma=3.0):
    """
    Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1
```
