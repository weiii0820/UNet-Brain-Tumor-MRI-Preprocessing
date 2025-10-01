import tensorflow as tf
from tensorflow.keras import backend as K


class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='dice_coef', smooth=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_value = (2. * intersection + self.smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + self.smooth)
        self.dice.assign_add(dice_value)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice / self.count

    def reset_states(self):
        self.dice.assign(0.0)
        self.count.assign(0.0)


class SafeMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, name='safe_mean_iou', num_classes=2, threshold=0.3, **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.int32)
        y_true = tf.cast(tf.round(y_true), tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)


# === Tversky Loss & Combo Loss ===
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def combined_bce_tversky_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + tversky_loss(y_true, y_pred)