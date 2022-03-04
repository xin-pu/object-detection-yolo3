import tensorflow as tf
from tensorflow.keras.losses import Loss


class Loss(Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 ignore_thresh=0.5,
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xywh_scale=1,
                 class_scale=1,
                 name=None):
        self.anchor_array = anchor_array
        self.image_size = image_size
        self.batch_size = batch_size
        self.pattern_array = pattern_array
        self.ignore_thresh = ignore_thresh
        self.grid_scale = grid_scale
        self.scale_object = obj_scale
        self.scale_noobj = noobj_scale
        self.scale_coord = xywh_scale
        self.scale_class = class_scale
        super().__init__(name=name)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """
        cal loss
        :param y_pred: [b,13,13,anchors*25]
        :param y_true: [b,13,13,anchors,25]
        :return: loss
        """
        shape_stand = [self.batch_size,
                       y_pred.shape[1],
                       y_pred.shape[2],
                       3,
                       y_pred.shape[-1] // 3]

        # Step 1 reshape y_preds from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        y_pred = self.reshape_pred(y_pred)

        anchors_lay = self.pattern_array.index(shape_stand[1])
        anchors_current = tf.constant(self.anchor_array[anchors_lay], dtype=float)

        object_mask_from_true = tf.expand_dims(y_true[..., 4], 4)  # 真实值中有物体的矩阵
        
        
        return loss
