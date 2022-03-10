from tensorflow.keras.losses import Loss
from tensorflow.keras.backend import *
from Loss.loss_helper import *
from Utils.tf_iou import *


class LossYolo4(Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 iou_ignore_thresh=0.55,
                 name=None):
        self.anchor_array = anchor_array
        self.image_size = image_size
        self.batch_size = batch_size
        self.pattern_array = pattern_array
        self.ignore_thresh = iou_ignore_thresh
        super().__init__(name=name)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """
        cal loss
        :param y_pred:  [b,13,13,anchors*25]
                        [t_x, t_y, t_w, t_h, t_confidence，t_class * 20]
        :param y_true:  [b,13,13,anchors,25]
                        [b_x, b_y, t_w, t_h, 1/0,            class * 20]
        :return: loss = loss_ciou + loss_confidence + loss_class
        """

        shape_stand = [self.batch_size,
                       y_pred.shape[1],
                       y_pred.shape[2],
                       3,
                       y_pred.shape[-1] // 3]
        anchors_lay = self.pattern_array.index(shape_stand[1])
        anchors_current = tf.constant(self.anchor_array[anchors_lay], dtype=float)

        # Step 1 reshape y_preds from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        y_pred = tf.reshape(y_pred, shape_stand)

        # Step 2 get object mask from true
        mask_object = y_true[..., 4:5]

        # Step 3 convert pred coord to bounding box coord
        pred_b_xy = tf.sigmoid(y_pred[..., 0:2]) + create_grid_xy_offset(shape_stand[0:4])
        pred_b_wh = tf.exp(y_pred[..., 2:4]) * create_mesh_anchor(shape_stand, anchors_current)
        pred_b_confidence = tf.sigmoid(y_pred[..., 4:5])

        # Step 3 convert true coord to bounding box coord
        true_b_xy = y_true[..., 0:2]
        true_b_wh = tf.exp(y_true[..., 2:4]) * anchors_current

        # Step 3 adjust pred tensor to [bx, by, bw, bh] and get ignore mask by iou
        pred_b_coord = tf.concat([pred_b_xy, pred_b_wh], axis=-1)
        true_b_coord = tf.concat([true_b_xy, true_b_wh], axis=-1)
        iou = get_tf_iou(pred_b_coord, true_b_coord)
        max_iou_mask = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        mask_background = tf.expand_dims(
            (1 - tf.squeeze(mask_object, axis=-1)) * tf.cast(max_iou_mask < self.ignore_thresh, tf.float32), axis=-1)

        # Step 4 cal 3 part loss
        ciou = tf.expand_dims(get_tf_ciou(pred_b_coord, true_b_coord), axis=-1)
        bbox_loss_scale = 2 - 1.0 * y_true[..., 2:3] * y_true[..., 3:4] / self.image_size ** 2
        loss_ciou = mask_object * bbox_loss_scale * (1 - ciou)
        final_loss_ciou = tf.reduce_mean(tf.reduce_sum(loss_ciou, axis=[1, 2, 3, 4]))

        final_loss_class = self.get_class_loss(y_true[..., 5:],
                                               y_pred[..., 5:],
                                               mask_object)

        pos_loss = mask_object * (0 - tf.math.log(pred_b_confidence))
        neg_loss = mask_background * (0 - tf.math.log(1 - pred_b_confidence))
        loss_confidence = pos_loss + neg_loss

        final_loss_confidence = tf.reduce_mean(tf.reduce_sum(loss_confidence, axis=[1, 2, 3, 4]))

        return final_loss_ciou + final_loss_confidence + final_loss_class

    @staticmethod
    def get_class_loss(class_truth,
                       class_pred,
                       object_mask,
                       lambda_class=1):
        """
        calculate class loss by sigmoid_cross_entropy_with_logits
        class_pred will sigmoid to cal loss
        @param class_truth: shape[..., n] tensor with n class
        @param class_pred:  shape[..., n] tensor with n class
        @param object_mask:
        @param lambda_class:
        @return:
        """
        loss_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=class_truth, logits=class_pred)
        class_delta = object_mask * loss_cross_entropy
        return lambda_class * tf.reduce_mean(tf.reduce_sum(class_delta, axis=[1, 2, 3, 4]))


if __name__ == "__main__":
    from DataSet.batch_generator import *

    config_file = r"..\config\pascalVoc.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    train_generator = BatchGenerator(model_cfg, train_cfg, True)

    _, test_y_true = train_generator.return_next_batch()
    single_y_true = test_y_true[0]
    shape = single_y_true.shape
    end_dims = shape[-1] * shape[-2]
    new_shape = [shape[0], shape[1], shape[2], end_dims]
    test_y_pred = single_y_true.reshape(new_shape)
    test_y_pred = tf.zeros_like(test_y_pred)
    test_loss = LossYolo4(model_cfg.input_size, train_cfg.batch_size,
                          model_cfg.anchor_array, train_generator.pattern_shape).call(single_y_true, test_y_pred)
    print("Sum Loss:\t{}".format(test_loss))
