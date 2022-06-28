from tensorflow.python.keras.losses import *

from Loss.loss_helper import *
from Nets.yolo3_net import get_yolo3_backend
from Utils.tf_iou import get_tf_iou


class LossYolo3(Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 iou_ignore_thresh=0.5,
                 obj_scale=2,
                 noobj_scale=1,
                 coord_scale=1,
                 class_scale=1,
                 name=None):
        self.anchor_array = anchor_array
        self.image_size = image_size
        self.batch_size = batch_size
        self.pattern_array = pattern_array
        self.iou_ignore_thresh = iou_ignore_thresh
        self.lambda_object = obj_scale
        self.lambda_no_object = noobj_scale
        self.lambda_coord = coord_scale
        self.lambda_class = class_scale
        super().__init__(name=name)

    @tf.autograph.experimental.do_not_convert
    def call(self, y_true, y_pred):
        """
        cal loss
        :param y_pred:  [b,13,13,anchors*25]
                        [t_x, t_y, t_w, t_h, t_confidence，t_class * 20]
        :param y_true:  [b,13,13,anchors,25]
                        [b_x, b_y, b_w, b_h, 1/0,            class * 20]
        :return: loss
        """

        shape_stand = [self.batch_size,
                       y_pred.shape[1],
                       y_pred.shape[2],
                       3,
                       y_pred.shape[-1] // 3]
        anchors_lay = self.pattern_array.index(shape_stand[1])  # 所属层
        anchors_current = tf.constant(self.anchor_array[anchors_lay], dtype=float)  # 当前层的三个Anchor
        gird_cell_size = self.image_size / self.pattern_array[anchors_lay]  # 采样率，每个格子像素[8,16,32]
        grid_size = y_pred.shape[1]  # [52,26,13]

        # Step 1 transform all pred output
        # reshape y_pred from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        y_pred = tf.reshape(y_pred, shape_stand)
        pred_xy = y_pred[..., 0:2]
        pred_t_coord = y_pred[..., 0:4]

        pred_b_xy = (tf.sigmoid(pred_xy) + create_grid_xy_offset(shape_stand[0:4])) / grid_size
        pred_b_wh = tf.exp(y_pred[..., 2:4]) * create_mesh_anchor(shape_stand, anchors_current) / self.image_size
        pred_b_coord = tf.concat([pred_b_xy, pred_b_wh], axis=-1)

        # Step 3 convert true coord to bounding box coord
        true_b_xy = y_true[..., 0:2]
        true_b_wh = y_true[..., 2:4]
        true_t_xy = tf.math.mod(true_b_xy * grid_size, 1)
        true_t_wh = tf.math.log(true_b_wh * self.image_size / anchors_current)
        true_t_wh = tf.where(tf.math.is_inf(true_t_wh), tf.zeros_like(true_t_wh), true_t_wh)
        true_t_coord = tf.concat(true_t_xy, true_t_wh)

        # Step 4 Get box loss scale and mask object and mask ignore
        box_loss_scale = tf.expand_dims(2 - true_b_wh[..., 0] * true_b_wh[..., 1], axis=-1)  # 制衡大小框导致的loss不均衡

        mask_object = tf.expand_dims(y_true[..., 4], 4)

        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(get_tf_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_b_coord, true_t_coord, mask_object), tf.float32)
        mask_ignore = tf.cast(best_iou < self.iou_ignore_thresh, tf.float32)

        # 置信度损失
        loss_confidence = self.get_confidence_loss_cross(y_true[..., 4],
                                                         y_pred[..., 4],
                                                         mask_object,
                                                         mask_ignore,
                                                         self.lambda_object,
                                                         self.lambda_no_object)
        # 坐标损失
        loss_coord = self.get_coordinate_loss(true_t_coord,
                                              pred_t_coord,
                                              mask_object,
                                              self.lambda_coord,
                                              box_loss_scale)

        # 分类损失
        loss_class = self.get_class_loss(y_true[..., 5:],
                                         y_pred[..., 5:],
                                         mask_object,
                                         self.lambda_class)
        total_loss = loss_confidence + loss_coord + loss_class
        return total_loss

    @staticmethod
    def get_confidence_loss_cross(
            confidence_truth,
            confidence_pred,
            object_mask,
            ignore_mask,
            lambda_object=1,
            lambda_no_object=1):
        bc_loss = binary_crossentropy(confidence_truth, confidence_pred)
        obj_loss = lambda_object * object_mask * bc_loss + lambda_no_object * (1 - object_mask) * ignore_mask * bc_loss
        return tf.reduce_sum(obj_loss, list(range(1, 4)))

    @staticmethod
    def get_coordinate_loss(coordinate_truth,
                            coordinate_pred,
                            object_mask,
                            lambda_coord,
                            box_scale):
        mse_loss = object_mask * (tf.reduce_sum(tf.square(coordinate_pred - coordinate_truth), axis=-1)) * box_scale
        loss_coord = lambda_coord * tf.reduce_sum(mse_loss, list(range(1, 4)))
        return loss_coord

    @staticmethod
    def get_class_loss(class_truth,
                       class_pred,
                       object_mask,
                       lambda_class):
        loss_cross_entropy = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(class_truth, class_pred)
        return lambda_class * tf.reduce_sum(loss_cross_entropy, list(range(1, 4)))

    @staticmethod
    def convert_coord_to_bbox_for_pred(coord, anchors, input_shape):
        t_xy = coord[..., 0:2]
        t_wh = coord[..., 2:4]
        grid_xy_offset = create_grid_xy_offset(*input_shape[0:4])
        anchor_grid = create_mesh_anchor(anchors, input_shape)
        b_xy = grid_xy_offset + tf.sigmoid(t_xy)
        b_wh = tf.exp(t_wh) * anchor_grid
        return tf.concat([b_xy, b_wh], axis=-1)

    @staticmethod
    def convert_coord_to_bbox_for_true(coord, anchors, input_shape):
        t_xy = coord[..., 0:2]
        t_wh = coord[..., 2:4]
        anchor_grid = create_mesh_anchor(anchors, input_shape)
        b_xy = t_xy
        b_wh = tf.exp(t_wh) * anchor_grid
        return tf.concat([b_xy, b_wh], axis=-1)

    @staticmethod
    def box_iou(b1, b2):
        b1 = tf.expand_dims(b1, -2)
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        # Expand dim to apply broadcasting.
        b2 = tf.expand_dims(b2, 0)
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        intersect_mins = tf.maximum(b1_mins, b2_mins)
        intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

        return iou


if __name__ == "__main__":
    # a = tf.constant([[1, 2], [2, 3]])
    # mask = tf.constant([True, False])
    # b = tf.boolean_mask(a, mask)
    # print(b)
    # pass

    from DataSet.batch_generator import *

    config_file = r"..\config\raccoon.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    train_generator = BatchGenerator(model_cfg, train_cfg, True)

    x, test_y_true = train_generator.return_next_batch()

    yolo_net = get_yolo3_backend((416, 416), 1, True)
    test_y_pred = yolo_net(x)

    test_loss = LossYolo3(model_cfg.input_size, train_cfg.batch_size,
                          model_cfg.anchor_array,
                          train_generator.pattern_shape,
                          iou_ignore_thresh=0.25,
                          coord_scale=1,
                          class_scale=1,
                          obj_scale=1,
                          noobj_scale=1)

    for i in range(0, 3):
        object_count = tf.math.count_nonzero(test_y_true[i][..., 4]).numpy()
        loss = test_loss.call(test_y_true[i], test_y_pred[i])
