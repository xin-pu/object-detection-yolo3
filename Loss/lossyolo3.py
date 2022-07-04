from tensorflow.python.keras.losses import *

from Loss.loss_helper import *
from Nets.yolo3_net import get_yolo3_backend
from Utils.tf_iou import get_tf_iou


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)

    new_shape = tf.broadcast_dynamic_shape(box_1.shape, box_2.shape)
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    return get_tf_iou(box_1, box_2)


class LossYolo3(Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 classes,
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
        self.classes = classes
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
        # gird_cell_size = self.image_size / self.pattern_array[anchors_lay]  # 采样率，每个格子像素[8,16,32]
        grid_size = y_pred.shape[1]  # [52,26,13]

        # Step 1 transform all pred output
        # reshape y_pred from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        y_pred = tf.reshape(y_pred, shape_stand)
        pred_xy = y_pred[..., 0:2]
        pred_wh = y_pred[..., 2:4]
        pred_xy_coord = tf.sigmoid(pred_xy)
        pred_t_coord = tf.concat([pred_xy_coord, pred_wh], axis=-1)  # Coord Loss

        pred_b_xy = (pred_xy_coord + create_grid_xy_offset(shape_stand[0:4])) / grid_size
        pred_b_wh = tf.exp(y_pred[..., 2:4]) * create_mesh_anchor(shape_stand, anchors_current) / self.image_size
        pred_b_coord = tf.concat([pred_b_xy, pred_b_wh], axis=-1)

        # Step 3 convert true coord to bounding box coord
        true_b_xy = y_true[..., 0:2]
        true_b_wh = y_true[..., 2:4]
        true_t_xy = tf.math.mod(true_b_xy * grid_size, 1)
        true_t_wh = tf.math.log(true_b_wh * self.image_size / anchors_current)
        true_t_wh = tf.where(tf.math.is_inf(true_t_wh), tf.zeros_like(true_t_wh), true_t_wh)
        true_t_coord = tf.concat([true_t_xy, true_t_wh], axis=-1)  # Coord Loss

        # Step 4 Get box loss scale and mask object and mask ignore
        box_loss_scale = 2 - true_b_wh[..., 0] * true_b_wh[..., 1]  # 制衡大小框导致的loss不均衡
        mask_object = y_true[..., 4]

        true_b_coord_filter = tf.boolean_mask(y_true[..., 0:4], mask_object)
        if true_b_coord_filter.shape[0] == 0:
            mask_ignore = tf.ones([y_true.shape[0], grid_size, grid_size, 3])
        else:
            pred_b_coord, true_b_coord_filter = self.broadcast_box(pred_b_coord, true_b_coord_filter)
            ious = get_tf_iou(pred_b_coord, true_b_coord_filter)
            best_iou = tf.reduce_max(ious, axis=-1)
            mask_ignore = tf.where(best_iou < self.iou_ignore_thresh, tf.constant(1.), tf.constant(0.))

        # 置信度损失
        loss_confidence = self.get_confidence_loss_cross(y_true[..., 4:5],
                                                         y_pred[..., 4:5],
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
                                         self.lambda_class,
                                         self.classes)
        total_loss = loss_confidence + loss_coord + loss_class
        # print("{},{},{}".format(loss_confidence, loss_coord, loss_class))
        return total_loss

    @staticmethod
    def get_confidence_loss_cross(
            confidence_truth,
            confidence_pred,
            object_mask,
            ignore_mask,
            lambda_object=1,
            lambda_no_object=1):
        # bc_loss = binary_crossentropy(confidence_truth, tf.sigmoid(confidence_pred))
        bc_loss = binary_crossentropy(confidence_truth, tf.sigmoid(confidence_pred))

        object_loss = object_mask * bc_loss
        no_object_loss = (1 - object_mask) * ignore_mask * bc_loss

        obj_loss = lambda_object * object_loss + lambda_no_object * no_object_loss
        return tf.reduce_sum(obj_loss, list(range(1, 4)))

    @staticmethod
    def get_coordinate_loss(coordinate_truth,
                            coordinate_pred,
                            object_mask,
                            lambda_coord,
                            box_scale):
        mse_loss = object_mask * tf.reduce_sum(tf.square(coordinate_pred - coordinate_truth), axis=-1) * box_scale
        loss_coord = lambda_coord * tf.reduce_sum(mse_loss, list(range(1, 4)))
        return loss_coord

    @staticmethod
    def get_class_loss(class_truth,
                       class_pred,
                       object_mask,
                       lambda_class,
                       classes):
        class_prob_pred = tf.sigmoid(class_pred)
        if classes == 1:
            return lambda_class * tf.reduce_sum(object_mask * binary_crossentropy(class_truth, class_prob_pred),
                                                list(range(1, 4)))
        else:
            label = tf.argmax(class_truth, axis=-1)
            loss_cross_entropy = object_mask * binary_crossentropy(label, class_prob_pred)
            # print(tf.reduce_sum(class_pred, axis=-1))
            loss_class = lambda_class * tf.reduce_sum(loss_cross_entropy, list(range(1, 4)))
            return loss_class

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
    def broadcast_box(b1, b2):
        b1_shape = b1.shape
        b2_shape = tf.shape(b2)

        b1 = tf.expand_dims(b1, -2)
        b1 = tf.tile(b1, [1, 1, 1, 1, b2_shape[0], 1])

        b2 = tf.reshape(b2, shape=[1, 1, 1, 1, b2_shape[0], b2_shape[1]])
        b2 = tf.tile(b2, [b1_shape[0], b1_shape[1], b1_shape[2], b1_shape[3], 1, 1])
        return b1, b2


if __name__ == "__main__":

    from DataSet.batch_generator import *

    config_file = r"..\config\pascalVocDebug.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    train_generator = BatchGenerator(model_cfg, train_cfg, False)

    x, test_y_true = train_generator.return_next_batch()

    yolo_net = get_yolo3_backend((416, 416), 20, True)
    test_y_pred = yolo_net(x)

    test_loss = LossYolo3(model_cfg.input_size, train_cfg.batch_size,
                          model_cfg.anchor_array,
                          train_generator.pattern_shape,
                          train_generator.classes,
                          iou_ignore_thresh=0.5,
                          coord_scale=1,
                          class_scale=1,
                          obj_scale=1,
                          noobj_scale=1)

    for i in range(0, 3):
        loss = test_loss.call(test_y_true[i], test_y_pred[i])
