import tensorflow as tf
from tensorflow.keras.losses import Loss

import Utils.iou as iou_module


def create_grid_xy_offset(batch_size, grid_h, grid_w, n_box):
    """

    :param batch_size: 4
    :param grid_h: 13
    :param grid_w: 13
    :param n_box: 3
    :return: return shape is [batch_size,grid_h,grid_w,n_box,2], 2 means: x offset,y offset
    """
    basic = tf.reshape(tf.tile(tf.range(grid_h), [grid_w]), (1, grid_h, grid_w, 1, 1))

    mesh_x = tf.cast(basic, tf.float32)
    mesh_y = tf.transpose(mesh_x, (0, 2, 1, 3, 4))

    mesh_xy = tf.concat([mesh_x, mesh_y], -1)

    return tf.tile(mesh_xy, [batch_size, 1, 1, n_box, 1])


def create_mesh_anchor(anchors, pred_shape):
    """
    # Returns
        mesh_anchor : Tensor, shape of (batch_size, grid_h, grid_w, n_box, 2)
            [..., 0] means "anchor_w"
            [..., 1] means "anchor_h"
    """
    anchor_list = tf.reshape(anchors, shape=[6])
    batch_size, grid_h, grid_w, n_box = pred_shape[0:4]
    mesh_anchor = tf.tile(anchor_list, [batch_size * grid_h * grid_w])
    mesh_anchor = tf.reshape(mesh_anchor, [batch_size, grid_h, grid_w, n_box, 2])
    mesh_anchor = tf.cast(mesh_anchor, tf.float32)
    return mesh_anchor


def create_wh_scale(true_box_wh, anchors, image_size):
    image_size_ = tf.reshape(tf.cast([image_size, image_size], tf.float32), [1, 1, 1, 1, 2])
    anchors_ = tf.reshape(anchors, shape=[1, 1, 1, 3, 2])

    # [0, 1]-scaled width/height
    wh_scale = tf.exp(true_box_wh) * anchors_ / image_size_
    # the smaller the box, the bigger the scale
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)
    return wh_scale


class LossYolo3V2(Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 ignore_thresh=1,
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 coord_scale=1,
                 class_scale=1,
                 name=None):
        self.anchor_array = anchor_array
        self.image_size = image_size
        self.batch_size = batch_size
        self.pattern_array = pattern_array
        self.ignore_thresh = ignore_thresh
        self.grid_scale = grid_scale
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
                        b_x = sigmoid(t_x) + c_x
                        b_y = sigmoid(t_y) + c_y
                        b_w = e^t_w * anchor_w
                        b_h = e^t_h * anchor_h

                        t_w = ln(b_w / anchor_w)
                        t_h = ln(b_h / anchor_h)

        :param y_true:  [b,13,13,anchors,25]
                        [b_x, b_y, t_w, t_h, 1/0,            class * 20]
                        b_w = e^nor_w * anchor_w
                        b_h = e^nor_h * anchor_h

                        t_w = ln(b_w / anchor_w)
                        t_h = ln(b_h / anchor_h)

        :return: loss

            class loss = softmax_cross_entropy_with_logits(labels,logits)
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
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # Step 3 convert pred coord to bounding box coord
        pred_b_xy = tf.sigmoid(y_pred[..., 0:2]) + create_grid_xy_offset(*shape_stand[0:4])
        pred_b_wh = tf.exp(y_pred[..., 2:4]) * create_mesh_anchor(anchors_current, shape_stand)

        # Step 3 convert true coord to bounding box coord
        true_b_xy = y_true[..., 0:2]
        true_b_wh = tf.exp(y_true[..., 2:4]) * anchors_current

        # Step 3 adjust pred tensor to [bx, by, bw, bh] and get ignore mask by iou
        pred_b_coord = tf.concat([pred_b_xy, pred_b_wh], axis=-1)
        true_b_coord = tf.concat([true_b_xy, true_b_wh], axis=-1)
        iou_mask = iou_module.get_tf_diou(pred_b_coord, true_b_coord)
        ignore_mask = tf.cast(iou_mask < self.ignore_thresh, tf.float32)

        # Step 4 cal 3 part loss
        wh_scale = create_wh_scale(y_true[..., 2:4], anchors_current, self.image_size)  # 制衡大小框导致的loss不均衡
        loss_coord = self.get_coordinate_loss(y_true[..., 0:4],
                                              tf.concat([pred_b_xy, y_pred[..., 2:4]], axis=-1),
                                              object_mask,
                                              self.lambda_coord,
                                              wh_scale)

        loss_confidence = self.get_confidence_loss(y_true[..., 4],
                                                   tf.sigmoid(y_pred[..., 4]),
                                                   object_mask,
                                                   ignore_mask,
                                                   self.lambda_object,
                                                   self.lambda_no_object)

        loss_class = self.get_class_loss_original(y_true[..., 5:],
                                                  y_pred[..., 5:],
                                                  object_mask)

        return loss_coord + loss_confidence + loss_class

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
    def get_confidence_loss(
            confidence_truth,
            confidence_pred,
            object_mask,
            ignore_mask,
            lambda_object=5,
            lambda_no_object=1):
        confidence_mask = tf.squeeze(object_mask, axis=-1)
        confidence_loss = lambda_object * confidence_mask * (confidence_pred - confidence_truth) + \
                          lambda_no_object * (1 - confidence_mask) * ignore_mask * confidence_pred
        return tf.reduce_sum(tf.square(confidence_loss), list(range(1, 4)))

    @staticmethod
    def get_coordinate_loss(coordinate_truth,
                            coordinate_pred,
                            object_mask,
                            xywh_scale,
                            wh_scale):
        """
        计算有目标检测框的坐标损失
        @param coordinate_truth:
        @param coordinate_pred:
        @param object_mask:
        @param xywh_scale:
        @return:
        """
        xy_delta = object_mask * (coordinate_pred - coordinate_truth) * xywh_scale * wh_scale
        loss_coord = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        return loss_coord

    @staticmethod
    def get_class_loss(class_truth,
                       class_pred,
                       object_mask,
                       lambda_class=1):
        """
        calculate class loss

        @param class_truth: shape[..., n] tensor with n class
        @param class_pred:  shape[..., n] tensor with n class
        @param object_mask:
        @param lambda_class:
        @return:
        """
        true_class = tf.argmax(class_truth, -1)
        class_truth = tf.cast(true_class, tf.int64)
        loss_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(class_truth, class_pred)
        class_delta = object_mask * tf.expand_dims(loss_cross_entropy, 4)
        return lambda_class * tf.reduce_sum(class_delta, axis=[1, 2, 3, 4])

    @staticmethod
    def get_class_loss_original(class_truth,
                                class_pred,
                                object_mask,
                                lambda_class=1):
        """
        calculate class loss
        @param class_truth: shape[..., n] tensor with n class
        @param class_pred:  shape[..., n] tensor with n class
        @param object_mask:
        @param lambda_class:
        @return:
        """
        class_delta_mask = object_mask * tf.square(class_truth - class_pred)
        return lambda_class * tf.reduce_sum(class_delta_mask, axis=[1, 2, 3, 4])


if __name__ == "__main__":
    from DataSet.batch_generator import *

    config_file = r"..\config\raccoon.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    train_generator = BatchGenerator(model_cfg, train_cfg, True)

    _, y_true = train_generator.return_next_batch()
    single_y_true = y_true[0]
    shape = single_y_true.shape
    end_dims = shape[-1] * shape[-2]
    new_shape = [shape[0], shape[1], shape[2], end_dims]
    y_pred = single_y_true.reshape(new_shape)
    y_pred = tf.zeros_like(y_pred)
    test_loss = LossYolo3V2(model_cfg.input_size, train_cfg.batch_size,
                            model_cfg.anchor_array, train_generator.pattern_shape).call(single_y_true, y_pred)
    print("Sum Loss:\t{}".format(test_loss))
