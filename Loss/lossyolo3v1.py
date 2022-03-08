import tensorflow as tf
from tensorflow.keras.losses import Loss


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

    mesh_xy_nbox = tf.tile(mesh_xy, [batch_size, 1, 1, n_box, 1])
    return mesh_xy_nbox


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


def adjust_pred_tensor(y_pred, input_shape):
    xy = y_pred[..., 0:2]
    wh = y_pred[..., 2:4]

    grid_xy_offset = create_grid_xy_offset(*input_shape[0:4])

    pred_xy = grid_xy_offset + tf.sigmoid(xy)  # sigmoid(t_x) + c_x and sigmoid(t_y) + c_y
    pred_wh = wh

    pred_conf = tf.sigmoid(y_pred[..., 4])  # adjust confidence
    pred_classes = y_pred[..., 5:]

    return tf.concat([pred_xy, pred_wh, tf.expand_dims(pred_conf, axis=-1), pred_classes], axis=-1)


def adjust_true_tensor(y_true):
    xywh_confidence = y_true[..., :5]
    true_class = tf.argmax(y_true[..., 5:], -1)
    trues = tf.concat([xywh_confidence, tf.expand_dims(tf.cast(true_class, tf.float32), -1)], axis=-1)
    return trues


def conf_delta_tensor(y_true, y_pred, pred_shape, anchors, ignore_thresh):
    pred_box_xy, pred_box_wh, pred_box_conf = y_pred[..., :2], y_pred[..., 2:4], y_pred[..., 4]

    anchor_grid = create_mesh_anchor(anchors, pred_shape)
    anchors_ = tf.reshape(anchors, shape=[1, 1, 1, 3, 2])

    true_wh = y_true[:, :, :, :, 2:4]
    true_wh = anchor_grid * tf.exp(true_wh)
    true_wh = true_wh * tf.expand_dims(y_true[:, :, :, :, 4], 4)

    # then, ignore the boxes which have good overlap with some true box
    true_xy = y_true[..., 0:2]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = pred_box_xy
    pred_wh = tf.exp(pred_box_wh) * anchors_

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    best_ious = tf.truediv(intersect_areas, union_areas)

    conf_delta = pred_box_conf * tf.cast(best_ious < ignore_thresh, tf.float32)
    return conf_delta


def wh_scale_tensor(true_box_wh, anchors, image_size):
    image_size_ = tf.reshape(tf.cast([image_size, image_size], tf.float32), [1, 1, 1, 1, 2])
    anchors_ = tf.reshape(anchors, shape=[1, 1, 1, 3, 2])

    # [0, 1]-scaled width/height
    wh_scale = tf.exp(true_box_wh) * anchors_ / image_size_
    # the smaller the box, the bigger the scale
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)
    return wh_scale


def loss_coord_tensor(object_mask, pred_box, true_box, wh_scale, xywh_scale):
    xy_delta = object_mask * (pred_box - true_box) * wh_scale * xywh_scale
    loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
    return loss_xy


def loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, obj_scale, noobj_scale, conf_delta):
    object_mask_ = tf.squeeze(object_mask, axis=-1)
    conf_delta = object_mask_ * (pred_box_conf - true_box_conf) * obj_scale + (
            1 - object_mask_) * conf_delta * noobj_scale
    loss_conf = tf.reduce_sum(tf.square(conf_delta), list(range(1, 4)))
    return loss_conf


def loss_class_tensor(object_mask, pred_box_class, true_box_class, class_scale):
    true_box_class_ = tf.cast(true_box_class, tf.int64)
    class_delta = object_mask * tf.expand_dims(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class_, logits=pred_box_class),
        4) * class_scale
    loss_class = tf.reduce_sum(class_delta, list(range(1, 5)))
    return loss_class


class LossYolo3V1(Loss):
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

        # Step final cal loss and sum
        loss = self.cal_loss(y_true, y_pred, shape_stand)
        return loss

    def reshape_pred(self, y_pred):
        """
        讲预测结果张量调整到该Anchor Lay下 五维张量
        reshape y_preds from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        :param y_pred:
        :return:
        """
        anchor_shape = self.anchor_array.shape
        patter_shape = y_pred.shape[1:3]
        target_shape = [self.batch_size, patter_shape[0], patter_shape[1], anchor_shape[0],
                        y_pred.shape[-1] // anchor_shape[0]]
        return tf.reshape(y_pred, target_shape)

    def cal_loss(self, y_true, y_pred, input_shape):
        """
        计算预测框 于 真实框 的损失
        :param y_true: tensor with shape [batch size, 13, 13, 3, 25]
        :param y_pred: tensor with shape [batch size, 13, 13, 3, 25]
        :param input_shape: [batch size, 13, 13, 3, 25]
        :return: loss of yolo detect object
        """

        anchors_lay = self.pattern_array.index(input_shape[1])
        anchors_current = tf.constant(self.anchor_array[anchors_lay], dtype=float)
        object_mask = tf.expand_dims(y_true[..., 4], 4)  # 真实值中有物体的矩阵

        # 2. Adjust prediction (bxy, twh)
        preds = adjust_pred_tensor(y_pred, input_shape)

        # 3. Adjust ground truth (bxy, twh)
        # shape: [batch, 13, 13, 3, 6] 6 means: x,y,w,h,confidence,true class
        trues = adjust_true_tensor(y_true)

        # 4. conf_delta tensor
        conf_delta = conf_delta_tensor(y_true, preds, input_shape, anchors_current, self.ignore_thresh)

        # 5. loss tensor
        wh_scale = wh_scale_tensor(trues[..., 2:4], anchors_current, self.image_size)

        loss_box = loss_coord_tensor(object_mask, preds[..., :4], trues[..., :4], wh_scale, self.scale_coord)
        loss_conf = loss_conf_tensor(object_mask, preds[..., 4], trues[..., 4], self.scale_object, self.scale_noobj,
                                     conf_delta)
        loss_class = loss_class_tensor(object_mask, preds[..., 5:], trues[..., 5], self.scale_class)
        loss = loss_box + loss_conf + loss_class
        return loss


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
    test_loss = LossYolo3V1(model_cfg.input_size, train_cfg.batch_size,
                            model_cfg.anchor_array, train_generator.pattern_shape).call(single_y_true, y_pred)
    print("Sum Loss:\t{}".format(test_loss))
    #
