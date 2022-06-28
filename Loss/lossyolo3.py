import tf2onnx.onnx_opset.math
from tensorflow.keras.losses import Loss

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
                        [b_x, b_y, t_w, t_h, 1/0,            class * 20]
        :return: loss
        """

        shape_stand = [self.batch_size,
                       y_pred.shape[1],
                       y_pred.shape[2],
                       3,
                       y_pred.shape[-1] // 3]
        anchors_lay = self.pattern_array.index(shape_stand[1])
        anchors_current = tf.constant(self.anchor_array[anchors_lay], dtype=float)
        gird_pattern = self.image_size / self.pattern_array[anchors_lay]  # 采样率，每个格子像素[8,16,32]
        grid_shape = y_pred.shape[1]

        # Step 1 reshape y_preds from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        y_pred = tf.reshape(y_pred, shape_stand)

        # Step 2 get object mask from true
        mask_object = tf.expand_dims(y_true[..., 4], 4)

        # Step 3 convert true coord to bounding box coord
        true_b_xy = tf.math.mod(y_true[..., 0:2] * gird_pattern, 1)
        image_size_ = tf.reshape(tf.cast([self.image_size, self.image_size], tf.float32), [1, 1, 1, 1, 2])
        anchors_ = image_size_ / tf.reshape(anchors_current, shape=[1, 1, 1, 3, 2])
        true_b_wh = tf.math.log(y_true[..., 2:4] * anchors_)
        true_b_wh = tf.where(mask_object == 1., true_b_wh, tf.zeros_like(true_b_wh))
        true_b_coord = tf.concat([true_b_xy, true_b_wh], axis=-1)

        wh_scale = tf.expand_dims(2 - y_true[..., 2] * y_true[..., 3], axis=4)  # 制衡大小框导致的loss不均衡

        pred_b_xy = (tf.sigmoid(y_pred[..., 0:2]) + create_grid_xy_offset(shape_stand[0:4])) / grid_shape
        pred_b_wh = tf.exp(y_pred[..., 2:4]) * create_mesh_anchor(shape_stand, anchors_current) / self.image_size
        pred_b_coord = tf.concat([pred_b_xy, pred_b_wh], axis=-1)

        # Find Ignore mask
        mask_object_bool = tf.cast(y_true[..., 4], 'bool')
        a = []
        for b in range(0, self.batch_size):
            true_box = tf.boolean_mask(y_true[b, ..., 0:4], mask_object_bool[b, ...])
            if true_box.shape[0] == 0:
                a.append(tf.ones([grid_shape, grid_shape, 3]))
            else:
                ious = self.box_iou(pred_b_coord[b], true_box)
                best_iou = tf.reduce_max(ious, axis=-1)
                ignore_mask = tf.where(best_iou < self.iou_ignore_thresh, tf.constant(1.), tf.constant(0.))
                a.append(ignore_mask)
        final_ignore_mask = tf.stack(a)

        # 置信度损失
        loss_confidence = self.get_confidence_loss_cross(y_true[..., 4],
                                                         y_pred[..., 4],
                                                         mask_object,
                                                         final_ignore_mask,
                                                         self.lambda_object,
                                                         self.lambda_no_object)
        # 坐标损失
        loss_coord = self.get_coordinate_loss(true_b_coord,
                                              y_pred[..., 0:4],
                                              mask_object,
                                              self.lambda_coord,
                                              wh_scale,
                                              model=0)

        # 分类损失
        loss_class = self.get_class_loss(y_true[..., 5:],
                                         y_pred[..., 5:],
                                         mask_object,
                                         self.lambda_class)
        total_loss = loss_confidence + loss_coord + loss_class
        return total_loss

    @staticmethod
    def get_confidence_loss(
            confidence_truth,
            confidence_pred,
            object_mask,
            ignore_mask,
            lambda_object=5,
            lambda_no_object=1):
        con_pred = tf.sigmoid(confidence_pred)

        mask_object = tf.squeeze(object_mask, axis=-1)
        loss_object = mask_object * tf.square(con_pred - confidence_truth)
        loss_no_object = ignore_mask * (1 - mask_object) * tf.square(con_pred - confidence_truth)
        confidence_loss = lambda_object * loss_object + lambda_no_object * loss_no_object
        # print("Object J:{}\tNo_object J{}\tSum J{}".format(loss_object, loss_no_object, confidence_loss))
        return tf.reduce_mean(confidence_loss, list(range(1, 4)))

    @staticmethod
    def get_confidence_loss_cross(
            confidence_truth,
            confidence_pred,
            object_mask,
            ignore_mask,
            lambda_object=5,
            lambda_no_object=1):
        object_mask = tf.squeeze(object_mask, axis=-1)
        mask_no_object = (1 - object_mask) * ignore_mask
        loss_object = object_mask * \
                      tf.nn.sigmoid_cross_entropy_with_logits(confidence_truth, confidence_pred)
        loss_no_object = mask_no_object * \
                         tf.nn.sigmoid_cross_entropy_with_logits(confidence_truth, confidence_pred)

        confidence_loss = lambda_object * loss_object + lambda_no_object * loss_no_object
        return tf.reduce_sum(confidence_loss, list(range(1, 4)))

    @staticmethod
    def get_coordinate_loss(coordinate_truth,
                            coordinate_pred,
                            object_mask,
                            lambda_coord,
                            box_scale,
                            model=0):
        """
        计算有目标检测框的坐标损失
        :param model:
        :param coordinate_truth:
        :param coordinate_pred:
        :param object_mask:
        :param lambda_coord:
        :param box_scale:
        :return:
        """
        if model == 0:
            # Xin.Pu cross entropy is helpful to avoid exp overflow
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(coordinate_truth[..., 0:2],
                                                                    coordinate_pred[..., 0:2])
            xy_loss = object_mask * box_scale * cross_entropy
            wh_loss = object_mask * box_scale * 0.5 * (
                tf.math.square(coordinate_truth[..., 0:2] - coordinate_pred[..., 0:2]))
            loss_coord = lambda_coord * tf.reduce_sum(xy_loss + wh_loss, list(range(1, 5)))
            return loss_coord
        elif model == 1:
            mse_loss = object_mask * (tf.square(coordinate_pred - tf.sigmoid(coordinate_truth))) * box_scale
            loss_coord = lambda_coord * tf.reduce_sum(mse_loss, list(range(1, 5)))
            return loss_coord

    @staticmethod
    def get_class_loss(class_truth,
                       class_pred,
                       object_mask,
                       lambda_class):
        """
        calculate class loss by sigmoid_cross_entropy_with_logits
        class_pred will sigmoid to cal loss
        @param class_truth: shape[..., n] tensor with n class
        @param class_pred:  shape[..., n] tensor with n class
        @param object_mask:
        @param lambda_class:
        @return:
        """
        loss_cross_entropy = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_truth,
                                                                                   logits=class_pred)
        return lambda_class * tf.reduce_sum(loss_cross_entropy, axis=[1, 2, 3, 4])

    @staticmethod
    def create_wh_scale(true_box_wh, anchors, image_size):
        image_size_ = tf.reshape(tf.cast([image_size, image_size], tf.float32), [1, 1, 1, 1, 2])
        anchors_ = tf.reshape(anchors, shape=[1, 1, 1, 3, 2])

        # [0, 1]-scaled width/height
        wh_scale = tf.exp(true_box_wh) * anchors_ / image_size_
        # the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)
        return wh_scale

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
        '''Return iou tensor

        Parameters
        ----------
        b1: tensor, shape=(i1,...,iN, 4), xywh
        b2: tensor, shape=(j, 4), xywh

        Returns
        -------
        iou: tensor, shape=(i1,...,iN, j)

        '''

        # Expand dim to apply broadcasting.
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
