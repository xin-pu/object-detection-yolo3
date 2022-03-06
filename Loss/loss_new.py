from Loss.iou import *


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


class LossYolo3(tf.keras.losses.Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 ignore_thresh=0.5,
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xy_scale=1,
                 wh_scale=1,
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
        self.lambda_coord_xy = xy_scale
        self.lambda_coord_wh = wh_scale
        self.lambda_class = class_scale
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
        anchors_lay = self.pattern_array.index(shape_stand[1])
        anchors_current = tf.constant(self.anchor_array[anchors_lay], dtype=float)

        # Step 1 reshape y_preds from [b,13,13,anchors*25] to [b,13,13,anchors,25]
        y_pred = tf.reshape(y_pred, shape_stand)

        # Step 2 get object mask from true
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # Step 3 adjust pred tensor to [bx, by, bw, bh] and get ignore mask by iou
        y_pred_coord = self.convert_coord_to_bbox(y_pred[..., 0:4], anchors_current, shape_stand)
        y_true_coord = self.convert_coord_to_bbox(y_true[..., 0:4], anchors_current, shape_stand)
        iou = get_tf_iou(y_true_coord, y_pred_coord)
        ignore_mask = tf.cast(iou < self.ignore_thresh, tf.float32)

        # Step 4 cal 3 part loss
        loss_coord = self.get_coordinate_loss(y_true[..., 0:4],
                                              y_pred[..., 0:4],
                                              object_mask,
                                              self.lambda_coord_xy,
                                              self.lambda_coord_wh)
        print("loss_box:\t{}".format(loss_coord))

        loss_confidence = self.get_confidence_loss(y_true[..., 4],
                                                   y_pred[..., 4],
                                                   object_mask,
                                                   ignore_mask,
                                                   self.lambda_object,
                                                   self.lambda_no_object)
        print("loss_confidence:\t{}".format(loss_confidence))

        loss_class = self.get_class_loss_original(y_true[..., 5:],
                                         y_pred[..., 5:],
                                         object_mask)
        print("loss_class:\t{}".format(loss_class))

        return loss_class

    @staticmethod
    def convert_coord_to_bbox(coord, anchors, input_shape):
        t_xy = coord[..., 0:2]
        t_wh = coord[..., 2:4]
        grid_xy_offset = create_grid_xy_offset(*input_shape[0:4])
        anchor_grid = create_mesh_anchor(anchors, input_shape)
        b_xy = grid_xy_offset + tf.sigmoid(t_xy)
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
        confidence_mask_ = tf.squeeze(object_mask, axis=-1)
        confidence_loss = lambda_object * confidence_mask_ * tf.square(confidence_pred - confidence_truth) + \
                          lambda_no_object * (1 - confidence_mask_) * ignore_mask * confidence_pred
        return tf.reduce_sum(confidence_loss)

    @staticmethod
    def get_coordinate_loss(coordinate_truth,
                            coordinate_pred,
                            object_mask,
                            lambda_xy=1,
                            lambda_wh=1):
        """
        计算有目标检测框的坐标损失
        @param coordinate_truth:
        @param coordinate_pred:
        @param object_mask:
        @param lambda_xy:
        @param lambda_wh:
        @return:
        """
        coord_mask = tf.expand_dims(object_mask, 4)
        coord_delta = coord_mask * tf.square(coordinate_pred[..., 0:4] - coordinate_truth[..., 0:4])
        xy_delta = lambda_xy * tf.reduce_sum(coord_delta[..., 0:2])
        wh_delta = lambda_wh * tf.reduce_sum(coord_delta[..., 2:4])
        return xy_delta + wh_delta

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
        class_truth = tf.cast(class_truth, tf.int64)
        loss_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(class_truth, class_pred)
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

    @staticmethod
    def adjust_y_pred(y_pred, input_shape, anchors):
        """
        Convert pred [tx, ty, tw, th]=> [bx, by, bw, bh]
        sigmoid(confidence), sigmoid confidence here to cal loss
        """
        t_xy = y_pred[..., 0:2]
        t_wh = y_pred[..., 2:4]
        pred_confidence = y_pred[..., 5]

        grid_offset_xy = create_grid_xy_offset(*input_shape[0:4])
        grid_anchor = create_mesh_anchor(anchors, input_shape)

        b_xy = tf.sigmoid(t_xy) + grid_offset_xy
        # b_wh = grid_anchor * tf.exp(t_wh)
        b_wh = t_wh
        confidence = tf.sigmoid(pred_confidence)
        classes = y_pred[..., 5:]
        return tf.concat([b_xy, b_wh, tf.expand_dims(confidence, axis=-1), classes], axis=-1)

    @staticmethod
    def adjust_y_true(y_true):
        xywh_confidence = y_true[..., :5]
        true_class = tf.argmax(y_true[..., 5:], -1)
        trues = tf.concat([xywh_confidence, tf.expand_dims(tf.cast(true_class, tf.float32), -1)], axis=-1)
        return trues


if __name__ == "__main__":
    from DataSet.batch_generator import *

    config_file = r"..\config\pascal_voc.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    train_generator = BatchGenerator(model_cfg, train_cfg, True)
    print(train_generator)
    x, y = train_generator.get_next_batch()
    print(x,y)
    y_true_test = np.ones((train_cfg.batch_size, 1, 1, 3, 25)).astype(np.float32)
    y_pred_test = np.ones((train_cfg.batch_size, 1, 1, 75)).astype(np.float32)

    test_loss = LossYolo3(model_cfg.input_size, train_cfg.batch_size,
                          model_cfg.anchor_array, [1, 2, 3]).call(y_true_test, y_pred_test)
    print("Sum Loss:\t{}".format(test_loss))
