from tensorflow.keras.losses import Loss


class LossYolo4(Loss):
    def __init__(self,
                 image_size,
                 batch_size,
                 anchor_array,
                 pattern_array,
                 iou_ignore_thresh=0.5,
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
        mask_object = tf.expand_dims(y_true[..., 4], 4)

        # Step 3 convert pred coord to bounding box coord
        pred_b_xy = tf.sigmoid(y_pred[..., 0:2]) + create_grid_xy_offset(*shape_stand[0:4])
        pred_b_wh = tf.exp(y_pred[..., 2:4]) * create_mesh_anchor(shape_stand, anchors_current)

        # Step 3 convert true coord to bounding box coord
        true_b_xy = y_true[..., 0:2]
        true_b_wh = tf.exp(y_true[..., 2:4]) * anchors_current

        # Step 3 adjust pred tensor to [bx, by, bw, bh] and get ignore mask by iou
        pred_b_coord = tf.concat([pred_b_xy, pred_b_wh], axis=-1)
        true_b_coord = tf.concat([true_b_xy, true_b_wh], axis=-1)
        iou = iou_module.get_tf_iou(pred_b_coord, true_b_coord)
        mask_background = (1 - mask_object) * tf.cast(iou < self.ignore_thresh, tf.float32)

        # Step 4 cal 3 part loss
        ciou = iou_module.get_tf_ciou(pred_b_coord, true_b_coord)
        bbox_loss_scale = 2 - 1.0 * y_true[..., 2:3] * y_true[..., 3:4] / self.image_size ** 2
        loss_ciou = mask_object * bbox_loss_scale * (1 - ciou)

        loss_class = mask_object * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                                           logits=class_pred[..., 5:])

        pos_loss = mask_object * (0 - tf.log(y_pred + tf.exp))
        neg_loss = mask_background * (0 - tf.log(1 - y_pred + tf.exp))
        loss_confidence = pos_loss + neg_loss

        return loss_ciou + loss_confidence + loss_class
