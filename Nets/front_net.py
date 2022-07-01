import cv2
import tensorflow as tf
from Utils.bound_box import BoundBox, nms_boxes
from Utils.convert import *

IDX_X = 0
IDX_Y = 1
IDX_W = 2
IDX_H = 3
IDX_OBJECT_NESS = 4
IDX_CLASS_PROB = 5


# 输出预测
class YoloDetector(object):

    def __init__(self, model, anchors, image_size=416, obj_thresh=0.5, class_thresh=0.5, nms_thresh=0.5):
        self.yolo_net = model
        self.anchors = anchors
        self.image_size = image_size
        self.obj_thresh = obj_thresh
        self.class_thresh = class_thresh
        self.nms_thresh = nms_thresh

    def detect_from_file(self, image_source):
        image = cv2.imread(image_source)
        image = np.array(image)

        image_h, image_w, _ = image.shape

        # 1. Convert To =>[1,416,416,3]
        preprocess_img = cv2.resize(image / 255., (self.image_size, self.image_size))
        new_image = np.expand_dims(preprocess_img, axis=0)

        # 2. Predict Yolo OutPut [1,52,52,3,25],[1,26,26,3,25],[1,13,13,3,25]
        yolo_res = self.yolo_net.predict(new_image)

        # 3. Decode Boxes
        all_filter_out = []

        for i, yolo in enumerate(yolo_res):
            anchors = self.anchors[i]
            decode_out = self.decode(yolo, anchors)
            mask = tf.cast((tf.cast(decode_out[..., 4] > self.obj_thresh, float)) * (
                tf.cast(decode_out[..., 5] >= self.class_thresh, float)), bool)
            filter_out = tf.boolean_mask(decode_out, mask)
            if filter_out.shape[0] != 0:
                all_filter_out.append(filter_out)
        if len(all_filter_out) == 0:
            return []
        all_prob = np.vstack(all_filter_out)

        # 4. correct_yolo_boxes
        all_prob[..., 0] = all_prob[..., 0] * image_w
        all_prob[..., 2] = all_prob[..., 2] * image_w
        all_prob[..., 1] = all_prob[..., 1] * image_h
        all_prob[..., 3] = all_prob[..., 3] * image_h

        # 5. nms boxes

        bbox = all_prob[..., 0:4]
        scores = all_prob[..., 4] * all_prob[..., 5]
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(bbox,
                                                                                     scores,
                                                                                     100,
                                                                                     iou_threshold=self.nms_thresh)
        count_before_nms = all_prob.shape[0]
        count_after_nms = len(selected_indices)
        print("{} non max suppression => {}".format(count_before_nms, count_after_nms))
        # 6. Convert to Boxes
        boxes = []
        for i in selected_indices:
            box = BoundBox(all_prob[i][0],
                           all_prob[i][1],
                           all_prob[i][2],
                           all_prob[i][3],
                           all_prob[i][4],
                           all_prob[i][5],
                           all_prob[i][6])
            print(box)
            print(tf.reduce_sum(all_prob[i][7:]))
            boxes.append(box)

        # 6. return boxes
        return boxes

    @staticmethod
    def create_mesh_grid(pattern):
        basic = tf.reshape(tf.tile(tf.range(pattern), [pattern]), (pattern, pattern, 1, 1))
        mesh_x = tf.cast(basic, tf.float32)
        mesh_y = tf.transpose(mesh_x, (1, 0, 2, 3))
        mesh_xy = tf.concat([mesh_x, mesh_y], -1)
        return tf.tile(mesh_xy, [1, 1, 3, 1])

    @staticmethod
    def create_mesh_anchor(pattern, anchors):
        """
        # Returns
            mesh_anchor : Tensor, shape of (batch_size, grid_h, grid_w, n_box, 2)
                [..., 0] means "anchor_w"
                [..., 1] means "anchor_h"
        """
        anchor_list = tf.reshape(anchors, shape=[6])
        grid_h, grid_w, n_box = pattern, pattern, 3
        mesh_anchor = tf.tile(anchor_list, [grid_h * grid_w])
        mesh_anchor = tf.reshape(mesh_anchor, [grid_h, grid_w, n_box, 2])
        mesh_anchor = tf.cast(mesh_anchor, tf.float32)
        return mesh_anchor

    # 从单尺度 Y_Pre 中解码 并抑制 得到Boxes
    def decode(self, yolo_single_scale, anchors, nb_box=3):
        """
        # Args
            netout : (n_rows, n_cols, 3, 4+1+n_classes)
            anchors

        """
        pattern = yolo_single_scale.shape[1]
        yolo = yolo_single_scale.reshape((pattern, pattern, 3, -1))

        t_xy = yolo[..., 0:2]
        t_wh = yolo[..., 2:4]
        b_xy = (tf.sigmoid(t_xy) + self.create_mesh_grid(pattern)) / pattern
        b_wh = tf.exp(t_wh) * self.create_mesh_anchor(pattern, anchors) / self.image_size

        objectness_prob = tf.sigmoid(yolo[..., 4:5])
        classes_probs = tf.sigmoid(yolo[..., 5:])

        label_index = tf.expand_dims(tf.cast(tf.argmax(classes_probs, axis=-1), tf.float32), axis=-1)
        class_prob = tf.expand_dims(tf.reduce_max(classes_probs, axis=-1), axis=-1)
        decode_res = tf.concat([b_xy, b_wh, objectness_prob, class_prob, label_index, classes_probs], axis=-1)

        return decode_res

    @staticmethod
    def decode_coordinate(netout, row, col, b, anchors):
        x, y, w, h = netout[row, col, b, :IDX_H + 1]

        x = col + tf.sigmoid(x)
        y = row + tf.sigmoid(y)
        w = anchors[0] * tf.exp(w)
        h = anchors[1] * tf.exp(h)

        return x, y, w, h

    @staticmethod
    def activate_probs(objectness, classes, obj_thresh=0.3):
        """
        # Args
            objectness : scalar
            classes : (n_classes, )

        # Returns
            objectness_prob : (n_rows, n_cols, n_box)
            classes_conditional_probs : (n_rows, n_cols, n_box, n_classes)
        """
        # 1. sigmoid activation
        objectness_prob = sigmoid(objectness)

        classes_probs = sigmoid(classes)

        # 2. conditional probability
        classes_conditional_probs = classes_probs * objectness_prob

        # 3. thresholding
        classes_conditional_probs *= objectness_prob > obj_thresh
        return objectness_prob, classes_conditional_probs


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


if __name__ == "__main__":
    pass
