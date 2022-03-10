import tensorflow as tf

"""
This is module for help to cal yolo loss
"""


def create_grid_xy_offset(pred_shape):
    """
    create
    :param pred_shape:
    :return: return shape is [batch_size,grid_h,grid_w,n_box,2], 2 means: x offset,y offset
    """
    batch_size, grid_h, grid_w, n_box = pred_shape[0:4]

    basic = tf.reshape(tf.tile(tf.range(grid_h), [grid_w]), (1, grid_h, grid_w, 1, 1))

    mesh_x = tf.cast(basic, tf.float32)
    mesh_y = tf.transpose(mesh_x, (0, 2, 1, 3, 4))

    mesh_xy = tf.concat([mesh_x, mesh_y], -1)

    return tf.tile(mesh_xy, [batch_size, 1, 1, n_box, 1])


def create_mesh_anchor(pred_shape, anchors):
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
