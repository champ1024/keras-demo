import tensorflow as tf


def action():
    t = tf.constant([[1, 2, 3], [6, 7, 9]])
    print(t)
    print(t.shape)
    t1 = tf.transpose(t)
    print(t1)
