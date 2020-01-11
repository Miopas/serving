import tensorflow as tf
import numpy as np


class SVM(object):
    """
    """

    def __init__(
      self, sequence_length, num_classes, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.compat.v1.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.compat.v1.get_variable(
                "W",
                shape=[sequence_length, num_classes],
                initializer=tf.keras.initializers.GlorotUniform())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.input_x, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            #delta = 1.0
            #self.loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, self.scores - self.input_y + delta), 1)) - delta \
            #                + l2_reg_lambda * l2_loss

            # Use squared-hinge
            #self.loss = tf.reduce_mean(tf.keras.losses.squared_hinge(self.scores, self.input_y)) + l2_reg_lambda * l2_loss
            self.loss = tf.reduce_mean(tf.keras.losses.squared_hinge(self.scores, self.input_y))


        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
