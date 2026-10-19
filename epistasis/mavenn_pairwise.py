from mavenn.src.layers.gpmap import PairwiseGPMapLayer
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)


class BilinearPairwiseGPMapLayer(PairwiseGPMapLayer):
    def __init__(self, L, alphabet, theta_regularization, **kwargs):
        super().__init__(mask_type="pairwise", L=L, alphabet=alphabet,
                         theta_regularization=theta_regularization, **kwargs)

    def call(self, x_lc):
        width = self.L * self.C
        x = tf.reshape(x_lc, [-1, width])
        single = tf.reshape(self.theta_lc, [width, 1])
        pair = tf.reshape(self.theta_lclc * self.mask_dict["theta_lclc"], [width, width])
        return self.theta_0 + tf.matmul(x, single) + tf.reduce_sum(tf.matmul(x, pair) * x, axis=1, keepdims=True)
