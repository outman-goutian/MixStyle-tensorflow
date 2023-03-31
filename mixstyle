from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp
import random
import numpy as np
class Mixstyle(layers.Layer):
    def __init__(self,a,b,prob=0.5, alpha=0.1, eps=1e-6, **kwargs):
        super(Mixstyle, self).__init__(**kwargs)
        self.prob = prob
        self.alpha = alpha
        self.eps = eps

    def build(self, input_shape):
        self.built = True

    def call(self, x, training=None):
        if np.random.rand() > self.prob:
            return x
        batch_size = tf.shape(x)[0]
        f_mu = tf.reduce_mean(x, axis=[1,2], keepdims=True)    
        f_var = tf.math.reduce_variance(x, axis=[1,2], keepdims=True)  
        f_sig = tf.math.sqrt(f_var + self.eps)  
        f_mu, f_sig = tf.stop_gradient(f_mu), tf.stop_gradient(f_sig)  # block gradients
        x_normed = (x - f_mu) / f_sig  # normalize input 
        lmda = tfp.distributions.Beta(self.alpha, self.alpha).sample((batch_size, 1, 1, 1))  # sample instance-wise convex weights
        perm = tf.linspace(0, batch_size - 1, batch_size) 
        perm = tf.random.shuffle(perm)      
        perm = tf.dtypes.cast(perm, tf.int32)
        f_mu_perm, f_sig_perm = tf.gather(f_mu, axis=0, indices=perm), tf.gather(f_var, axis=0, indices=perm)
        mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
        sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
        mix_x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed statisticsstics

        return K.in_train_phase(mix_x, x, training=training)

    def get_config(self):
        config = {
            'prob': self.prob,
            'alpha': self.alpha,
            'eps' : self.eps,
        }
        base_config = super(Mixstyle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

