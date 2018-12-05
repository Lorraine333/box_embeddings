import tensorflow as tf
import math


class Layers:
    """
    Helper class for quickly creating NN layers with different initializations
    """
    tf.set_random_seed(20160408)

    def W(self, input_dim=100, output_dim=100, name='W', init='Xavier', trainable=True):
        if init == 'Normal':
            return self.normal_W(input_dim, output_dim, name, trainable)
        elif init == 'Uniform':
            return self.uniform_W(input_dim, output_dim, name, trainable)
        else:
            return self.xavier_W(input_dim, output_dim, name, trainable)

    def b(self, dim=100, name='B', init='Xavier', trainable=True):
        if init == 'Normal':
            return self.normal_b(dim, name, trainable)
        elif init == 'Uniform':
            return self.uniform_b(dim, name, trainable)
        else:
            return self.xavier_b(dim, name, trainable)

    def normal_W(self, input_dim=100, output_dim=100, name='W', trainable=True):
        return tf.get_variable(name, tf.random_normal([input_dim, output_dim], stddev=1.0 / math.sqrt(input_dim), seed=12132015), trainable=trainable)

    def normal_b(self, dim=100, name='B', trainable=True):
        return tf.get_variable(name, tf.random_normal([dim], stddev=1.0 / math.sqrt(dim), seed=12132015), trainable = trainable)

    def xavier_W(self, input_dim=100, output_dim=100, name='W', trainable=True):
        init_range = math.sqrt(6.0 / (input_dim + output_dim))
        return tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, seed=20160429), name=name, trainable=trainable)

    def xavier_b(self, dim=100, name='B', trainable=True):
        init_range = math.sqrt(6.0 / dim)
        return tf.Variable(tf.random_uniform([dim], minval=-init_range, maxval=init_range, seed=20160429), name=name, trainable=trainable)

    def uniform_W(self, input_dim=100, output_dim=100, name="W", trainable=True):
        return tf.Variable(tf.random_uniform([input_dim, output_dim], minval=-0.05, maxval=0.05, seed=12132015), name=name, trainable=trainable)

    def uniform_b(self, dim=100, name="B", trainable=True):
        return tf.Variable(tf.random_uniform([dim], minval=-0.05, maxval=0.05, seed=12132015), name=name, trainable=trainable)

    def placeholder(self, dim, name):
        return tf.placeholder("float", shape=[None, dim], name=name)

    # cube feed forward init lizat
    def layer1_bias(self, dim, init, init_value = -5.0, name = 'layer1_b'):
        if init == 'norm':
            bias  = self.xavier_b(dim, name)
        elif init == 'pre':
            bias  = tf.Variable(tf.random_uniform([dim], minval=init_value, maxval=(init_value + 0.1), seed=20160429), name = name, trainable=True)
        else:
            bias = None
            print('invalid layer1 initialization')
        return bias

    def layer2_bias(self, dim, init, init_value, name = 'layer2_b'):
        if init == 'norm':
            bias = self.xavier_b(dim, name)
        elif init == 'pre':
            bias = tf.Variable(tf.random_uniform([dim], minval=float(init_value), maxval=float(init_value)+0.5, seed=20160429), name = name, trainable=True)
        else:
            bias = None
            print('invalid layer2 initialization')
        return bias