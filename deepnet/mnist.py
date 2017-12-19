
'''
input > hidden layer 1 (acvitation function) > weights hidden layer 2
(activation function) -> weights -> output layer

Compare the output to inteded output (cost or loss function(cross-entropy))
optimization function (attempt to minimiza the costF(Adam, SGD, Adagrad))
feed forward + backprop = Epoch
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network(data):


    hidden_1_layer = {'weights':tf.variable(tf.random_normal([784, n_nodes_hl1])),
               'biases':tf.variable(tf.random_normal(n_nodes_hl1))
              }

    hidden_2_layer = {'weights':tf.variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
               'biases':tf.variable(tf.random_normal(n_nodes_hl2))
              }

    hidden_3_layer = {'weights':tf.variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3 ])),
               'biases':tf.variable(tf.random_normal(n_nodes_hl3))
              }

    output_layer = {'weights':tf.variable(tf.random_normal([n_nodes_hl3, n_classes])),
               'biases':tf.variable(tf.random_normal(n_classes))
              }


    #(input_data * weights) + biases

    l1 = tf.add(matmul(data, hidden_1_layer['weights'] + hidden_1_layer['biases']))
    l1 = tf.nn.relu(l1)

    l2 = tf.add(matmul(l1, hidden_2_layer['weights'] + hidden_2_layer['biases']))
    l2 = tf.nn.relu(l2)

    l3 = tf.add(matmul(l2, hidden_3_layer['weights'] + hidden_3_layer['biases']))
    l3 = tf.nn.relu(l3)

    output = tf.add(l3, output_layer['weights'] + output_layer['biases'])
    return output
