from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
from six.moves import range

pickle_file= '/input/banglaIsolated_clean.pickle'

with open(pickle_file, 'rb') as f:
    save=pickle.load(f)
    train_dataset=save['train_dataset']
    train_labels=save['train_labels']
    valid_dataset=save['valid_dataset_clean']
    valid_labels=save['valid_labels']
    test_dataset=save['test_dataset_clean']
    test_labels=save['test_labels']
    del save #hint to help gc to free up memory 
    print('training set', train_dataset.shape,train_labels.shape)
    print('validation set', valid_dataset.shape, valid_labels.shape)
    print('test set', test_dataset.shape,test_labels.shape)

image_size=28
num_labels=84
num_channels=1 

def reformat(dataset, labels):
    dataset=dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
    # Map 1 to [0.0,1.0,0.0....], 2 to [0.0,0.0,1.0.....]
    labels=(np.arange(num_labels) ==labels[:,None]).astype(np.float32)
    return dataset,labels
train_dataset, train_labels= reformat(train_dataset, train_labels)
valid_dataset, valid_labels=reformat(valid_dataset, valid_labels)
test_dataset, test_labels =reformat(test_dataset, test_labels)
print( 'training set', train_dataset.shape,train_labels.shape)
print('validation set', valid_dataset.shape,valid_labels.shape)
print('test set', test_dataset.shape,test_labels.shape)

def accuracy(predictions, labels):
    return(100.0*np.sum(np.argmax(predictions, 1)==np.argmax(labels,1))/ predictions.shape[0])

log_dir='/output/inception_class84_log'
train_dir=log_dir+'/train'
test_dir=log_dir+'/test'
##print(os.listdir(train_dir))
##print(os.listdir(test_dir))

def dir_maker(path_dir):
    if(os.path.isdir(path_dir)):
        dir_list=os.listdir(path_dir)
        c=0
        for current_dir in dir_list:
            if(current_dir==str(c)):
                print(current_dir)
            c=c+1
        os.makedirs(path_dir+'/'+str(c))
        path_dir=path_dir+'/'+str(c)
        return path_dir
train_dir=dir_maker(train_dir)
test_dir=dir_maker(test_dir)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

batch_size = 50
patch_size1 = 3
patch_size2 = 5
depth = 16
depth1 = 32
depth2 = 16
depth3 = 8
concat_depth = 48
num_hidden = 800
num_hidden2 = 400
keep_prob = 0.5
decay_step = 2000
base = 0.9
graph = tf.Graph()


# Model.
def model(data, useDropout):
    with tf.name_scope('conv1'):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer1_biases)
        tf.summary.histogram('conv1_activation1', hidden)
    with tf.name_scope('conv1'):
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        max_pooled = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(max_pooled + layer2_biases)
        tf.summary.histogram('conv2_activation2', hidden)

    inception1x1_conv = tf.nn.conv2d(hidden, inception1x1_weights, [1, 1, 1, 1], padding='SAME')
    inception1x1_relu = tf.nn.relu(inception1x1_conv + inception1x1_biases)

    inception3x3_conv = tf.nn.conv2d(inception1x1_relu, inception3x3_weights, [1, 1, 1, 1], padding='SAME')
    inception3x3_relu = tf.nn.relu(inception3x3_conv + inception3x3_biases)

    inception5x5_conv = tf.nn.conv2d(inception1x1_relu, inception5x5_weights, [1, 1, 1, 1], padding='SAME')
    inception5x5_relu = tf.nn.relu(inception5x5_conv + inception5x5_biases)

    inception3x3_maxpool = tf.nn.max_pool(hidden, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
    inception1x1_post_maxpool = tf.nn.conv2d(inception3x3_maxpool, inception1x1_post_mxpool_wts, [1, 1, 1, 1], padding='SAME')
    inception1x1_post_maxpool = tf.nn.relu(inception1x1_post_maxpool + post_maxpool_biases)

    concat_filter = tf.concat([inception1x1_relu, inception3x3_relu, inception5x5_relu, inception1x1_post_maxpool],3)
    concat_maxpooled = tf.nn.max_pool(concat_filter, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    shape = concat_maxpooled.get_shape().as_list()

    reshape = tf.reshape(concat_maxpooled, [shape[0], shape[1] * shape[2] * shape[3]])

    if useDropout == 1:
        dropout_layer2 = tf.nn.dropout(tf.nn.relu(reshape), keep_prob)
    else:
        dropout_layer2 = tf.nn.relu(reshape)
    hidden = tf.nn.relu(tf.matmul(dropout_layer2, layer3_weights) + layer3_biases)

    hidden = tf.nn.relu(tf.matmul(hidden, layer4_weights) + layer4_biases)
    return tf.matmul(hidden, layer5_weights) + layer5_biases

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    with tf.name_scope('weights1'):
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, num_channels, depth], stddev=0.3))
        variable_summaries(layer1_weights)
    with tf.name_scope('biases1'):
        layer1_biases = tf.Variable(tf.zeros([depth]))
        variable_summaries(layer1_biases)

    with tf.name_scope('weights2'):
        layer2_weights = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, depth, depth1], stddev=0.05))
        variable_summaries(layer2_weights)
    with tf.name_scope('biases2'):
        layer2_biases = tf.Variable(tf.constant(0.0, shape=[depth1]))
        variable_summaries(layer2_biases)

    with tf.name_scope('weights3'):
        layer3_weights = tf.Variable(tf.truncated_normal(
            [((image_size + 3) // 4) * ((image_size + 3) // 4) * concat_depth, num_hidden], stddev=0.05))
        variable_summaries(layer3_weights)
    with tf.name_scope('biases3'):
        layer3_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))
        variable_summaries(layer3_biases)

    with tf.name_scope('weights4'):
        layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden2], stddev=0.01))
        variable_summaries(layer4_weights)
    with tf.name_scope('biases4'):
        layer4_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden2]))
        variable_summaries(layer4_biases)

    with tf.name_scope('weights5'):
        layer5_weights = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.01))
        variable_summaries(layer5_weights)
    with tf.name_scope('biases5'):
        layer5_biases = tf.Variable(tf.constant(0.0, shape=[num_labels]))
        variable_summaries(layer5_biases)


    ## Inception Module
    with tf.name_scope('inception_1x1_weights1'):
        inception1x1_weights = tf.Variable(tf.truncated_normal(
              [1, 1, depth1, depth2], stddev=0.25))
        variable_summaries(inception1x1_weights)
    with tf.name_scope('inception_1x1_biases1'):
        inception1x1_biases = tf.Variable(tf.constant(0.0, shape=[depth2]))
        variable_summaries(inception1x1_biases)

    with tf.name_scope('inception_3x3_weights2'):
        inception3x3_weights = tf.Variable(tf.truncated_normal(
              [patch_size1, patch_size1, depth2, depth3], stddev=0.05))
        variable_summaries(inception3x3_weights)
    with tf.name_scope('inception_3x3_biases2'):
        inception3x3_biases = tf.Variable(tf.constant(0.0, shape=[depth3]))
        variable_summaries(inception3x3_biases)

    with tf.name_scope('inception_5x5_weights3'):
        inception5x5_weights = tf.Variable(tf.truncated_normal(
              [patch_size2, patch_size2, depth2, depth3], stddev=0.08))
        variable_summaries(inception5x5_weights)
    with tf.name_scope('inception_5x5_biases3'):
        inception5x5_biases = tf.Variable(tf.constant(0.0, shape=[depth3]))
        variable_summaries(inception5x5_biases)

    with tf.name_scope('inception_maxPool_weights4'):
        inception1x1_post_mxpool_wts = tf.Variable(tf.truncated_normal(
              [1, 1, depth1, depth2], stddev=0.04))
        variable_summaries(inception1x1_post_mxpool_wts)
    with tf.name_scope('inception_maxPool_biases4'):
        post_maxpool_biases = tf.Variable(tf.constant(0.0, shape=[depth2]))
        variable_summaries(post_maxpool_biases)

    global_step = tf.Variable(0, trainable = False)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.005, global_step, decay_step, base)

        


    # Training computation.
    logits = model(tf_train_dataset, 1)
    with tf.name_scope('cross_entropy'):
            # The raw formulation of cross-entropy,
            #
            # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
            #                               reduction_indices=[1]))
            #
            # can be numerically unstable.
            #
            # So here we use tf.nn.softmax_cross_entropy_with_logits on the
            # raw outputs of the nn_layer above, and then average across
            # the batch.
            with tf.name_scope('total'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    tf.summary.scalar('cross_entropy', loss)

    # Optimizer.
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(model(tf_train_dataset, 0))
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 0))
    test_prediction = tf.nn.softmax(model(tf_test_dataset, 0))



num_steps = 30001

with tf.Session(graph=graph) as session:
    merged = tf.summary.merge_all()
    print(log_dir+'/train')
    train_writer = tf.summary.FileWriter(log_dir+'/train',session.graph)
    test_writer = tf.summary.FileWriter(log_dir+'/test')
    tf.global_variables_initializer().run()
    ## tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _,summary, l, predictions = session.run([optimizer,merged, loss, train_prediction], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            #print(tf.Print(layer1_weights, [layer1_weights]).eval())
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels)) 

session.close()









