import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


# define flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('n_hidden1', 16,
                            """Number of nodes in first hidden layer.""")
tf.app.flags.DEFINE_integer('n_hidden2', 16,
                            """Number of nodes in first hidden layer.""")
tf.app.flags.DEFINE_boolean('dropout', True,
                            """Whether to implement dropout.""")
tf.app.flags.DEFINE_float('dropout_prob', 0.10,
                          """Dropout probability.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          """Number of batches to run.""")
tf.app.flags.DEFINE_integer('n_epochs', 70,
                            """Number of epochs to train for.""")
tf.app.flags.DEFINE_boolean('train', False,
                            """Train on a subset of the data.""")
tf.app.flags.DEFINE_boolean('train_all', False,
                            """Train on all the data.""")
tf.app.flags.DEFINE_boolean('infer', False,
                            """Load the new data and generate predicitions.""")


def main(argv=None):
    # load data
    X_train = np.load('./data/X_train.npy')
    y_train = np.load('./data/y_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')
    X = np.load('./data/X.npy')
    y = np.load('./data/y.npy')
    X_2018 = np.load('./data/X_2018.npy')

    # process data
    n_outputs = int(np.max(y) + 1)
    Y_train = np.eye(n_outputs)[y_train]
    Y_test = np.eye(n_outputs)[y_test]
    Y = np.eye(n_outputs)[y]

    # model settings
    n_samples, n_inputs = X_train.shape
    n_hidden1 = FLAGS.n_hidden1
    n_hidden2 = FLAGS.n_hidden2
    learning_rate = FLAGS.learning_rate
    initializer = tf.variance_scaling_initializer()

    # construct model graph
    X_placeholder = tf.placeholder(tf.float32, [None, n_inputs])
    Y_placeholder = tf.placeholder(tf.float32, [None, n_outputs])

    with tf.contrib.framework.arg_scope([fully_connected], weights_initializer=initializer):
        hidden1 = fully_connected(X_placeholder, n_hidden1)
        hidden2 = fully_connected(hidden1, n_hidden2)
        if FLAGS.dropout:
            hidden2_dropout = tf.nn.dropout(hidden2, keep_prob=FLAGS.dropout_prob)
            logits = fully_connected(hidden2_dropout, n_outputs, activation_fn=None)
        else:
            logits = fully_connected(hidden2, n_outputs, activation_fn=None)

    Y_ = tf.nn.softmax(logits)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_placeholder, logits=logits)
    mean_loss = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(Y_placeholder, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # training settings
    n_epochs = FLAGS.n_epochs

    if FLAGS.train:
        # execute graph on train and test to identify hyper parameters
        with tf.Session() as sess:
            model_path = './models/train/model.ckpt'
            sess.run(init)
            feed_dict_train = {X_placeholder: X_train, Y_placeholder: Y_train}
            feed_dict_test = {X_placeholder: X_test, Y_placeholder: Y_test}
            for n_epoch in range(n_epochs):
                _ = sess.run(train_op, feed_dict=feed_dict_train)
                loss_train_eval, accuracy_train_eval = sess.run([mean_loss, accuracy], feed_dict=feed_dict_train)
                loss_test_eval, accuracy_test_eval = sess.run([mean_loss, accuracy], feed_dict=feed_dict_test)
                print('Epoch:', n_epoch,
                      'Train Loss:', loss_train_eval, 'Train Accuracy:', accuracy_train_eval,
                      'Test Loss:', loss_test_eval, 'Test Accuracy:', accuracy_test_eval)
            saver.save(sess, model_path)

    if FLAGS.train_all:
        # execute graph on all data
        with tf.Session() as sess:
            model_path = './models/all/model.ckpt'
            sess.run(init)
            feed_dict_train = {X_placeholder: X, Y_placeholder: Y}
            for n_epoch in range(n_epochs):
                _ = sess.run(train_op, feed_dict=feed_dict_train)
                loss_train_eval, accuracy_train_eval = sess.run([mean_loss, accuracy], feed_dict=feed_dict_train)
                print('Epoch:', n_epoch, 'Train Loss:', loss_train_eval, 'Train Accuracy:', accuracy_train_eval)
            saver.save(sess, model_path)

    if FLAGS.infer:
        # execute graph on new data for inference
        with tf.Session() as sess:
            model_path = './models/all/model.ckpt'
            saver.restore(sess, model_path)
            Y_pred = sess.run(Y_, feed_dict={X_placeholder: X_2018})
        np.save('./data/Y_pred.npy', Y_pred)


if __name__ == '__main__':
    tf.app.run()
