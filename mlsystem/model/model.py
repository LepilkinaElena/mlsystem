import tensorflow as tf

import mlsystem.dataset.dataset as dataset

from optparse import OptionParser, OptionGroup

def model(datasets, classes_num):
    features_size = 7
    x = tf.placeholder(tf.float32, [None, features_size])
    W = tf.Variable(tf.zeros([features_size, classes_num]))
    b = tf.Variable(tf.zeros([classes_num]))
    y = tf.matmul(x, W) + b

    # Define loss function.
    y_ = tf.placeholder(tf.float32, [None, classes_num])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #cross_entropy = tf.reduce_mean(tf.square(y - y_))
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train.
    
    #for _ in range(1000):
    batch = datasets.train.next_batch(100)
    while batch.size():
        #print(batch.features)
            #print()
        v = sess.run(train_step, feed_dict={x: batch.features, y_:batch.labels})
        batch = datasets.train.next_batch(100)
            #curr_W, curr_b, curr_loss =  sess.run([W, b, cross_entropy], {x:batch.features, y_:batch.labels})
            #print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

    print(W.eval())

    # Test model.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accurancy, feed_dict={x: datasets.test.features,
                                         y_: datasets.test.labels}))