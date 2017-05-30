import tensorflow as tf

import mlsystem.dataset.dataset as dataset

from optparse import OptionParser, OptionGroup

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Building the encoder
def encoder(x, weights, biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x,weights, biases):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

def model(datasets, classes_num):
    features_size = 7

    n_hidden_1 = 5 # 1st layer num features
    n_hidden_2 = 3

    x = tf.placeholder(tf.float32, [None, features_size])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([features_size, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, features_size])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([features_size])),
    }

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            batch = datasets.train.next_batch(batch_size)
            # Loop over all batches
            while batch.size():
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch.features})
                batch = datasets.train.next_batch(batch_size)
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Applying encode and decode over test set
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
        accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accurancy, feed_dict={X: datasets.test.features, y_: datasets.test.labels}))