import tensorflow as tf

import mlsystem.dataset.dataset as dataset

from optparse import OptionParser, OptionGroup




def model(datasets, classes_num):
    features_size = 7


    def input_train():
        features = {k: tf.constant(batch.features[k]) for k in batch.used_features}
        labels = tf.constant(batch.labels)
        return features, labels

    def input_test():
        features = {k: tf.constant(datasets.test.features[k]) for k in datasets.test.used_features}
        labels = tf.constant(datasets.test.labels)
        return features, labels

    features_columns = [tf.contrib.layers.real_valued_column(name) for name in datasets.test.used_features]
    classifier = tf.contrib.learn.DNNLinearCombinedClassifier(linear_feature_columns=features_columns,
                                                              dnn_feature_columns=features_columns,
                                                dnn_hidden_units=[100, 50], fix_global_step_increment_bug=True)
    batch = datasets.train.next_batch(5000, True)

    #while batch.size():
    classifier.fit(input_fn=input_train, steps=5000)
    #batch = datasets.train.next_batch(150, True)

    accurancy = classifier.evaluate(input_fn=input_test, steps=1)

    print(accurancy)