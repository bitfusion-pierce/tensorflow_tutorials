import tensorflow as tf
from sklearn import metrics
import numpy as np
import cPickle
import gzip

tf.logging.set_verbosity(tf.logging.INFO)

# Read Data
# Use this when lecunn's website is back up
# mnist = learn.datasets.load_dataset('mnist')

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)

print np.shape(test_set[0])
print np.shape(test_set[1])

f.close()

# Model Training
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    valid_set[0],
    valid_set[1],
    every_n_steps=500)


feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(train_set[0])
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[512, 128],
    n_classes=10,
    model_dir="./output",
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10)
)

classifier.fit(x=train_set[0],
               y=train_set[1],
               batch_size=100,
               steps=10000,
               monitors=[validation_monitor])

# Model Testing
score = metrics.accuracy_score(test_set[1],
                               list(classifier.predict(test_set[0])))
print('Accuracy: {0:f}'.format(score))
