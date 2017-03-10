import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import tensorflow.contrib.metrics as tfmetrics
import cPickle
import gzip

from logging import StreamHandler, INFO, getLogger

logger = getLogger('tensorflow')
logger.removeHandler(logger.handlers[0])

logger.setLevel(INFO)


class DebugFileHandler(StreamHandler):
    def __init__(self):
        StreamHandler.__init__(self)

    def emit(self, record):
        if not record.levelno == INFO:
            return
        StreamHandler.emit(self, record)

logger.addHandler(DebugFileHandler())

# Read Data
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


# Model Definition
def fully_connected_model(features, labels):
    features = layers.flatten(features)
    labels = tf.one_hot(tf.cast(labels, tf.int32), 10, 1, 0)

    layer1 = layers.fully_connected(features, 512, activation_fn=tf.nn.relu, scope='fc1')
    layer2 = layers.fully_connected(layer1, 128, activation_fn=tf.nn.relu, scope='fc2')
    logits = layers.fully_connected(layer2, 10, activation_fn=None, scope='out')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='SGD',
        learning_rate=0.01
    )

    return tf.argmax(logits, 1), loss, train_op

# Model Training
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    x=valid_set[0],
    y=valid_set[1],
    metrics={'accuracy': MetricSpec(tfmetrics.streaming_accuracy)},
    every_n_steps=500)

classifier = learn.Estimator(model_fn=fully_connected_model,
                       model_dir="./output",
                       config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))

classifier.fit(x=train_set[0],
               y=train_set[1],
               batch_size=100,
               steps=10000,
               monitors=[validation_monitor])

# Model Testing
score = classifier.evaluate(x=test_set[0], y=test_set[1],
                            metrics={'accuracy': MetricSpec(tfmetrics.streaming_accuracy)})
print('Accuracy: {0:f}'.format(score['accuracy']))
