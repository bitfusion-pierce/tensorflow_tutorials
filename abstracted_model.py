import tensorflow as tf
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import tensorflow.contrib.metrics as tfmetrics
import cPickle
import gzip

# Set logging for removal of SKLearn compatibility issues.
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

# tf.logging.set_verbosity(tf.logging.INFO)

# Read Data
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
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
score = classifier.evaluate(x=test_set[0], y=test_set[1])

print('Accuracy: {0:f}'.format(score['accuracy']))
