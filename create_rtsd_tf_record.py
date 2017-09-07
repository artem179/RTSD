from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import pandas as pd

sys.path.append('/home/ubuntu/tensorflow/models')
from object_detection.utils import dataset_util
from label_map import create_label_map

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_path', '', 'Path to input data file')
flags.DEFINE_string('label_path', '', 'Path to new label file')
FLAGS = flags.FLAGS


def main(_):
    data = pd.read_csv(FLAGS.input_path)
    signs = data.sign_class.unique()
    label_path = create_label_map(signs, FLAGS.label_path)


if __name__ == '__main__':
    tf.app.run()