import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import logging

sys.path.append('/home/ubuntu/tensorflow/models')
from object_detection.utils import dataset_util
from label_map import create_label_map

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_float('threshold', 0.1, 'threshold for splitting dataset on val and test')
flags.DEFINE_string('data_dir', '', 'Path to input data file')
flags.DEFINE_string('label_path', '', 'Path to new label file')
flags.DEFINE_bool('split', True, 'Split on test and train')
flags.DEFINE_bool('debug', True, 'Output info')
FLAGS = flags.FLAGS


def split_on_train_val(data, signs, percent, threshold=3):
    logging.info('Split dataset on train and test')
    val = []
    train = []
    for sign in signs:
        tr = data[data.sign_class == sign]
        if tr.shape[0] < threshold:
            train.append(tr)
        else:
            val_mask = np.random.rand(tr.shape[0]) < percent
            if val_mask.shape[0] == 0:
                val_mask = [0]
            elif tr.shape[0]/val_mask.shape[0] <= 2:
                val_mask = [i for i in range(round(tr.shape[0]/2) - 1)]
            val.append(tr.iloc[val_mask])
            tr = tr.drop(tr.iloc[val_mask].index)
            train.append(tr)
            if tr.shape[0] < 20 or val[-1].shape[0] < 10:
                logging.info('Small dataset for name sign : %s\ntrain - %d imgs \ntest - %d imgs' % 
                    (sign, tr.shape[0], val[-1].shape[0]))
    logging.info('Creating train.csv test.csv')
    train = pd.concat(train).reset_index(drop=True)
    val = pd.concat(val).reset_index(drop=True)
    train.to_csv('train.csv')
    val.to_csv('val.csv')
    return train, val


def main(_):
    if FLAGS.debug:
        logging.getLogger().setLevel(logging.INFO)
    logging.info('Creating label_map.pbtxt')
    data = pd.read_csv(FLAGS.data_dir)
    signs = data.sign_class.unique()
    label_path = create_label_map(signs, FLAGS.label_path)
    
    train, val = split_on_train_val(data, signs, FLAGS.threshold)
    logging.info('Finished')

if __name__ == '__main__':
    tf.app.run()