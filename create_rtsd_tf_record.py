import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import logging
import PIL.Image
import os
import io
import hashlib

sys.path.append('/home/ubuntu/tensorflow/models')
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from label_map import create_label_map

flags = tf.app.flags
flags.DEFINE_string('train_output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('val_output_path', '', 'Path to output TFRecord')
flags.DEFINE_float('threshold', 0.1, 'threshold for splitting dataset on val and test')
flags.DEFINE_string('data_dir', '', 'Path to input data file')
flags.DEFINE_string('label_path', '', 'Path to new label file')
flags.DEFINE_string('image_dir', '/dev/shm/data/rtsd-frames/', 'Path to images dir')
flags.DEFINE_bool('split', True, 'Split on test and train')
flags.DEFINE_bool('debug', True, 'Output info')
FLAGS = flags.FLAGS


def date_to_tf_file(data, label_map_dict, image_dir):
    image_path = os.path.join(image_dir, data['filename'].iloc[0])
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG, check it pls')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    width, height = image.size
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
   
    for idx in data.index:
        xmin.append(float(data.loc[idx].x_from) / width)
        ymin.append(float(data.loc[idx].y_from) / height)
        xmax.append(float(xmin[-1] + data.loc[idx].width) / width)
        ymax.append(float(ymin[-1] + data.loc[idx].height) / height)
        classes_text.append(data.loc[idx].sign_class.encode('utf-8'))
        classes.append(label_map_dict[data.loc[idx].sign_class])
    
    tf_dat = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(data['filename'].iloc[0].encode('utf-8')),
      'image/source_id': dataset_util.bytes_feature(data['filename'].iloc[0].encode('utf-8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf-8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(b'jpg'),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))
    return tf_dat


def create_tf_record(output_filename, label_map_dict, data):
    writer = tf.python_io.TFRecordWriter(output_filename)
    all_files = data.filename.unique().shape[0]
    for idx, image in enumerate(data.filename.unique()):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, all_files)
        tf_file = date_to_tf_file(data[data.filename == image], label_map_dict, FLAGS.image_dir)
        writer.write(tf_file.SerializeToString())
        
    writer.close()
        


def split_on_train_val(data, signs, percent, threshold=4):
    logging.info('Split dataset on train and test')
    val = []
    train = []
    for sign in signs:
        tr = data[data.sign_class == sign]
        if tr.shape[0] < threshold:
            train.append(tr)
        else:
            val_mask = np.random.rand(tr.shape[0]) < percent
            if tr.iloc[val_mask].shape[0] == 0:
                val_mask = [0]
            elif tr.shape[0]/tr.iloc[val_mask].shape[0] <= 2:
                val_mask = [i for i in range(round(tr.shape[0]/2) - 1)]
            val.append(tr.iloc[val_mask])
            tr = tr.drop(tr.iloc[val_mask].index)
            train.append(tr)
            if tr.shape[0] < 2000 or val[-1].shape[0] < 1000:
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
    label_map_dict = label_map_util.get_label_map_dict(label_path)
    
    train, val = split_on_train_val(data, signs, FLAGS.threshold)
    logging.info('%d training and %d validation sets', train.shape[0], val.shape[0])
    
    logging.info('Starting write files')
    create_tf_record(FLAGS.train_output_path, label_map_dict, train)
    create_tf_record(FLAGS.val_output_path, label_map_dict, val)
    logging.info('Finished')


if __name__ == '__main__':
    tf.app.run()