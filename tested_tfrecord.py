import tensorflow as tf
import io
from skimage import io as sk_io
import pandas as pd
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('path_tfrecord', '/home/ubuntu/rtsd/data/train.tfrecord', 'Path to output TFRecord')
flags.DEFINE_string('path_to_csv', '/home/ubuntu/rtsd/train.csv', 'Path to .csv file')
flags.DEFINE_integer('step', 100, 'On how much steps you will compare images')
flags.DEFINE_string('path_to_data', '/dev/shm/data/rtsd-frames/', 'Directory with your photos')
FLAGS = flags.FLAGS


def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


def main(_):
    it = 0
    data = pd.read_csv(FLAGS.path_to_csv)
    example = tf.train.Example()
    path_to_data = FLAGS.path_to_data
    mean_mse = []
    for record in tf.python_io.tf_record_iterator(FLAGS.path_tfrecord):
        it += 1
        if it % FLAGS.step == 0:
            example.ParseFromString(record)
            f = example.features.feature
            part = data[data.filename == str(f['image/filename'].bytes_list.value[0].decode('utf-8'))]
            imageA = sk_io.imread(path_to_data + part['filename'].iloc[0])
            imageB = sk_io.imread(io.BytesIO(f['image/encoded'].bytes_list.value[0]))
            mean_mse.append(mse(imageA, imageB))                                                          
            print('mse on image - {}  :  {}'.format(it, mean_mse[-1]))
    print('mean MSE on images -  {}'.format(np.array(mean_mse).mean()))

    
if __name__ == '__main__':
    tf.app.run()