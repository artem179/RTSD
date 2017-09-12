import tensorflow as tf
import io
import skimage
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('path_tfrecord', '/home/ubuntu/rtsd/data/train.tfrecord', 'Path to output TFRecord')
flags.DEFINE_string('path_to_csv', '/home/ubuntu/rtsd/train.csv', 'Path to .csv file')
flags.DEFINE_int('step', 100, 'On how much steps you will compare images')
FLAGS = flags.FLAGS

exmaple = tf.train.Example()

def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])
    return err


def main(_):
    it = 0
    data = pd.read_csv(FLAGS.path_to_csv)
    
    for record in tf.python_io.tf_record_iterator(FLAGS.path_tfrecord):
        it += 1
        if it % FLAGS.step:
            example.ParseFromString(record)
            f = exmaple.features.feature
            part = data[data.filename == str(f['image/filename'].bytes_list.value[0])]
            imageA = skimage.io.imread(part['filename'].iloc[0])
            imageB = skimage.io.imread(io.BytesIO(f['image/encoded'].bytes_list.value[0]))
            print('mse on image - {}  :  {}'.format(mse(imageA, imageB)))
                        

if __name__ == '__main__':
    tf.app.run()