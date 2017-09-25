import numpy as np
import os
import tensorflow as tf
import sys
import pandas as pd

from collections import defaultdict
from io import StringIO
from PIL import Image
from tqdm import tqdm

sys.path.append("/home/artem/.virtualenvs/cv/lib/python3.5/site-packages/tensorflow/models/")

flags = tf.app.flags
flags.DEFINE_string('val_path', '/home/artem/RTSD/RTSD/val.csv', 'Path to val.csv')
flags.DEFINE_string('weights', '/home/artem/RTSD/RTSD/frozen4/frozen_inference_graph.pb', 'Path to weights')
flags.DEFINE_string('label', '/home/artem/RTSD/data2/label_map.pbtxt', 'Path to label_map.pbtxt')
flags.DEFINE_string('path_to_data', '/home/artem/RTSD/data/rtsd-frames/', 'Path to data dic with frames')
flags.DEFINE_string('data', '/home/artem/RTSD/data/full-gt.csv', 'Path to data.csv')
flags.DEFINE_integer('num_class', 1, 'num of classes')
flags.DEFINE_string('output', '', 'Path to output.csv')
FLAGS = flags.FLAGS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def bb_intersection_over_union(boxA, boxB, w, h):
    xA = max(boxA[0]*w, boxB[0])
    yA = max(boxA[1]*h, boxB[1])
    xB = min(boxA[2]*w, boxB[2])
    yB = min(boxA[3]*h, boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)

    boxAArea = (boxA[2]*w - boxA[0]*w + 1) * (boxA[3]*h - boxA[1]*h + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def get_stat(bA, bB, w, h, score):
    y_scores = []
    y_pred = []
    y_real = []
    infB = []
    
    boxB = []
    for i in range(bB.shape[0]):
        boxB.append([ bB.iloc[i].y_from, bB.iloc[i].x_from, bB.iloc[i].y_from+bB.iloc[i].height, 
                     bB.iloc[i].x_from+bB.iloc[i].width])
    mask = np.ones(len(boxB), dtype=bool)
    it = 0
    filename = bB.iloc[0].filename
    for boxA in bA:
        flag = True
        for i in range(len(boxB)):
            if (bb_intersection_over_union(boxA, boxB[i], w, h) >= 0.3) and (mask[i] == True):
                flag = False
                if (bB.iloc[i].width <= 24) and (bB.iloc[i].height <= 24):
                    infB.append(['small', filename])
                else:
                    infB.append(['large', filename])
                y_pred.append(int(bB.iloc[i].sign_class[0]))
                y_real.append(int(bB.iloc[i].sign_class[0]))
                y_scores.append(score[it])
                mask[i] = False
            
        if flag == True:
            if (abs(boxA[0]-boxA[2])*w <= 24) and (abs(boxA[1]-boxA[3])*h <= 24):
                infB.append(['small', filename])
            else:
                infB.append(['large', filename])
            y_pred.append(-1)
            y_real.append(0)
            y_scores.append(score[it])
        it += 1
    for j in range(mask.shape[0]):
        if mask[j] == True:
            if (bB.iloc[j].width <= 24) and (bB.iloc[j].height <= 24):
                infB.append(['small', filename])
            else:
                infB.append(['large', filename])
            y_pred.append(0)
            y_real.append(int(bB.iloc[j].sign_class[0]))
            y_scores.append(0)
    return y_scores, y_pred, y_real, infB


def predict_on_val(path_to_ckpt, path_to_files, TEST_IMAGE_PATHS, data, output):
    y_scores = []
    y_pred = []
    y_real = []
    infB = []
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for i in tqdm(range(TEST_IMAGE_PATHS.shape[0])):
                image = Image.open(path_to_files + TEST_IMAGE_PATHS[i])
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                cur_scores, cur_y_pred, cur_y_real, cur_infB = get_stat(boxes[scores > 0.0001], 
                                                                   data[data.filename == TEST_IMAGE_PATHS[i]],
                                                                        image_np.shape[0], image_np.shape[1], 
                                                                        scores[0])
                y_scores += cur_scores
                y_pred += cur_y_pred
                y_real += cur_y_real
                infB += cur_infB
    pd.DataFrame(np.hstack((np.array(y_pred).reshape(-1, 1), np.array(y_real).reshape(-1, 1), 
                            np.array(y_scores).reshape(-1, 1), np.array(infB))), 
                            columns=['pred', 'real', 'scores', 'size', 'filename']).to_csv(output)
    

def main(_):
    val = pd.read_csv(FLAGS.val_path)
    TEST_IMAGE_PATHS = val.filename
    data = pd.read_csv(FLAGS.data)
    predict_on_val(FLAGS.weights, FLAGS.path_to_data, TEST_IMAGE_PATHS, data, FLAGS.output)


if __name__ == '__main__':
    tf.app.run()