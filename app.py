from flask import Flask, render_template, request

import os
import glob

import six.moves.urllib as urllib
import sys
from collections import defaultdict
from io import StringIO
from PIL import Image

import label_map_util

import visualization_utils as vis_util

import tensorflow as tf
import numpy as np

from helpers import load_image_into_numpy_array, run_inference_for_single_image

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('uploads.html')


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, "images")

    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = '/'.join([target, filename])
        print(destination)
        file.save(destination)

    PATH_TO_TEST_IMAGES_DIR = target
    pb_fname = os.path.join(APP_ROOT, 'frozen_inference_graph.pb')
    PATH_TO_CKPT = pb_fname
    PATH_TO_LABELS = os.path.join(APP_ROOT, 'label_map.pbtxt')

    assert os.path.isfile(pb_fname)
    assert os.path.isfile(PATH_TO_LABELS)
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, "*.*"))
    assert len(TEST_IMAGE_PATHS) > 0, 'No image found in `{}`.'.format(PATH_TO_TEST_IMAGES_DIR)
    print(TEST_IMAGE_PATHS)

    sys.path.append("..")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # %%
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    min_score_thresh = 0.5

    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=5,
            min_score_thresh=min_score_thresh)

    boxes = output_dict['detection_boxes']
    max_boxes_to_draw = boxes.shape[0]
    scores = output_dict['detection_scores']

    output_boxes = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):

        if scores is None or scores[i] > min_score_thresh:
            class_name = category_index[output_dict['detection_classes'][i]]['name']
            output_boxes.append(list(boxes[i]))

    os.system('rm ' + './images/' + request.files.getlist("file")[0].filename)
    return render_template("complete.html", value=output_boxes)


