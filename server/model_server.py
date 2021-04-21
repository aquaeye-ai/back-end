"""
Script that runs server that serves requests for inference_livestream_keras.py
Should be ran within same directory as script with following command from a terminal: gunicorn model_server:app
"""

import sys
sys.path.append("/share/tensorflow/models/research")

import os
import cv2
import yaml
import time
import json
import falcon
import base64
import logging
import pprint

import numpy as np
import tensorflow as tf

from falcon_cors import CORS
from tensorflow import keras
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class RequireJSON(object):

        def process_request(self, req, resp):
                if not req.client_accepts_json:
                        raise falcon.HTTPNotAcceptable(
                                'This API only supports responses encoded as JSON.',
                                href='http://docs.examples.com/api/json')

                if req.method in ('POST', 'PUT'):
                        if 'application/json' not in req.content_type:
                                raise falcon.HTTPUnsupportedMediaType(
                                        'This API only supports requests encoded as JSON.',
                                        href='http://docs.examples.com/api/json')


class JSONTranslator(object):

        def process_request(self, req, resp):
                # req.stream corresponds to the WSGI wsgi.input environ variable,
                # and allows you to read bytes from the request body.
                #
                # See also: PEP 3333
                if req.content_length in (None, 0):
                        # Nothing to do
                        return

                body = req.stream.read()
                if not body:
                        raise falcon.HTTPBadRequest('Empty request body',
                                                                                'A valid JSON document is required.')

                try:
                        req.context['doc'] = json.loads(body.decode('utf-8'))

                except (ValueError, UnicodeDecodeError):
                        raise falcon.HTTPError(falcon.HTTP_753,
                                                                   'Malformed JSON',
                                                                   'Could not decode the request body. The '
                                                                   'JSON was incorrect or not encoded as '
                                                                   'UTF-8.')

        def process_response(self, req, resp, resource, req_succeeded):
                if not hasattr(resp.context, 'result'):
                        return

                resp.body = json.dumps(resp.context.result)


# Sample resource from the docs to test the api is running correctly.
class QuoteResource:
        def on_get(self, req, resp):
                """Handles GET requests"""
                quote = {
                        'quote': 'I\'ve always been more interested in the future than in the past.',
                        'author': 'Grace Hopper'
                }

                resp.body = json.dumps(quote)


# Resource to provide the number of classes for input scales/sliders on FE gui.
class NumClassesResource:
        def init(self, num_classes=None):
                self.num_classes = num_classes

        def on_get(self, req, resp):
                """Handles GET requests"""
                response = {}
                response['num_classes'] = self.num_classes

                resp.body = json.dumps(response)


# Manage predict requests.
class PredictResource(object):

        def __init__(self):
                self.logger = logging.getLogger('modelserver.' +  __name__)

        def init(self, models=None):
                self.models = models

        def on_post(self, req, resp):

                try:
                        # print(req)
                        doc = req.context.doc
                # print(doc)
                except AttributeError:
                        raise falcon.HTTPBadRequest(
                                'Missing thing',
                                'A thing must be submitted in the request body.')

                # Get the image id, height, width and data from the JSON msg.
                image = base64.b64decode(doc['image'])
                image = np.frombuffer(image, dtype=np.uint8).copy()
                image = cv2.imdecode(image, flags=1)
                id = doc['id']
                height = doc['height']
                width = doc['width']
                depth = doc['depth']
                K = doc['K']
                m_name = doc['model']
                model = self.models[m_name]

                print("id: {}".format(id))
                print("source shape: {}x{}".format(height, width))
                print("target shape: {}x{}".format(model.model_input_height, model.model_input_width))
                print("model: {}".format(m_name))

                # scale image since we trained on scaled images
                image = image * (1. / model.scale)

                # reshape into original dimensions
                image = np.reshape(image, (height, width, depth))

                # resize image to fit model input dimensions
                image = cv2.resize(image, dsize=(model.model_input_height, model.model_input_width), interpolation=cv2.INTER_LINEAR)
                image = np.expand_dims(image, axis=0)

                # Perform evaluation of the image
                output_dict = model.predict(image)

                class_scores = output_dict[0]

                # sort the class_scores
                top_k_scores_idx = np.argsort(class_scores)[-K:]
                top_k_scores_idx = list(reversed(top_k_scores_idx))
                top_k_scores = class_scores[top_k_scores_idx]

                ## return the top-k classes and scores to text file
                class_labels = [model.category_index[i] for i in top_k_scores_idx]

                # Create the response message
                response = {}
                response['id'] = id
                response['top_k_classes'] = class_labels
                response['top_k_scores'] = [str(i) for i in top_k_scores]

                # Return the response message
                resp.body = json.dumps(response)
                resp.status = falcon.HTTP_201

# Manage find requests.
class FindResource(object):

        def __init__(self):
                self.logger = logging.getLogger('modelserver.' +  __name__)

        def init(self, models=None):
                self.models = models

        def inference_images(self, images=None, sess=None, model=None):
                with tf.Graph().as_default():
                        session = tf.compat.v1.Session()
                        with session as sess:
                                 od_graph_def = tf.compat.v1.GraphDef()         
                                 with tf.compat.v2.io.gfile.GFile(model.config['model'], "rb") as fid:
                                         serialized_graph = fid.read()
                                         od_graph_def.ParseFromString(serialized_graph)
                                         tf.import_graph_def(od_graph_def, name="")

                                         # Get handles to input and output tensors
                                         ops = tf.compat.v1.get_default_graph().get_operations()
                                         all_tensor_names = {output.name for op in ops for output in op.outputs}
                                         tensor_dict = {}
                                         for key in [
                                                 'num_detections', 'detection_boxes', 'detection_scores',
                                                 'detection_classes', 'detection_masks'
                                         ]:
                                                 tensor_name = key + ':0'
                                                 if tensor_name in all_tensor_names:
                                                         tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

                                         # we need to expand the first dimension if we only inference on one image, since the model inference expects a batch
                                         if len(images.shape) == 3:
                                                 images = np.expand_dims(images, 0)
                                         image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

                                         # Run inference
                                         t1 = time.time()
                                         output_dict = sess.run(tensor_dict, feed_dict={image_tensor: images})
                                         t2 = time.time()
                                         self.logger.info("inference time: {}".format(t2-t1))

                                         # all outputs are float32 numpy arrays, so convert types as appropriate
                                         output_dict['num_detections'][:] = np.array(list(map(int, output_dict['num_detections'][:]))).astype(np.float32)
                                         output_dict['detection_classes'][:] = output_dict['detection_classes'][:].astype(np.int64)
                                         output_dict['detection_boxes'][:] = output_dict['detection_boxes'][:]
                                         output_dict['detection_scores'][:] = output_dict['detection_scores'][:]

                                         return output_dict

        def on_post(self, req, resp):
                try:
                        # print(req)
                        doc = req.context.doc
                # print(doc)
                except AttributeError:
                        raise falcon.HTTPBadRequest(
                                'Missing thing',
                                'A thing must be submitted in the request body.')

                pp = pprint.PrettyPrinter(indent=4)
                # Get the image id, height, width and data from the JSON msg.
                image = base64.b64decode(doc['image'])
                image = np.frombuffer(image, dtype=np.uint8).copy()
                image = cv2.imdecode(image, flags=1)
                height, width, depth = image.shape
                id = doc['id']
                m_name = doc['model']
                model = self.models[m_name]
                threshold = float(doc['threshold'])

                print("id: {}".format(id))
                print("source shape: {}x{}".format(height, width))
                print("model: {}".format(m_name))
                print("threshold: {}".format(threshold))

                # reshape into original dimensions
                image = np.reshape(image, (height, width, depth))

                # Perform evaluation of the image
                output_dict = self.inference_images(image, model=model)

                detection_scores = output_dict['detection_scores'][0]
                detection_classes = output_dict['detection_classes'][0]
                detection_boxes = output_dict['detection_boxes'][0]

                response = {}
                response['id'] = id
                response['detection_scores'] = detection_scores.tolist()
                response['detection_classes'] = detection_classes.tolist()
                response['detection_boxes'] = detection_boxes.tolist()
                response['category_index'] = model.config['category_index']

                # Return the response message
                resp.body = json.dumps(response)
                resp.status = falcon.HTTP_201

# class to abstract the infrastructure around loading/predicting a model saved to a .h5 file
class Model_H5:
        def __init__(self, config):
           self.model_input_height = config["model_input_height"]
           self.model_input_width = config["model_input_width"]
           self.scale = config["scale"]
           self.category_index = config["category_index"]
           self.graph = tf.Graph()
           with self.graph.as_default():
                   self.session = tf.compat.v1.Session()
                   with self.session.as_default():
                           self.model = keras.models.load_model(path_to_model_h5)

        def predict(self, X):
                with self.graph.as_default():
                        with self.session.as_default():
                                return self.model.predict(X)

# class to abstract the infrastructure around loading/predicting a model saved to a .pb file
class Model_PB:
        def __init__(self, config):
                self.config = config

cors = CORS(
        allow_all_origins=True,
        allow_all_headers=True,
        allow_all_methods=True
)

# this can't be in __main__ for some reason...TODO find out why
app = falcon.API(middleware=[
                RequireJSON(),
                JSONTranslator(),
                cors.middleware
        ])
predict_resource = PredictResource()
find_resource = FindResource()
num_classes_resource = NumClassesResource()
app.add_route('/predict', predict_resource)
app.add_route('/find', find_resource)
app.add_route('/num-classes', num_classes_resource)
app.add_route('/quote', QuoteResource())


if __name__ == 'model_server':  
        # we expect, as a hand-shake agreement, that there is a .yml config file in top level of lib/configs directory
        config_dir = os.path.join('.')
        yaml_path = os.path.join(config_dir, 'model_server.yml')
        with open(yaml_path, "r") as stream:
                config = yaml.load(stream)
        
        models = {}

        for m_name in config["models"]:
                ## collect hyper parameters/args from config
                # NOTE: float() is required to parse any exponentials since YAML sends exponentials as strings
                m_config = config["models"][m_name]
                path_to_model_h5 = m_config["model"]
                path_to_labels = m_config["labels"]
                m_type = m_config["type"]

                # create category_index
                # The structure the category index differs between image classification models and object detection models
                category_index = {}
                if m_type == "image_classification":
                        # Dictionary of the strings that is used to add correct label for each class index in the model's output.
                        # key: index in output
                        # value: string name of class
                        with open(path_to_labels) as labels_f:
                                # we need to sort the labels since this is how keras reads the labels in during training: https://stackoverflow.com/questions/38971293/get-class-labels-from-keras-functional-model
                                idx = 0
                                for line in sorted(labels_f):
                                        category_index[idx] = line.strip()
                                        idx += 1
                elif m_type == "object_detection":
                        category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=False)

                num_classes_resource.init(num_classes=len(category_index.keys()))

                print("Loading model server: {}.".format(m_name))
                m_config["category_index"] = category_index

                model = None
                if m_type == "image_classification":
                        model = Model_H5(m_config)
                elif m_type == "object_detection":
                        model = Model_PB(m_config)
                models[m_name] = model
        predict_resource.init(models=models)
        find_resource.init(models=models)

        # # Expose port, transition to gunicorn or similar if needed.
        # print("Making simple server.")
        # httpd = simple_server.make_server('127.0.0.1', 8000, app)
        # print("Simple server made, starting server.")
        # httpd.serve_forever()
        # print("Server closed.")
