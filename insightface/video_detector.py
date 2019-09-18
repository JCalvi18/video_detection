import mxnet as mx
from mxnet import nd
from mtcnn_detector import MtcnnDetector
import face_model
import cv2
import numpy as np
import os
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

parser = argparse.ArgumentParser('Face recognition and verification using Insightface')
parser.add_argument('--image-size', type=str, default='112,112')
parser.add_argument('--faces-dir', type=str, default='../resources/faces')
parser.add_argument('--model', type=str, default='../models/model-r100-ii/model,0')
parser.add_argument('--in-file', type=str, default='../resources/variete.mp4')
parser.add_argument('--out-file', type=str, default='../resources/face_variete.mp4')
parser.add_argument('--ga-model', type=str, default='')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--det', type=int, default=0)
parser.add_argument('--flip', type=int, default=0)
parser.add_argument('--threshold', type=float, default=1.24)
parser.add_argument('--threshold-face', type=float, default=0.4)
parser.add_argument('--prepare', action='store_true', help='This is a boolean')
parser.add_argument('--recognize', action='store_true', help='Temporary flag to test only face identification')


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


box_colors = ['a50104', '261c15', 'ff01fb', '2e1e0f', '003051', 'f18f01', '6e2594']
box_colors = [hex2rgb(h) for h in box_colors]


class VideoDetector(object):
    def __init__(self, arguments, mx_context):
        self.args = arguments
        self.ctx = mx_context
        self.model = face_model.FaceModel(args)
        self.detector = MtcnnDetector(model_folder='mtcnn-model/', ctx=self.ctx, num_worker=4, accurate_landmark=False)
        self.names = None       # Names of the persons in the dataset
        self.dataset = None     # Collection of features of known names

    def prepare_faces(self, dataset_name='dataset.pkl'):
        image_names = os.listdir(self.args.faces_dir)
        face_names = set([x.split('_')[0] for x in image_names])

        dataset = {}
        for name in face_names:
            images = [cv2.imread(os.path.join(self.args.faces_dir, iname)) for iname in image_names if name in iname]
            features = [self.model.get_feature(self.model.get_input(img)) for img in images]
            features = np.stack(features)
            dataset[name] = features

        dataset_path = os.path.abspath(os.path.join(self.args.faces_dir, '..'))

        with open(dataset_path + '/'+dataset_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    def detect(self):
        if self.dataset is None:
            self.load_features()
        cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
        frame_w, frame_h = int(cap.get(3)), int(cap.get(4))  # Convert resolutions from float to integer.

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        renders = []

        frame_time = np.array([])
        for _ in tqdm(range(total_frames)):
            start = time()
            ret, frame = cap.read()
            if ret:
                render = self.detect_faces(frame)
                renders.append(render)
            frame_time = np.append(frame_time, time() - start)
        cap.release()
        return renders, {'w': frame_w, 'h': frame_h}, {'fr_exec': frame_time.mean()}

    def load_features(self, dataset_name='dataset.pkl'):
        dataset_path = os.path.abspath(os.path.join(self.args.faces_dir, '..'))
        with open(dataset_path + '/' + dataset_name, 'rb') as f: # Load Dataset on numpy format
            np_dataset = pickle.load(f)
        # Create dictionary with person names and their corresponding feature index
        self.names = {}
        i = 0
        for k, v in np_dataset.items():
            self.names[k] = slice(i, i + v.shape[0])
            i += v.shape[0]
        # Transform dataset to mx NDarray format
        self.dataset = nd.array(np.concatenate([v for v in np_dataset.values()]), ctx=self.ctx)

    def draw_names(self, frame, names):
        # names: dict{'name' : bounding_box}
        colors = box_colors[:len(names)]
        for name, b, c in zip(names.keys(), names.values(), colors):
            if name == 'unknown':
                for x in b:
                    cv2.rectangle(frame, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), colors[-1], 2)
                    # cv2.putText(frame, 'unknown', (int(b[0]),int(b[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
                cv2.putText(frame, name, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        return frame

    def name_faces(self, persons, total_boxes):
        faces_names = {}
        unknown_faces = []
        for person, box in zip(persons, total_boxes):
            face = self.model.get_input(person)
            if face is None:
                continue
            face = nd.array(self.model.get_feature(face), ctx=self.ctx)

            # Calculate the similarity between the known features and the current face feature
            sim = nd.dot(self.dataset, face)
            scores = {}
            for known_id, index in self.names.items():
                scores[known_id] = max(sim[index]).asnumpy()

            if max(scores.values()) > self.args.threshold_face:
                faces_names[max(scores, key=scores.get)] = box
            else:
                unknown_faces.append(box)

        if len(unknown_faces):
            faces_names['unknown'] = unknown_faces

        return faces_names

    def detect_faces(self, frame):
        resolution = int(self.args.image_size.split(',')[0])
        # run detector
        results = self.detector.detect_face(frame)
        if results is not None:
            total_boxes = results[0]
            points = results[1]
            # extract aligned face chips
            persons = self.detector.extract_image_chips(frame, points, resolution, 0.37)
            if self.args.recognize:
                faces_names = self.name_faces(persons, total_boxes)
            else:
                faces_names = {'unknown': [box for box in total_boxes]}
            return self.draw_names(frame, faces_names)

        else:
            return frame


if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu >= 0:
        print('Using gpu:{}'.format(args.gpu))
    else:
        print('Using cpu')

    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu(0)
    vd = VideoDetector(args, ctx)

    if args.prepare:
        print('Transforming images from: {}'.format(os.path.abspath(args.faces_dir)))
        vd.prepare_faces()
        print('Features saved on:{}'.format(os.path.abspath(args.faces_dir+'../dataset.pkl')))
    else:
        # Draw square on detected faces, and verify each (optional)
        print('Detecting Faces:')
        rendered_frames, frame_spec, measures = vd.detect()

        # Export rendered frames to a video file
        out = cv2.VideoWriter(args.out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_spec['w'], frame_spec['h']))
        for v in rendered_frames:
            out.write(v)
        out.release()
        print('Video saved on:{}'.format(os.path.abspath(args.out_file)))
        print('Average execution time per frame: %0.3f seg' % measures['fr_exec'])



