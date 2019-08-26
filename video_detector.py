import mxnet as mx
from mtcnn_detector import MtcnnDetector
import face_model
import cv2
import numpy as np
import os
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser('Prepare faces dataset')
parser.add_argument('--image-size', type=str, default='112,112')

#parser.add_argument('--faces-dir', type=str, default='../resources/seinfeld/imgs')
parser.add_argument('--faces-dir', type=str, default='../resources/faces')

parser.add_argument('--model', type=str, default='../models/model-r100-ii/model,0')

#parser.add_argument('--in-file', type=str, default='../resources/seinfeld/seinfeld.mp4')
parser.add_argument('--in-file', type=str, default='../resources/variete.mp4')

#parser.add_argument('--out-file', type=str, default='../resources/seinfeld/face_seinfeld.mp4')
parser.add_argument('--out-file', type=str, default='../resources/face_variete.mp4')

parser.add_argument('--ga-model', type=str, default='')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--det', type=int, default=0)
parser.add_argument('--flip', type=int, default=0)
parser.add_argument('--threshold', type=float, default=1.24)
parser.add_argument('--threshold-face', type=float, default=0.4)
parser.add_argument('--prepare', action='store_true', help='This is a boolean')


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


box_colors = ['a50104', '261c15', 'ff01fb', '2e1e0f', '003051', 'f18f01', '6e2594']
box_colors = [hex2rgb(h) for h in box_colors]


def prepare_faces(args, model, dataset_name='dataset.pkl'):
    image_names = os.listdir(args.faces_dir)
    face_names = set([x.split('_')[0] for x in image_names])

    dataset = {}
    for name in face_names:
        images = [cv2.imread(os.path.join(args.faces_dir, iname)) for iname in image_names if name in iname]
        features = [model.get_feature(model.get_input(img)) for img in images]
        features = np.stack(features)
        dataset[name] = features

    dataset_path = os.path.abspath(os.path.join(args.faces_dir, '..'))

    with open(dataset_path + '/'+dataset_name, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_features(args, dataset_name='dataset.pkl'):
    dataset_path = os.path.abspath(os.path.join(args.faces_dir, '..'))
    with open(dataset_path + '/' + dataset_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def draw_names(frame, names):
    # names: dict{'name' : bounding_box}
    colors = box_colors[:len(names)]

    for name, b, c in zip(names.keys(),names.values(),colors):
        if name == 'unknown':
            for x in b:
                cv2.rectangle(frame, (int(x[0]), int(x[1])), (int(x[2]), int(x[3])), colors[-1], 2)
                # cv2.putText(frame, 'unknown', (int(b[0]),int(b[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
            cv2.putText(frame, name, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    return frame


def name_faces(args, frame, model, detector, dataset_features):
    resolution = int(args.image_size.split(',')[0])

    # run detector
    results = detector.detect_face(frame)
    if results is not None:
        total_boxes = results[0]
        points = results[1]

        # extract aligned face chips
        persons = detector.extract_image_chips(frame, points, resolution, 0.37)
        faces_names = {}
        unknown_faces = []
        for person, box in zip(persons, total_boxes):
            face = model.get_input(person)
            if face is None:
                continue
            face = model.get_feature(face)
            scores = {}
            for known_id, known_features in dataset_features.items():
                # minimum distance of all the features of a particular id
                # dist = min([np.sum(np.square(feature - face)) for feature in known_features])
                # maximum similarity
                sim = max([np.dot(feature, face.T) for feature in known_features])
                scores[known_id] = sim

            if max(scores.values()) > args.threshold_face:
                faces_names[max(scores, key=scores.get)] = box
            else:
                unknown_faces.append(box)

        if len(unknown_faces):
            faces_names['unknown'] = unknown_faces

        return draw_names(frame, faces_names)

    else:
        return frame


if __name__ == '__main__':
    args = parser.parse_args()
    model = face_model.FaceModel(args)
    if args.prepare:
        print('Transforming images from: {}'.format(os.path.abspath(args.faces_dir)))
        prepare_faces(args, model)
        print('Features saved on:{}'.format(os.path.abspath(args.faces_dir+'../dataset.pkl')))
    else:
        detector_path = 'mtcnn-model/'
        detector = MtcnnDetector(model_folder=detector_path, ctx=mx.cpu(0), num_worker=4, accurate_landmark=False)
        dataset_features = load_features(args)

        cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w, frame_h = int(cap.get(3)), int(cap.get(4))  # Convert resolutions from float to integer.
        render = []

        print('Detecting Faces:')
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if ret:
                r = name_faces(args, frame, model, detector, dataset_features)
                render.append(r)
        cap.release()


        out = cv2.VideoWriter(args.out_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (frame_w, frame_h))
        for v in render:
            out.write(v)
        out.release()
        print('Video saved on:{}'.format(os.path.abspath(args.out_file)))
