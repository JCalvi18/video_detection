import mxnet as mx
from mxnet import nd
import cv2
import numpy as np
from numpy.linalg import norm
import os
import argparse
import pickle
# import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
# import face_recognition as fr
from utils import check_large_pose
from insightface.model_zoo import model_zoo
from insightface.utils import face_align


parser = argparse.ArgumentParser('Face recognition and verification using dlib')
parser.add_argument('--faces-dir', type=str, default='database/faces')
parser.add_argument('--in-file', type=str, default='variete.mp4')
parser.add_argument('--out-file', type=str, default='face_variete.mp4')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--threshold-det', type=float, default=0.6, help='Threshold for face detection')
parser.add_argument('--threshold-rec', type=float, default=0.5, help='Threshold for face recognition')
parser.add_argument('--threshold-l2', type=float, default=50.0, help='Threshold for l2 distance')
parser.add_argument('--prepare', action='store_true', help='Create the dataset based on the images of --faces_dir')


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


box_colors = ['a50104', '261c15', 'ff01fb', '2e1e0f', '003051', 'f18f01', '6e2594', 'FFFFFF']
box_colors = [hex2rgb(h) for h in box_colors]


class Person(object):
    def __init__(self, pre_point, box):
        self.pre_point = pre_point  # Center point of the face
        self.box = box  # Bounding box of the face
        self.name = None
        self.known = None  # Is in the dataset?

    def l2_distance(self, point):
        # Compare actual point with previous point
        # If l2 distance > l2 threshold return False else the value
        l2 = (((point[0] - self.pre_point[0]) ** 2) + ((point[1] - self.pre_point[1]) ** 2)) ** 0.5
        if l2 > args.threshold_l2:
            return -1
        else:
            return l2


class VideoDetector(object):
    def __init__(self, mx_context):
        self.ctx = mx_context
        self.dataset = None  # Collection of features of known names
        self.names = {}  # Names of known person
        self.persons = []  # List of person detected
        self.det_model = model_zoo.get_model('retinaface_mnet025_v2')
        self.rec_model = model_zoo.get_model('arcface_r100_v1')
        self.det_model.prepare(args.gpu)
        self.rec_model.prepare(args.gpu)

    def prepare_faces(self, dataset_name='dataset.pkl'):
        image_names = os.listdir(args.faces_dir)
        face_names = set([x.split('_')[0] for x in image_names])

        dataset = {}
        for name in face_names:
            images = [cv2.imread(os.path.join(args.faces_dir, iname)) for iname in image_names if name in iname]
            points = [self.det_model.detect(img, threshold=args.threshold_det)[1] for img in images]
            cr_images = [face_align.norm_crop(img, lndm[0]) for img, lndm in zip(images, points) if lndm.shape[0] == 1]
            embeddings = [self.rec_model.get_embedding(img).flatten() for img in cr_images]
            normed_embedding = [embedding / norm(embedding) for embedding in embeddings]
            dataset[name] = np.stack(normed_embedding)

        # for k, v in dataset.items():
        #     print(k, v.shape)
        dataset_path = os.path.abspath(os.path.join(args.faces_dir, '..'))

        with open(dataset_path + '/' + dataset_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    def detect(self):
        # if self.dataset is None:
        #     self.load_features()
        cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
        frame_w, frame_h = int(cap.get(3)), int(cap.get(4))  # Convert resolutions from float to integer.

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        renders = []
        frame_time = np.array([])
        for _ in tqdm(range(total_frames)):
            start = time()
            ret, frame = cap.read()
            if ret:
                total_boxes, points = self.detect_faces(frame)
                self.identify(frame, total_boxes, points)
                render = self.draw_names(frame)
                renders.append(render)
            frame_time = np.append(frame_time, time() - start)
        cap.release()
        return renders, {'w': frame_w, 'h': frame_h}, {'fr_exec': frame_time.mean()}

    def load_features(self, dataset_name='dataset.pkl'):
        dataset_path = os.path.abspath(os.path.join(args.faces_dir, '..'))
        with open(dataset_path + '/' + dataset_name, 'rb') as f:  # Load Dataset on numpy format
            np_dataset = pickle.load(f)
        # Create dictionary with person names and their corresponding feature index
        i = 0
        for k, dv in np_dataset.items():
            self.names[k] = slice(i, i + dv.shape[0])
            i += dv.shape[0]
        # Transform dataset to mx NDarray format
        # self.dataset = np.array(np.concatenate([val for val in np_dataset.values()]))
        self.dataset = nd.array(np.concatenate([val for val in np_dataset.values()]), ctx=self.ctx)

    def draw_names(self, frame):
        colors = box_colors[:len(self.persons)]
        for person, c in zip(self.persons, colors):
            b = person.box
            if person.name is None:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), colors[-1], 2)
            else:
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
                cv2.putText(frame, person.name, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3,
                            cv2.LINE_AA)
        return frame

    def name_face(self, face_image):
        # Name the face of a person based on the dataset

        # Transform the cropped image into a feature vector
        embedding = self.rec_model.get_embedding(face_image).flatten()
        face_feature = nd.array(embedding / norm(embedding), ctx=self.ctx)  # normed

        # Calculate the similarity between the known features and the current face feature
        sim = nd.dot(self.dataset, face_feature)
        scores = {}
        for known_id, index in self.names.items():
            scores[known_id] = max(sim[index])

        if max(scores.values()) > args.threshold_rec:
            return max(scores, key=scores.get)
        else:
            return None

    def name_person(self, frame, points, box, person=None):
        # This function will verify if a persons face is part or not of the dataset
        # Depending if the pose of the face is correct or if the face is in the dataset a name will be assigned
        if person is None:  # Create a new object
            person = Person(points[2], box)
        else:  # Update variables
            person.pre_point = points[2]
            person.box = box

        if person.known:  # Check if the person has already been checked
            return
        # Obtain face orientation and if looking to the front take a screenshot
        ret, l, r, u, d = check_large_pose(points, box[:-1])
        if ret != 4:
            cr_face = face_align.norm_crop(frame, points)
            name = self.name_face(cr_face)
            if name is not None:  # Face verified on the dataset
                person.name = name
                person.known = True

        if person not in self.persons:
            self.persons.append(person)

    def identify(self, frame, bbox, points):
        # Identify the person based on face verification or coordinate approximation

        if not self.persons and bbox.shape[0]:  # Check if persons list is empty and there are detected faces
            for i in range(bbox.shape[0]):
                box = bbox[i].astype(np.int)
                landmark5 = points[i].astype(np.int)
                self.name_person(frame, landmark5, box)

        elif len(self.persons) == bbox.shape[0]:  # Find corresponding person for each coordinate
            for i in range(bbox.shape[0]):
                box = bbox[i].astype(np.int)
                landmark5 = points[i].astype(np.int)
                distances = np.array([p.l2_distance(landmark5[2]) for p in self.persons])
                person_index = np.where(distances >= 0, distances, np.inf).argmin()
                self.name_person(frame, landmark5, box, person=self.persons[person_index])

        elif len(self.persons) < bbox.shape[0]:  # Identify previous persons and add new ones
            boxes = [box for box in bbox.astype(np.int)]
            landmarks = [point for point in points.astype(np.int)]

            centers = [p[2] for p in landmarks]
            distances = np.array([p.l2_distance(center) for p in self.persons for center in centers]).reshape(
                len(self.persons), -1)
            known_index = np.where(distances >= 0, distances, np.inf).argmin(axis=1)
            unknown_index = [i for i in range(distances.shape[-1]) if np.all(distances[:, i] == -1)]

            # update known persons
            for ki, p in zip(known_index, self.persons):
                self.name_person(frame, landmarks[ki], boxes[ki], person=p)
            # add unknown persons
            for uk in unknown_index:
                self.name_person(frame, landmarks[uk], boxes[uk])

        elif len(self.persons) > bbox.shape[0]:
            if not bbox.shape[0]:  # There is no faces detected
                del self.persons[:]  # Empty the hole list
                return
            # Identify previous persons and remove the ones that disappeared
            boxes = [box for box in bbox.astype(np.int)]
            landmarks = [point for point in points.astype(np.int)]

            centers = [p[2] for p in landmarks]
            distances = np.array([p.l2_distance(center) for p in self.persons for center in centers]).reshape(
                len(self.persons), -1)
            center_index = np.array([i for i in distances if np.any(i >= 0)])
            if bbox.shape[0] > 1:
                center_index = np.where(center_index >= 0, center_index, np.inf).argmin(axis=1)
            else:
                center_index = np.array([0])

            known_person = np.array([i for i in range(distances.shape[0]) if np.any(distances[i] >= 0)])
            d_person = [i for i in range(distances.shape[0]) if np.all(distances[i] == -1)]

            # update known persons
            for ci, ki in zip(center_index, known_person):
                self.name_person(frame, landmarks[ci], boxes[ci], person=self.persons[ki])
            # delete disappeared persons from list
            for d in reversed(d_person):
                del self.persons[d]

    def detect_faces(self, frame):
        if self.dataset is None:
            self.load_features()
        # run detector
        return self.det_model.detect(frame, threshold=args.threshold_det)
        # bbox, points = self.det_model(frame, threshold=args.threshold_det)
        # if bbox.shape[0]:  # is not empty
        #     landmarks = fr.face_landmarks(frame, face_locations=bbox)
        #     points = []
        #     for lm in landmarks:
        #         point = []
        #         for k in ['left_eye', 'right_eye']:
        #             x = int(abs((lm[k][0][0] - lm[k][3][0]) / 2) + lm[k][0][0])
        #             y = int(abs((lm[k][0][1] - lm[k][3][1]) / 2) + lm[k][0][1])
        #             point.append((x, y))
        #         point.append(lm['nose_bridge'][-1])
        #         point.append(lm['top_lip'][0])
        #         point.append(lm['top_lip'][6])
        #         points.append(point)
        #
        #     return np.array(bbox), np.stack(points)
        # else:
        #     return np.array([]), np.array([])


if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu >= 0:
        print('Using gpu:{}'.format(args.gpu))
    else:
        print('Using cpu')

    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu(0)
    vd = VideoDetector(ctx)

    if args.prepare:
        print('Transforming images from: {}'.format(os.path.abspath(args.faces_dir)))
        vd.prepare_faces()
        print('Features saved on:{}'.format(os.path.abspath(args.faces_dir + '../dataset.pkl')))
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
