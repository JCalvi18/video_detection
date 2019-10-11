import sys
import os
from typing import Dict, Any, Union

import numpy as np
import time
import cv2
from tqdm import tqdm
from time import time
import argparse
CENTERNET_PATH_LIB = os.path.abspath('.')+'/lib'
if CENTERNET_PATH_LIB not in sys.path: # ADD CenterNet to the python path
    sys.path.append(CENTERNET_PATH_LIB)

from detectors.detector_factory import detector_factory
from opts import opts

parser = argparse.ArgumentParser('Put title here')
parser.add_argument('--in-file', type=str, default='../exp/mini/variete.mp4', help='Input File')
parser.add_argument('--out-file', type=str, default='../exp/mini/ctdet_variete.mp4', help='Input File')
parser.add_argument('--total-frames', type=int, default=100, help='Number of frames to record, only used if in-file is a camera')
parser.add_argument('--load-model', type=str, default='../models/ctdet_coco_dla_2x.pth')
parser.add_argument('--gpus', type=int, default=-1, help='-1 CPU')
parser.add_argument('--arch', type=str, default='dla_34', help='Type of architecture')
parser.add_argument('--K', type=int, default=100, help='max number of output objects.')
parser.add_argument('--vis-thresh', type=float, default=0.7, help='visualization threshold')
parser.add_argument('--l2-thresh', type=float, default=50.0, help='Threshold for l2 distance')
parser.add_argument('--show-points', action='store_true', help='Show center points instead of boxes')


def hex2rgb(h):
    v = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return tuple(reversed(v))


def box_point(b):
    return [int(b[0]+((b[2]-b[0])/2)), int(b[1]+((b[3]-b[1])/2))]


def deb(frame):
    while True:
        cv2.imshow('na', frame)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()

video_ext = ['mp4', 'mov', 'avi', 'mkv']
local_args = ['in_file', 'out_file', 'total_frames', 'l2_thresh', 'show_points']
colors = ['a50104', '327baa', 'ff01fb', '2e1e0f', '003051', 'f18f01', '6e2594', 'FFFFFF']
colors = [hex2rgb(h) for h in colors]


def opt_args(arg_dict):
    d = [(k, v) for k, v in arg_dict.items() if k not in local_args]
    for i in np.tile([1, 0], len(d)):
        if i:
            yield '--'+d[-1][0]
        else:
            aux = d[-1][1]
            d.pop()
            yield str(aux)


class Person(object):
    def __init__(self, box, name, l2_thresh, vis_thresh):
        self.score = box[4]
        box = np.array(box[:-1], dtype=np.int32)
        self.pre_point = box_point(box)  # Center point of the person
        self.box = box  # Bounding box
        self.name = name
        self.l2_thresh = l2_thresh
        self.vis_thresh = vis_thresh

    def update(self, box):
        self.score = box[4]
        box = np.array(box, dtype=np.int32)
        self.pre_point = box_point(box)  # Center point of the person
        self.box = box  # Bounding box

    def l2_distance(self, point):
        # Compare actual point with previous point
        # If l2 distance > l2 threshold return False else the value
        l2 = (((point[0] - self.pre_point[0]) ** 2) + ((point[1] - self.pre_point[1]) ** 2)) ** 0.5
        if l2 > self.l2_thresh:
            return -1
        else:
            return l2

    def draw(self, frame, show_box=True, show_txt=True):
        bbox = self.box
        txt = '{}->{:.2f}'.format(self.name, self.score)
        c = colors[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]

        if show_box:
            cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        else:
            cv2.circle(frame, (self.pre_point[0], bbox[1]), 5, colors[0], -1)

        if show_txt and show_box:
            cv2.rectangle(frame,
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(frame, txt, (bbox[0], bbox[1] - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        elif show_txt:
            cv2.putText(frame, txt, (self.pre_point[0], bbox[1]-5),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


class CTDET(object):
    def __init__(self, args):
        self.args = args
        # Initialize CenterNet argument parser
        oargs = [a for a in opt_args(vars(args))]
        oargs.insert(0, TASK)
        self.opt = opts().init(oargs)
        # NN
        self.detector = detector_factory[self.opt.task](self.opt)
        # Video vars
        self.frame_w, self.frame_h = None, None
        self.total_frames = args.total_frames
        # Identification vars
        self.persons = []
        self.names = ['Person_' + str(i) for i in reversed(range(10))]

        # prepare Input data
        ext = args.in_file.split('.')[-1].lower()

        if ext in video_ext:
            cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
            if not cap.isOpened():  # Check if camera opened successfully
                print("Unable to open video capture feed")
                self.total_frames = -1
                return

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.frame_w, self.frame_h = int(cap.get(3)), int(cap.get(4))  # Convert resolutions from float to integer.
            cap.release()
        else:
            print('Video format not supported')

    def draw(self, frame, show_box=True, show_txt=True):
        for person in self.persons:
            person.draw(frame, show_box=show_box, show_txt=show_txt)

    def update_person(self, box, person=None):
        if person is None:
            person = Person(box, self.names[-1], args.l2_thresh, args.vis_thresh)
            self.names.pop()
        else:
            person.update(box)

        if person not in self.persons and person is not None:
            self.persons.append(person)

    def identify(self, det):
        bbox = np.array([b for b in det[1] if b[4] > args.vis_thresh])  # Only persons with high socores
        if not self.persons and bbox.shape[0]:  # Check if persons list is empty and there are detected persons
            for b in bbox:
                self.update_person(b)

        elif len(self.persons) == bbox.shape[0]:  # Find corresponding person for each coordinate
            for b in bbox:
                center_point = box_point(b)
                distances = np.array([p.l2_distance(center_point) for p in self.persons])
                person_index = np.where(distances >= 0, distances, np.inf).argmin()
                self.update_person(b, person=self.persons[person_index])

        elif len(self.persons) < bbox.shape[0]:  # Identify previous persons and add new ones
            boxes = [box for box in bbox]
            centers = [box_point(box) for box in bbox]
            distances = np.array([p.l2_distance(center) for p in self.persons for center in centers]).reshape(
                len(self.persons), -1)
            known_index = np.where(distances >= 0, distances, np.inf).argmin(axis=1)
            unknown_index = [i for i in range(distances.shape[-1]) if np.all(distances[:, i] == -1)]
            # update known persons
            for ki, p in zip(known_index, self.persons):
                self.update_person(boxes[ki], person=p)
            # add unknown persons
            for uk in unknown_index:
                self.update_person(boxes[uk])

        elif len(self.persons) > bbox.shape[0]:
            if not bbox.shape[0]:  # There is no faces detected
                del self.persons[:]  # Empty the hole list
                return
            # Identify previous persons and remove the ones that disappeared
            boxes = [box for box in bbox]
            centers = [box_point(box) for box in bbox]

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
                self.update_person(boxes[ci], person=self.persons[ki])
            # delete disappeared persons from list
            self.persons = [self.persons[d] for d in range(len(self.persons)) if d not in d_person]

    def detect(self):
        cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
        total_time = np.array([])
        detection_time = np.array([])
        renders = []
        for _ in tqdm(range(self.total_frames)):
            start = time()
            ret, frame = cap.read()
            if ret:
                results = self.detector.run(frame)
                detection = results['results']
                self.identify(detection)
                self.draw(frame, show_box=True)
                detection_time = np.append(detection_time, results['tot'])
                total_time = np.append(total_time, time() - start)
                renders.append(frame)
        cap.release()

        return renders, {'fr_exec': total_time.mean(), 'det_exec': detection_time.mean()}


if __name__ == '__main__':
    args = parser.parse_args()
    TASK = 'ctdet'  # For this program the task wil always be center detection
    ctdet = CTDET(args)

    renders, measures = ctdet.detect()
    # Export rendered frames to a video file
    out = cv2.VideoWriter(args.out_file, cv2.VideoWriter_fourcc(*'mp4v'), ctdet.fps, (ctdet.frame_w, ctdet.frame_h))
    for fr in renders:
        out.write(fr)  # as a render frame
    out.release()

    print('Video saved on:{}'.format(os.path.abspath(args.out_file)))
    print('Average execution time per frame: %0.3f seg' % measures['fr_exec'])
    print('Average execution time per detection: %0.3f seg' % measures['det_exec'])
