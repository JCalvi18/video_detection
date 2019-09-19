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
from utils.debugger import Debugger

parser = argparse.ArgumentParser('Put title here')
parser.add_argument('--in-file', type=str, default='../exp/mini/variete.mp4', help='Input File')
parser.add_argument('--out-file', type=str, default='../exp/mini/ctdet_variete.mp4', help='Input File')
parser.add_argument('--total-frames', type=int, default=100, help='Number of frames to record, only used if in-file is a camera')
parser.add_argument('--load-model', type=str, default='../models/ctdet_coco_dla_2x.pth')
parser.add_argument('--gpus', type=int, default=-1, help='-1 CPU')
parser.add_argument('--arch', type=str, default='dla_34', help='Type of architecture')
parser.add_argument('--K', type=int, default=100, help='max number of output objects.')
# parser.add_argument('--bool', action='store_true', help='This is a boolean')


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


video_ext = ['mp4', 'mov', 'avi', 'mkv']
local_args = ['in_file', 'out_file', 'total_frames']
box_colors = ['a50104', '261c15', 'ff01fb', '2e1e0f', '003051', 'f18f01', '6e2594', 'FFFFFF']
box_colors = [hex2rgb(h) for h in box_colors]


def opt_args(arg_dict):
    d = [(k, v) for k, v in arg_dict.items() if k not in local_args]
    for i in np.tile([1, 0], len(d)):
        if i:
            yield '--'+d[-1][0]
        else:
            aux = d[-1][1]
            d.pop()
            yield str(aux)


class CTDET(object):
    def __init__(self, args):
        self.args = args

        # Initialize CenterNet argument parser
        oargs = [a for a in opt_args(vars(args))]
        oargs.insert(0, TASK)
        self.opt = opts().init(oargs)

        self.detector = detector_factory[self.opt.task](self.opt)

        self.frame_w, self.frame_h = None, None
        self.total_frames = args.total_frames

        self.names = ['Person']
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

    def draw(self, i, frame, res, show_txt=True):
        for bbox in res[1]:  # only human
            if bbox[4] > self.opt.vis_thresh:
                bbox = np.array(bbox, dtype=np.int32)
                c = box_colors[1]
                txt = '{}'.format(self.names[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
                cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
                if show_txt:
                    cv2.rectangle(frame,
                                  (bbox[0], bbox[1] - cat_size[1] - 2),
                                  (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
                    cv2.putText(frame, txt, (bbox[0], bbox[1] - 2),
                                font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    def detect(self):
        cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
        total_time = np.array([])
        detection_time = np.array([])
        for frame_count in tqdm(range(self.total_frames)):
            start = time()
            ret, frame = cap.read()
            if ret:
                results = self.detector.run(frame)
                detection = results['results']
                self.draw(frame_count, frame, detection)
                detection_time = np.append(detection_time, results['tot'])
                total_time = np.append(total_time, time() - start)
                yield frame
        cap.release()

        yield {'fr_exec': total_time.mean(), 'det_exec': detection_time.mean()}


if __name__ == '__main__':
    args = parser.parse_args()
    TASK = 'ctdet'  # For this program the task wil always be center detection
    ctdet = CTDET(args)

    # Export rendered frames to a video file
    out = cv2.VideoWriter(args.out_file, cv2.VideoWriter_fourcc(*'mp4v'), ctdet.fps, (ctdet.frame_w, ctdet.frame_h))
    for fr in ctdet.detect():
        if type(fr) == np.ndarray:
            out.write(fr)  # as a render frame
        else:
            out.release()
            measures = fr  # as a result
    print('Video saved on:{}'.format(os.path.abspath(args.out_file)))
    print('Average execution time per frame: %0.3f seg' % measures['fr_exec'])
    print('Average execution time per detection: %0.3f seg' % measures['det_exec'])
