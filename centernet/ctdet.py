import sys
import os
import numpy as np
import time
import cv2
from time import time
import argparse
CENTERNET_PATH_LIB = os.path.abspath('.')+'/lib'
if CENTERNET_PATH_LIB not in sys.path: # ADD CenterNet to the python path
    sys.path.append(CENTERNET_PATH_LIB)

from detectors.detector_factory import detector_factory
from opts import opts
from utils.debugger import Debugger

parser = argparse.ArgumentParser('Put title here')
parser.add_argument('--model-path', type=str, default='../models/ctdet_coco_dla_2x.pth')
parser.add_argument('--in-file', type=str, default='../exp/mini/variete.mp4', help='Input File')
parser.add_argument('--out-file', type=str, default='../exp/mini/ctdet_variete.mp4', help='Input File')
parser.add_argument('--total-frames', type=int, default=100, help='Number of frames to record, only used if in-file is a camera')
parser.add_argument('--gpu', type=int, default=-1, help='-1 CPU')
parser.add_argument('--arch', type=str, default='dla_34', help='Type of architecture')
# parser.add_argument('--bool', action='store_true', help='This is a boolean')

video_ext = ['mp4', 'mov', 'avi', 'mkv']


class CTDET(object):
    def __init__(self, args):
        self.args = args

        # Initialize CenterNet argument parser
        self.opt = opts().init('{} --load_model {} --gpus {} --arch {}'.format(TASK, args.model_path, args.gpu,
                                                                               args.arch).split(' '))

        self.detector = detector_factory[self.opt.task](self.opt)
        self.debugger = Debugger(dataset=self.opt.dataset, ipynb=True,theme=self.opt.debugger_theme)

        self.frame_w, self.frame_h = None, None
        self.total_frames = args.total_frames

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

    def draw(self, i, frame, res):
        frame_id = TASK + '_frame' + str(i)
        self.debugger.add_img(frame, img_id=frame_id)
        for j in range(1, self.detector.num_classes + 1):
            for bbox in res[j]:
                if bbox[4] > self.opt.vis_thresh:
                    self.debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id=frame_id)

    def detect(self):
        cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
        total_time = np.array([])
        detection_time = np.array([])
        for frame_count in range(self.total_frames):
            start = time()
            ret, frame = cap.read()
            if ret:
                results = self.detector.run(frame)
                detection = results['results']
                self.draw(frame_count, frame, detection)
                detection_time = np.append(detection_time, results['tot'])
                total_time = np.append(total_time, time() - start)
        cap.release()
        renders = [r for r in self.debugger.imgs.values()]
        return renders, {'w': self.frame_w, 'h': self.frame_h, 'fps': self.fps}, {'fr_exec': total_time.mean(),
                                                                                  'det_exec': detection_time.mean()}


if __name__ == '__main__':
    args = parser.parse_args()
    TASK = 'ctdet'  # For this program the task wil always be center detection
    ctdet = CTDET(args)
    ctdet.detect()

    rendered_frames, frame_spec, measures = ctdet.detect()

    # Export rendered frames to a video file
    out = cv2.VideoWriter(args.out_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_spec['fps'], (frame_spec['w'],
                                                                                              frame_spec['h']))
    for v in rendered_frames:
        out.write(v)
    out.release()
    print('Video saved on:{}'.format(os.path.abspath(args.out_file)))
    print('Average execution time per frame: %0.3f seg' % measures['fr_exec'])
    print('Average execution time per detection: %0.3f seg' % measures['det_exec'])
