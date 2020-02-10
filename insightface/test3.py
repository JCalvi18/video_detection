from __future__ import division
import cv2
import numpy as np
# from insightface.model_zoo import model_zoo
from test_utils import get_retinaface
from time import time
import argparse


parser = argparse.ArgumentParser('Test 3: Insightface using batches')
parser.add_argument('--in-file', type=str, default='database/variete.mp4')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--batch-size', type=int, default=1)


def video_get(detector, nframes=60):
    cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
    frame_w, frame_h = int(cap.get(3)), int(cap.get(4))
    batch_time = np.array([])
    faces_detected = 0
    for _ in range(nframes//args.batch_size):
        frames = np.zeros((args.batch_size, frame_h, frame_w, 3))
        for b in range(args.batch_size):
            r, f = cap.read()
            if r:
                frames[b] = f


        start = time()
        boxes = detector.detect(frames, threshold=0.6)[0]
        faces_detected += boxes.shape[0]
        batch_time = np.append(batch_time, time() - start)
    cap.release()
    return batch_time.mean(), faces_detected


if __name__ == '__main__':
    args = parser.parse_args()

    ## InsightFace
    in_det = get_retinaface('mnet025_v2')
    in_det.prepare(args.gpu)

    fr_time, nfaces = video_get(in_det)
    print('InsightFace:\nAverage time per batch: %0.3f\nNumber of faces: %i' % (fr_time, nfaces))
    print()
