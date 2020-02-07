import cv2
import numpy as np
from insightface.model_zoo import model_zoo
from time import time
import argparse

parser = argparse.ArgumentParser('Test 2: Comparison between Insightface models')
parser.add_argument('--in-file', type=str, default='database/variete.mp4')
parser.add_argument('--gpu', type=int, default=-1)


def video_get(detector, nframes=60):
    cap = cv2.VideoCapture(args.in_file)  # Create a VideoCapture object
    total_frames = nframes
    frame_time = np.array([])
    faces_detected = 0
    for _ in range(total_frames):
        start = time()
        ret, frame = cap.read()
        if ret:
            boxes = detector.detect(frame, threshold=0.6)[0]
            faces_detected += boxes.shape[0]
        frame_time = np.append(frame_time, time() - start)
    cap.release()
    return frame_time.mean(), faces_detected


if __name__ == '__main__':
    args = parser.parse_args()
    models = ['retinaface_r50_v1', 'retinaface_mnet025_v1', 'retinaface_mnet025_v2']
    for m in models:
        in_det = model_zoo.get_model('retinaface_r50_v1')
        in_det.prepare(args.gpu)

        fr_time, nfaces = video_get(in_det)
        del in_det
        print('%s:\nAverage time per frame: %0.3f\nNumber of faces: %i' % (m, fr_time, nfaces))

