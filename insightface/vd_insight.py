import mxnet as mx
import cv2
import os
import argparse
from video_detector import VideoDetector

parser = argparse.ArgumentParser('Face recognition and verification using dlib')
parser.add_argument('--faces-dir', type=str, default='database/faces')
parser.add_argument('--in-file', type=str, default='variete.mp4')
parser.add_argument('--out-file', type=str, default='face_variete.mp4')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--threshold-det', type=float, default=0.6, help='Threshold for face detection')
parser.add_argument('--threshold-rec', type=float, default=0.5, help='Threshold for face recognition')
parser.add_argument('--threshold-l2', type=float, default=50.0, help='Threshold for l2 distance')
parser.add_argument('--prepare', action='store_true', help='Create the dataset based on the images of --faces_dir')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu >= 0:
        print('Using gpu:{}'.format(args.gpu))
    else:
        print('Using cpu')

    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu(0)
    vd = VideoDetector(ctx, args)

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
