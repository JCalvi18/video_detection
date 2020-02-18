from __future__ import division
import os
import numpy as np
from numpy.linalg import norm
import mxnet as mx
import mxnet.ndarray as nd
from insightface.model_zoo.face_detection import FaceDetector, anchors_plane, bbox_pred, landmark_pred
from concurrent.futures import Future
from threading import Thread
from video_detector import VideoDetector
from insightface.model_zoo import model_zoo
from insightface.utils import face_align
from tqdm import tqdm
from time import time
import pickle
import cv2
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


def call_with_future(fn, future, arg, kwargs):
    try:
        result = fn(*arg, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)


def threaded(fn):
    def wrapper(*arg, **kwargs):
        future = Future()
        Thread(target=call_with_future, args=(fn, future, arg, kwargs)).start()
        return future

    return wrapper


def no_return_thread(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


def clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    H, W = tensor.shape[1:]
    h, w = pad_shape
    if h < H or w < W:
        tensor = tensor[:, :h, :w].copy()
    return tensor


class Detector(FaceDetector):
    def __init__(self, _file, rac):
        super().__init__(_file, rac)

    def detect(self, images, threshold=0.5, scale=1.0):
        im_tensor = np.zeros((images.shape[0], 3, images.shape[1], images.shape[2]))
        for b_idx, img in enumerate(images):
            if scale == 1.0:
                im = img
            else:
                im = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            for i in range(3):
                im_tensor[b_idx, i, :, :] = im[:, :, 2 - i]
        data = nd.array(im_tensor)
        db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
        self.model.forward(db, is_train=False)
        net_out = self.model.get_outputs()
        det_list = []
        landmark_list = []
        for batch in range(images.shape[0]):
            proposals_list = []
            scores_list = []
            landmarks_list = []
            for _idx, s in enumerate(self._feat_stride_fpn):
                _key = 'stride%s' % s
                stride = int(s)
                if self.use_landmarks:
                    idx = _idx * 3
                else:
                    idx = _idx * 2
                scores = net_out[idx].asnumpy()
                scores = scores[batch, self._num_anchors['stride%s' % s]:, :, :]
                idx += 1
                bbox_deltas = net_out[idx].asnumpy()[batch]

                height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
                A = self._num_anchors['stride%s' % s]
                K = height * width
                key = (height, width, stride)
                if key in self.anchor_plane_cache:
                    anchors = self.anchor_plane_cache[key]
                else:
                    anchors_fpn = self._anchors_fpn['stride%s' % s]
                    anchors = anchors_plane(height, width, stride, anchors_fpn)
                    anchors = anchors.reshape((K * A, 4))
                    if len(self.anchor_plane_cache) < 100:
                        self.anchor_plane_cache[key] = anchors

                scores = clip_pad(scores, (height, width))
                scores = scores.transpose((1, 2, 0)).reshape((-1, 1))

                bbox_deltas = clip_pad(bbox_deltas, (height, width))
                bbox_deltas = bbox_deltas.transpose((1, 2, 0))
                bbox_pred_len = bbox_deltas.shape[2] // A
                bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

                proposals = bbox_pred(anchors, bbox_deltas)
                # proposals = clip_boxes(proposals, im_info[:2])

                scores_ravel = scores.ravel()
                order = np.where(scores_ravel >= threshold)[0]
                proposals = proposals[order, :]
                scores = scores[order]

                proposals[:, 0:4] /= scale

                proposals_list.append(proposals)
                scores_list.append(scores)

                if self.use_landmarks:
                    idx += 1
                    landmark_deltas = net_out[idx].asnumpy()[batch]
                    landmark_deltas = clip_pad(landmark_deltas, (height, width))
                    landmark_pred_len = landmark_deltas.shape[0] // A
                    landmark_deltas = landmark_deltas.transpose((1, 2, 0)).reshape((-1, 5, landmark_pred_len // 5))
                    landmark_deltas *= self.landmark_std
                    # print(landmark_deltas.shape, landmark_deltas)
                    landmarks = landmark_pred(anchors, landmark_deltas)
                    landmarks = landmarks[order, :]

                    landmarks[:, :, 0:2] /= scale
                    landmarks_list.append(landmarks)

            proposals = np.vstack(proposals_list)
            landmarks = None
            if proposals.shape[0] == 0:
                if self.use_landmarks:
                    landmarks = np.zeros((0, 5, 2))
                return np.zeros((0, 5)), landmarks
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]
            if self.use_landmarks:
                landmarks = np.vstack(landmarks_list)
                landmarks = landmarks[order].astype(np.float32, copy=False)

            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = self.nms(pre_det)
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
            det_list.append(det)
            landmark_list.append(landmarks)
        return det_list, landmark_list


def get_retinaface(name, rac='net3l', root='~/.insightface/models'):
    from insightface.model_zoo.model_store import get_model_file
    _file = get_model_file("retinaface_%s" % name, root=root)
    return Detector(_file, rac)


class ThreadedVideoDetector(VideoDetector):
    # detect_faces = threaded(VideoDetector.detect_faces)

    def __init__(self, args):
        # super().__init__(mx_context, args)
        self.args = args
        # self.ctx = mx.cpu()
        self.ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu(0)
        self.dataset = None  # Collection of features of known names
        self.names = {}  # Names of known person
        self.persons = []  # List of person detected
        self.det_model = get_retinaface('mnet025_v2')
        self.rec_model = model_zoo.get_model('arcface_r100_v1')
        if self.args.gpu < 0:
            self.det_model.prepare(-1)
            self.rec_model.prepare(-1)
        else:
            self.det_model.prepare(self.args.gpu)
            self.rec_model.prepare(self.args.gpu)
        self.renders = []

    def detect(self):
        cap = cv2.VideoCapture(self.args.in_file)  # Create a VideoCapture object
        frame_w, frame_h = int(cap.get(3)), int(cap.get(4))  # Convert resolutions from float to integer.
        batch_size = self.args.batch_size
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        batch_time = np.array([])
        n_batches = int(np.ceil(total_frames / batch_size))
        # frames = self.get_buff(cap, batch_size)  # get first buffer
        # handle_faces = self.detect_faces(frames)  # start thread on face detection (first time)
        for _ in tqdm(range(n_batches)):
            start = time()
            frames = self.get_buff(cap, batch_size)
            total_boxes, points = self.detect_faces(frames)
            self.buffered_identify(frames, total_boxes, points)
            # total_boxes, points = handle_faces.result()  # wait until batch is finished
            # handle_identify = self.buffered_identify(frames, total_boxes, points)  # start identification thread
            # frames = self.get_buff(cap, batch_size)  # get new frames
            # handle_faces = self.detect_faces(frames)  # start thread on face detection
            # handle_identify.join()  # wait for identify, rendered images
            batch_time = np.append(batch_time, time() - start)
        cap.release()
        return self.renders, {'w': frame_w, 'h': frame_h, 'batch_exec': batch_time.mean(), 'fps': total_frames/batch_time.sum()}

    def prepare_faces(self, dataset_name='dataset.pkl'):
        image_names = os.listdir(self.args.faces_dir)
        face_names = set([x.split('_')[0] for x in image_names])
        dataset = {}
        for name in face_names:
            images = np.stack([cv2.imread(os.path.join(self.args.faces_dir, iname)) for iname in image_names if name in iname])
            _, points = self.det_model.detect(images, threshold=self.args.threshold_det)
            cr_images = [face_align.norm_crop(img, lndm[0]) for img, lndm in zip(images, points) if lndm.shape[0] == 1]
            embeddings = [self.rec_model.get_embedding(img).flatten() for img in cr_images]
            normed_embedding = [embedding / norm(embedding) for embedding in embeddings]
            dataset[name] = np.stack(normed_embedding)

        dataset_path = os.path.abspath(os.path.join(self.args.faces_dir, '..'))

        with open(dataset_path + '/' + dataset_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    # @no_return_thread
    def buffered_identify(self, frames, total_boxes, points):
        for frame, box, point in zip(frames, total_boxes, points):
            self.identify(frame, box, point)
            render = self.draw_names(frames)
            self.renders.append(render)

    @staticmethod
    def get_buff(cap, b):
        buff = []
        for _ in range(b):
            r, img = cap.read()
            if not r and img is None:
                break
            else:
                buff.append(img)
        return np.stack(buff)
