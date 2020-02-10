from __future__ import division
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from insightface.model_zoo.face_detection import FaceDetector, anchors_plane, clip_pad, bbox_pred, landmark_pred



class Detector(FaceDetector):
    def __init__(self, _file, rac):
        super().__init__(_file, rac)

    def detect(self, images: np.array, threshold=0.5, scale=1.0):

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
        return net_out

    def post_process(self, net_out, threshold=0.5, scale=1.0):
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
            scores = scores[:, self._num_anchors['stride%s' % s]:, :, :]
            idx += 1
            bbox_deltas = net_out[idx].asnumpy()

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]
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
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A
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
                landmark_deltas = net_out[idx].asnumpy()
                landmark_deltas = clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
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

        return det, landmarks


def get_retinaface(name, rac='net3', root='~/.insightface/models', **kwargs):
    from insightface.model_zoo.model_store import get_model_file
    _file = get_model_file("retinaface_%s"%name, root=root)
    return Detector(_file, rac)