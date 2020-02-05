import numpy as np
import cv2

pose_titles = ['Ret', 'L', 'R', 'U', 'D']


def check_large_pose(landmark, bbox):
    assert landmark.shape == (5, 2)
    assert len(bbox) == 4

    def get_theta(base, x, y):
        vx = x - base
        vy = y - base
        vx[1] *= -1
        vy[1] *= -1
        tx = np.arctan2(vx[1], vx[0])
        ty = np.arctan2(vy[1], vy[0])
        d = ty - tx
        d = np.degrees(d)
        if d < -180.0:
            d += 360.
        elif d > 180.0:
            d -= 360.0
        return d

    landmark = landmark.astype(np.float32)

    theta1 = get_theta(landmark[0], landmark[3], landmark[2])
    theta2 = get_theta(landmark[1], landmark[2], landmark[4])
    # print(va, vb, theta2)
    theta3 = get_theta(landmark[0], landmark[2], landmark[1])
    theta4 = get_theta(landmark[1], landmark[0], landmark[2])
    theta5 = get_theta(landmark[3], landmark[4], landmark[2])
    theta6 = get_theta(landmark[4], landmark[2], landmark[3])
    theta7 = get_theta(landmark[3], landmark[2], landmark[0])
    theta8 = get_theta(landmark[4], landmark[1], landmark[2])
    # print(theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8)
    left_score = 0.0
    right_score = 0.0
    up_score = 0.0
    down_score = 0.0
    if theta1 <= 0.0:
        left_score = 10.0
    elif theta2 <= 0.0:
        right_score = 10.0
    else:
        left_score = theta2 / theta1
        right_score = theta1 / theta2
    if theta3 <= 10.0 or theta4 <= 10.0:
        up_score = 10.0
    else:
        up_score = max(theta1 / theta3, theta2 / theta4)
    if theta5 <= 10.0 or theta6 <= 10.0:
        down_score = 10.0
    else:
        down_score = max(theta7 / theta5, theta8 / theta6)
    mleft = (landmark[0][0] + landmark[3][0]) / 2
    mright = (landmark[1][0] + landmark[4][0]) / 2
    box_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    ret = 0
    if left_score >= 3.0:
        ret = 1
    if ret == 0 and left_score >= 2.0:
        if mright <= box_center[0]:
            ret = 1
    if ret == 0 and right_score >= 3.0:
        ret = 2
    if ret == 0 and right_score >= 2.0:
        if mleft >= box_center[0]:
            ret = 2
    if ret == 0 and up_score >= 2.0:
        ret = 3
    if ret == 0 and down_score >= 5.0:
        ret = 4
    return ret, left_score, right_score, up_score, down_score


def draw_pose(frame, points, bbox):
    for i in range(bbox.shape[0]):
        box = bbox[i].astype(np.int)
        # color = (0, 0, 255)
        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        if points is not None:
            landmark5 = points[i].astype(np.int)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                if l == 2:
                    color = (255, 0, 0)

                cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

            poses = check_large_pose(landmark5, box)

            y = 450
            for t, vl in zip(pose_titles, poses):
                txt = str(t + ':' + str(vl)) if type(vl) == int else str(t + ':' + str(np.round(vl, 2)))
                cv2.putText(frame, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                y += 60
    return frame


def debug(frame):
    cv2.imshow('na', frame)
    cv2.waitKey(10)


def draw_rect(frame, b):
    cv2.rectangle(frame, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), (255, 255, 255), 2)