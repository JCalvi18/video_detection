import numpy as np
import cv2


class Person(object):
    def __init__(self):
        self.pre_point = None
        self.name = None
        self.l2_thresh = 100

    def l2_distance(self, point):
        # Compare actual point with previous point
        # If l2 distance > l2 threshold return False else the value
        l2 = (((point[0]-self.pre_point[0])**2)+((point[1]-self.pre_point[1])**2))**0.5
        if l2 > self.l2_thresh:
            return 0
        else:
            return l2


pad = 10
width = 500
x = np.arange(100)*5
y = [int(v) for v in x*np.sin(x)]

height = int(max(y)-min(y)+2*pad)
yoff = int(height/2)
frame = np.full((height, width+2*pad, 3), 255, dtype=np.uint8)
blue = [255, 0, 0]
red = [0, 0, 255]

person = Person()
person.pre_point = (x[0], yoff+y[0])

for i in range(1, len(x)):
    cp = frame.copy()
    point = (x[i], yoff+y[i])
    l2 = int(person.l2_distance(point))

    if l2:
        cv2.circle(cp, point, l2, blue, 1)
    else:
        cv2.circle(cp, point, person.l2_thresh, red, 1)
    cv2.line(cp, point, person.pre_point, blue)
    cv2.imshow('na', cp)
    person.pre_point = point
    if cv2.waitKey(275) & 0xFF == ord('q'):
        break

