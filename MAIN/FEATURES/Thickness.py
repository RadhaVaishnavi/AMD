import cv2
import numpy as np


def thickness_calc(img):
    cnts = cv2.findContours(img, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    Thickness = []
    for c in cnts:
        two_points = []
        coord_x = []
        coord_y = []
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, False)
        if area > 1 and perimeter > 1:
            x, y, w, h = cv2.boundingRect(c)
            cx = int((x + (w / 2))) - 5
            cy = int((y + (h / 2))) + 15
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for a in range(cx, cx + 70):
                coord_x.append(a)
                coord_y.append(cy)
            coord = list(zip(coord_x, coord_y))
            arrayxy = np.array(coord)

            for b in arrayxy:
                two_points.append(b)
        pointsarray = np.array(two_points)
        Thickness.append(np.mean(pointsarray))

    return np.mean(np.nan_to_num(Thickness))

