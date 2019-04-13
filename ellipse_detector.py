"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
from math import sin, cos, pi
from typing import List, Optional

import cv2 as cv
import numpy as np


class EllipseDetector:
    def __init__(self, min_contour_size=5, use_canny=False, canny_threshold=100):
        """
        Split image to contours and try to fit ellipses into every countour. Contours may be separated
        by canny detector or simply pass binary image (thresholded). If using binary img, disable canny
        detector - otherwise it may generate strange results.
        :param min_contour_size: try to fit ellipses only to bigger contours than this (smaller ignored)
        :param use_canny: use canny detector to split image to contours; turn off for binary images
        :param canny_threshold: threshold for canny detector
        """
        self.min_contour_size = min_contour_size
        self.use_canny = use_canny
        self.canny_threshold = canny_threshold

    def detect(self, image, use_distance_filter=False, get_largest=False) -> Optional[np.ndarray]:
        # TODO comment + size filter?
        ellipses = self.detect_all_ellipses(image)
        if not ellipses:
            return None

        if get_largest:
            areas = [e[1][0]*e[1][1]*pi for e in ellipses]
            largest = areas.index(max(areas))
            ellipses = [ellipses[largest]]

        if use_distance_filter:  # FIXME axes major/minor length is full or half???
            distances = cv.distanceTransform(image, cv.DIST_L2, 5)
            _, _, _, max_loc = cv.minMaxLoc(distances)  # minVal, maxVal, minLoc, maxLoc
            for ellipse in ellipses:
                xp = max_loc[0]  # point coordinates
                yp = max_loc[1]
                xe = ellipse[0][0]  # ellipse center
                ye = ellipse[0][1]
                a = ellipse[1][0]/2  # ellipse major, minor axis
                b = ellipse[1][1]/2
                alpha = ellipse[2]+90  # ellipse rotation
                res = (cos(alpha)*(xp-xe) + sin(alpha)*(yp-ye))**2 / a**2 + (sin(alpha)*(xp-xe) - cos(alpha)*(yp-ye))**2 / b**2
                if res < 1:
                    return ellipse
        else:
            return ellipses[0]  # WARNING! By default return just first found ellipse

    def detect_all_ellipses(self, image) -> List[np.ndarray]:
        if self.use_canny:
            image = cv.Canny(image, self.canny_threshold, self.canny_threshold * 2)

        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # Find ellipses for each contour
        ellipses = [cv.fitEllipse(c) for c in contours if c.shape[0] >= self.min_contour_size]

        return ellipses
