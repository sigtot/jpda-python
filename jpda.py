import numpy as np
import cv2 as cv

from target import Target

vid = cv.VideoCapture("PETS09-S2L1.mp4")
bg_subber = cv.createBackgroundSubtractorMOG2()
med_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
small_kernel = np.ones((3, 3), dtype=np.uint8)
while True:
    ret, frame = vid.read()
    if frame is None:
        break
    fg = bg_subber.apply(frame)
    cv.threshold(fg, 200, 255, cv.THRESH_BINARY, dst=fg)
    cleaned = cv.morphologyEx(fg, cv.MORPH_OPEN, med_kernel)
    dist, labels = cv.distanceTransformWithLabels(cleaned, cv.DIST_L2, 2)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    dist = cv.dilate(dist, med_kernel)
    dist = cv.morphologyEx(dist, cv.MORPH_DILATE, med_kernel, iterations=2)
    cv.imshow('Distance transform', dist)
    cv.threshold(dist, 0.6, 1.0, cv.THRESH_BINARY, dst=dist)
    dist_8u = dist.astype('uint8')
    contours, hierarchy = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    markers = np.zeros(dist.shape, dtype=np.float64)
    moments = [cv.moments(c) for c in contours]
    targets = [Target(int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])) for m in moments]
    for t in targets:
        cv.circle(markers, (t.x, t.y), 7, (255, 255, 255), -1)
        cv.circle(frame, (t.x, t.y), 7, (255, 255, 255), -1)
    cv.imshow("Markers", markers)
    cv.imshow("Frame", frame)

    key = cv.waitKey(30)
    if key == "q" or key == 27:
        break
