import numpy as np
import cv2 as cv

from cvtargetmaker import CVTargetMaker

vid = cv.VideoCapture("PETS09-S2L1.mp4")
bg_subber = cv.createBackgroundSubtractorMOG2()
med_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
small_kernel = np.ones((3, 3), dtype=np.uint8)

Q = 40*np.eye(4, 4)
R = 10*np.eye(2, 2)
T = 0.1
g = 8
target_maker = CVTargetMaker(T, Q, R, g)
have_measurements = False
targets = []
gate = 40
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
    moments = [cv.moments(c) for c in contours]
    Z = [np.array([[np.float(m["m10"] / m["m00"])], [np.float(m["m01"] / m["m00"])]]) for m in moments]
    if len(Z) > 4 and not have_measurements:
        have_measurements = True
        for z in Z:
            targets.append(target_maker.new(z[0][0], z[1][0]))

    for t in targets:
        t.predict()
        eigs = np.linalg.eig(t.P[0:2, 0:2])

        cv.ellipse(img=frame,
                   center=(t.x[0], t.x[1]),
                   axes=(int(round(np.sqrt(eigs[0][0]))), int(round(np.sqrt(eigs[0][1])))),
                   angle=0,
                   startAngle=0,
                   endAngle=360,
                   color=(255, 0, 0),
                   thickness=2,
                   lineType=1)

    for z in Z:
        cv.circle(frame, (z[0], z[1]), 3, (0, 255, 0), -1)
        for t in targets:
            if np.linalg.norm(t.x[0:2] - z) < gate:
                t.update(z)
    cv.imshow("Frame", frame)

    key = cv.waitKey(30)
    if key == "q" or key == 27:
        break
