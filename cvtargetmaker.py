import numpy as np

from cvtarget import CVTarget


class CVTargetMaker:
    F = np.empty((4, 4))
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])
    Q = np.eye(4, 4)
    R = np.eye(2, 2)

    def __init__(self, T: np.float, Q: np.array, R: np.array, g: np.float, P_D: np.float, lam: np.float):
        self.lam = lam
        self.Q = Q
        self.R = R
        self.F = np.array([
            [1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.g = g
        self.P_D = P_D

    def new(self, x: np.float, y: np.float):
        return CVTarget(x, y, self.F, self.H, self.Q, self.R, self.g, self.P_D, self.lam)
