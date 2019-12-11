import numpy as np


class CVTarget:
    def __init__(self, x: np.float, y: np.float, F: np.array, H: np.array, Q: np.array, R: np.array):
        self.x = np.array([
            [x],
            [y],
            [0],
            [0],
        ])
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = np.eye(4, 4)

    def predict(self):
        self.x = self.F@self.x
        self.P = self.F@self.P@np.transpose(self.F) + self.Q

    def update(self, z):
        ny = z - self.H@self.x
        S = self.H@self.P@np.transpose(self.H) + self.R
        W = self.P@np.transpose(self.H)@np.linalg.inv(S)

        self.x = self.x + W@ny
        self.P = (np.eye(4, 4) - W@self.H)@self.P

