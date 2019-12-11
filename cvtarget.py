import numpy as np


class CVTarget:
    def __init__(self, x: np.float, y: np.float, F: np.array, H: np.array, Q: np.array, R: np.array, g: np.float):
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
        self.S = self.innov_cov(self.H, self.P, self.R)
        self.g = g

    @staticmethod
    def innov_cov(H, P, R):
        return H@P@np.transpose(H) + R

    def predict(self):
        self.x = self.F@self.x
        self.P = self.F@self.P@np.transpose(self.F) + self.Q
        self.S = self.innov_cov(self.H, self.P, self.R)

    def update(self, z):
        ny = z - self.zpred()
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)

        self.x = self.x + W@ny
        self.P = (np.eye(4, 4) - W@self.H)@self.P

    def zpred(self):
        return self.H@self.x
    
    def gate(self, Z):
        zpred = self.zpred()  # declare this first for efficiency
        return [z for z in Z if np.transpose(z - zpred)@np.linalg.inv(self.S)@(z - zpred) < self.g**2]

