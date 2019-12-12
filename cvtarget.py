import numpy as np

from utils import gauss


class CVTarget:
    def __init__(self,
                 x: np.float,
                 y: np.float,
                 F: np.array,
                 H: np.array,
                 Q: np.array,
                 R: np.array,
                 g: np.float,
                 P_D: np.float,
                 lam: np.float):
        self.lam = lam
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
        self.P_D = P_D

    @staticmethod
    def innov_cov(H, P, R):
        return H@P@np.transpose(H) + R

    def predict(self):
        self.x = self.F@self.x
        self.P = self.F@self.P@np.transpose(self.F) + self.Q
        self.S = self.innov_cov(self.H, self.P, self.R)

    def update(self, Z: list):
        self.S = self.innov_cov(self.H, self.P, self.R)
        zpred = self.zpred()
        innovations = []
        betas_unorm = []
        for z in Z:
            if z.size != 2:
                raise Exception("z has wrong dimension", z)
            innovations.append(z - zpred)
            betas_unorm.append(self.P_D * gauss(z, zpred, self.S))
        betas = betas_unorm / np.sum(betas_unorm)

        # Reduce the mixture to a single innovation weighted by betas
        ny = np.zeros_like(zpred)
        for j, assoc_ny in enumerate(innovations):
            ny += betas[j] * assoc_ny
        W = self.P@np.transpose(self.H)@np.linalg.inv(self.S)

        self.x = self.x + W@ny

        beta_boi = 0
        for j, assoc_ny in enumerate(innovations):
            beta_boi += betas[j]*assoc_ny@assoc_ny.T
        sprd_innov = W@(beta_boi - ny@ny.T)@W.T
        beta_0 = self.lam * (1 - self.P_D)

        self.P = self.P - (1 - beta_0)*W@self.S@W.T + sprd_innov

    def zpred(self):
        return self.H@self.x
    
    def gate(self, Z):
        zpred = self.zpred()  # declare this first for efficiency
        gated = np.array([z for z in Z if np.transpose(z - zpred)@np.linalg.inv(self.S)@(z - zpred) < self.g**2])
        return gated.reshape((gated.size // 2, 2, 1))
