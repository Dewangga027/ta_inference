import numpy as np

class KalmanFilter1D:
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1):
        self.x = np.array([[0], [0]])  # [position, velocity]
        self.P = np.eye(2)
        self.Q = process_noise * np.eye(2)
        self.R = np.array([[measurement_noise]])
        self.H = np.array([[1, 0]])
        self.F = np.array([[1, 1],
                           [0, 1]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array([[z]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def step(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x[0, 0]