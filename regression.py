import numpy as np
import matplotlib.pyplot as plt


class spectralFilter(object):

    def __init__(self,
                 timeHorizon=10,
                 filterSize=10,
                 learningRate=0.001,
                 systemBound=0.5):
        self._n = 0
        self._m = 0
        self._T = timeHorizon
        self._k = filterSize
        self._eta = learningRate
        self._rm = systemBound
        self._ks = self._n * (self._k + 2) + self._m
        z = np.zeros((self._T, self._T))
        for i in range(self._T):
            for j in range(self._T):
                z[i, j] = 2 / ((i + j + 2) ** 3 - (i + j + 2))
        eigvalz, eigvecz = np.linalg.eig(z)
        self._sig = np.real(eigvalz[:self._k])
        self._phi = np.real(eigvecz[:, :self._k].T)
        self._x = np.zeros((self._T, self._n))
        self._y = 0
        self._weights = np.zeros((self._m, self._ks))
        # Debug variables
        self._grad = []
        self._fs = []


    def fit(self, input, output):
        self._n = len(input[0])
        if output.ndim == 1:
            output = output.reshape((len(output), 1))
        self._m = len(output[0])
        self._ks = self._n * (self._k + 2) + 1
        self._x = np.zeros((self._T, self._n))
        self._y = np.zeros((self._m))
        self._weights = np.zeros((self._m, self._ks))
        self._fs = np.zeros((self._m, self._ks))

        self.update(input, output)

    def update(self, input, output):
        if output.ndim == 1:
            output = output.reshape((len(output), 1))
        if input.ndim == 1:
            _ = self._update(input, output)
        else:
            for i in range(len(input)):
                print('\rTraining [ %.1f %%] ' % (100 * (i + 1) / len(input)), end="")
                _ = self._update(input[i], output[i])

    def _update(self, input, output):
        # Prediction
        for j in range(self._k):
            self._fs[-1, j * self._n : (j + 1) * self._n] = self._sig[j] ** 0.25 * np.dot(self._x[:self._T].T, self._phi[j])
        self._fs[-1, (j + 1) * self._n:] = np.hstack((self._x[0], input, self._y[0]))
        pred = np.dot(self._weights, np.nan_to_num(self._fs[-1], nan=0))
        # Update Weights
        self._grad = - 2 * self._eta * np.outer(pred - output, self._fs[-1])
        self._weights += self._grad
        if (np.linalg.norm(self._weights, 2, axis=1) >= self._rm).any():
            i = np.linalg.norm(self._weights, 2, axis=1) >= self._rm
            self._weights[i] = np.einsum('ab,a->ab', self._weights[i], self._rm / np.linalg.norm(self._weights[i], 2, axis=1))
        # Update memory
        self._y[-1] = output[0]
        self._y = np.roll(self._y, 1)
        self._x[-1] = input
        self._x = np.roll(self._x, 1, axis=0)
        self._fs = np.roll(self._fs, 1, axis=0)
        return pred

    def predict(self, input, output):
        if output.ndim == 1:
            output = output.reshape((len(output), 1))
        elif output.ndim == 0:
            output = np.array(output)
        if input.ndim == 1:
            return self._predict(input, output)
        else:
            prediction = np.zeros((len(input), self._m))
            for i in range(len(input)):
                print('\rPredicting [ %.1f %%] ' % (100 * (i + 1) / len(input)), end="")
                prediction[i] = self._predict(input[i], output[i])
            return prediction

    def _predict(self, input, output):
        fs = np.zeros((self._ks))
        for j in range(self._k):
            fs[j * self._n : (j + 1) * self._n] = self._sig[j] ** 0.25 * np.dot(self._x[:self._T].T, self._phi[j])
        fs[(j + 1) * self._n:] = np.hstack((self._x[0], input, self._y[0]))
        pred = np.dot(self._weights, np.nan_to_num(fs, nan=0))

        # Update weights! ONLINE LEARNING
        self._grad =  - 2 * self._eta * np.outer(np.dot(self._weights, self._fs[-1]) - np.flip(self._y), self._fs[-1])
        self._weights += self._grad
        if (np.linalg.norm(self._weights, 2, axis=1) >= self._rm).any():
            i = np.linalg.norm(self._weights, 2, axis=1) >= self._rm
            self._weights[i] = np.einsum('ab,a->ab', self._weights[i],
                                         self._rm / np.linalg.norm(self._weights[i], 2, axis=1))
        # Add to memory
        self._y[-1] = output
        self._y = np.roll(self._y, 1)
        self._x[-1] = input
        self._x = np.roll(self._x, 1, axis=0)
        self._fs[-1] = fs
        self._fs = np.roll(self._fs, 1, axis=0)
        return pred

    def shiftData(self, input, output, shift):
        samples = len(output)
        newOutput = np.zeros((samples - shift + 1, shift))
        for i in range(samples - shift + 1):
            newOutput[i] = output[i:i + shift]
        return input[:samples - shift + 1], newOutput
    
    def score(self, output, prediction):
        mean = np.mean(output)
        ss_tot = np.sum((output - mean) ** 2)
        ss_res = np.sum((output - prediction) ** 2)
        return 1 - ss_res / ss_tot

