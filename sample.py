import abc
import numpy as np
from termcolor import colored


class Sample(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def shape(self):
        raise NotImplementedError("Implement me")

    @abc.abstractmethod
    def omega(self):
        raise NotImplementedError("Implement me")

    @abc.abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError("Implement me")

    def items(self):
        return [(key, self[key]) for key in self.omega()]

    @abc.abstractmethod
    def evaluate(self, X):
        raise NotImplementedError("Implement me")


class LabeledSample(Sample):
    def __init__(self, M, omega):
        self._M = M
        self._omega = omega

    @property
    def shape(self):
        return self._M.shape

    def omega(self):
        return self._omega

    def __getitem__(self, key):
        i, j = key
        return self._M[i][j]

    def evaluate(self, X):
        assert X.shape == self.shape

        loss = np.linalg.norm(X - self._M, ord = "fro")
        print(f"Loss (Frobenius norm): {colored('{:.4f}'.format(loss), 'blue')}")
        return loss


class UnlabeledSample(Sample):
    def __init__(self, mapping, n, m):
        self._mapping = mapping
        self._n = n
        self._m = m

    @property
    def shape(self):
        return (self._n, self._m)

    def omega(self):
        return self._mapping.keys()

    def __getitem__(self, key):
        if key not in self.omega():
            raise KeyError(f"{key} not in the given sample")

        return self._mapping[key]
