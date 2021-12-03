import abc
from sample import Sample
import cvxpy as cp

import numpy as np
np.set_printoptions(precision=3, suppress=True)


class MatrixCompletion(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def solve(self, sample: Sample):
        raise NotImplementedError("Implement me")


class TraceNorm(MatrixCompletion):
    def solve(self, sample: Sample):
        n, m = sample.shape
        X = cp.Variable(sample.shape)

        objective = cp.Minimize(cp.norm(X, "nuc"))
        constraints = [
            X[i][j] == mij
            for (i, j), mij in sample.items()
            if 0 <= i < n and 0 <= j < m
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return X.value


class PCPursuit(MatrixCompletion):
    def solve(self, sample: Sample):
        n, m = sample.shape
        alpha = 1. / np.sqrt(max(n, m))
        L = cp.Variable(sample.shape)
        S = cp.Variable(sample.shape)

        loss = cp.norm(L, "nuc") + alpha * cp.pnorm(S, p=1)
        objective = cp.Minimize(loss)
        constraints = [
            L[i][j] + S[i][j] == mij
            for (i, j), mij in sample.items()
            if 0 <= i < n and 0 <= j < m
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return L.value + S.value

class Naive(MatrixCompletion):
    def solve(self, sample: Sample):
        avg = np.average([m for (i, j), m in sample.items()])
        X = avg * np.ones(sample.shape)
        for (i, j), m in sample.items():
            X[i][j] = m
        return X
