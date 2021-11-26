import abc
from sample import Sample
import cvxpy as cp


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
    pass
