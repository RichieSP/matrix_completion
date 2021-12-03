import abc
from matrix_completion import MatrixCompletion, TraceNorm, PCPursuit, Naive
import numpy as np
import random
from sample import Sample, LabeledSample
from termcolor import colored
import cvxpy as cp


random.seed(1)


def uniform_random_sample(X, size):
    n, m = X.shape
    indices = [(i, j) for j in range(m) for i in range(n)]
    omega = random.sample(indices, size)
    return LabeledSample(X, omega)


def print_sample(sample: Sample):
    n, m = sample.shape
    print(
        "\n".join(
            "\t".join(
                "{:.2f}".format(sample[(i, j)])
                if (i, j) in sample.omega()
                else colored("???", "red")
                for j in range(m)
            )
            for i in range(n)
        )
    )


def print_matrix(M: np.ndarray, sample: Sample):
    n, m = M.shape
    print(
        "\n".join(
            "\t".join(
                "{:.2f}".format(M[i][j])
                if (i, j) in sample.omega()
                else colored("{:.2f}".format(M[i][j]), "green")
                for j in range(m)
            )
            for i in range(n)
        )
    )


class Evaluator:
    def __init__(self, mc: MatrixCompletion, verbose = False):
        self._mc = mc
        self._verbose = verbose

    def test_all(self):
        self.test_small()
        self.test_approx_zeros()
        self.test_approx_ones()
        self.test_geographical()
        self.test_generated()

    def test(self, sample):
        if self._verbose:
            print(f"Testing sample:")
            print_sample(sample)

        X = self._mc.solve(sample)

        if self._verbose:
            print(f"Recovered matrix:")
            print_matrix(X, sample)

        sample.evaluate(X)

    def test_small(self):
        X = np.ones((4, 4))
        sample = uniform_random_sample(X, 12)
        self.test(sample)

    def test_approx_zeros(self):
        X = np.zeros((10, 10))
        # perturb sparse subset
        indices = [(i, j) for j in range(10) for i in range(10)]
        for i, j in random.sample(indices, 10):
            X[i][j] = random.random() * 0.2 - 0.1
        sample = uniform_random_sample(X, 80)
        self.test(sample)

    def test_approx_ones(self):
        n = 20
        X = np.ones((n, n))
        # perturb sparse subset
        indices = [(i, j) for j in range(n) for i in range(n)]
        for i, j in random.sample(indices, n):
            X[i][j] = random.random() * 0.2 - 0.1
        sample = uniform_random_sample(X, int(0.8 * n * n))
        self.test(sample)

    def test_geographical(self):
        X = np.loadtxt(open("data/Paris-Distances.csv", "rb"), delimiter=",", skiprows=1)
        sample = uniform_random_sample(X, 80)
        self.test(sample)

    def test_generated(self):
        X = np.loadtxt(open("data/exact.csv", "rb"), delimiter=",")
        sample = uniform_random_sample(X, 80)
        self.test(sample)

    def test_num_samples(self, perturb = True):
        n = 30
        vecs = [np.random.normal(size = n).reshape(-1, 1) for i in range(3)]
        X_original = sum([v @ v.T for v in vecs])
        N = X_original.shape[0] * X_original.shape[1]

        # perturb sparse subset
        if perturb:
            indices = [(i, j) for j in range(n) for i in range(n)]
            X_sparse = np.zeros((n, n))
            for i, j in random.sample(indices, 10):
                X_sparse[i][j] = random.random() * 0.2 - 0.1
            X_original += X_sparse

        x = list(range(1, N))
        y = []

        for i in range(1, N):
            X = np.copy(X_original)
            sample = uniform_random_sample(X, i)
            X = self._mc.solve(sample)
            loss = sample.evaluate(X)
            y.append(loss)

        print(x)
        print(y)


#mc = TraceNorm()
mc = PCPursuit()
#mc = Naive()
evaluator = Evaluator(mc, verbose = True)
#evaluator.test_all()
evaluator.test_num_samples(perturb = True)
