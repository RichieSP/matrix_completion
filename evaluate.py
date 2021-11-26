import abc
from matrix_completion import MatrixCompletion, TraceNorm
from sample import Sample, LabeledSample
import random
import numpy as np
from termcolor import colored


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
        self.test_geographical()

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

    def test_geographical(self):
        X = np.loadtxt(open("data/Paris-Distances.csv", "rb"), delimiter=",", skiprows=1)
        sample = uniform_random_sample(X, 80)
        self.test(sample)


mc = TraceNorm()
evaluator = Evaluator(mc, verbose = True)
evaluator.test_all()
