import abc
from matrix_completion import MatrixCompletion, TraceNorm, PCPursuit, Naive
import numpy as np
import random
from sample import Sample, LabeledSample
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

    def test_geographical(self):
        X = np.loadtxt(open("data/Paris-Distances.csv", "rb"), delimiter=",", skiprows=1)
        sample = uniform_random_sample(X, 80)
        self.test(sample)

    def test_generated(self):
        X = np.loadtxt(open("data/exact.csv", "rb"), delimiter=",")
        sample = uniform_random_sample(X, 80)
        self.test(sample)

    def test_num_samples(self):
        x = list(range(1, 100))
        y = []

        for i in range(1, 100):
            X = np.loadtxt(open("data/exact.csv", "rb"), delimiter=",")
            sample = uniform_random_sample(X, i)
            X = self._mc.solve(sample)
            loss = sample.evaluate(X)
            y.append(loss)

        print(x)
        print(y)


#mc = TraceNorm()
#mc = PCPursuit(alpha = 3.)
mc = Naive()
evaluator = Evaluator(mc, verbose = True)
#evaluator.test_all()
#evaluator.test_generated()
evaluator.test_num_samples()
