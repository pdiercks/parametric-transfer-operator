from pymor.parameters.base import Parameters, ParameterSpace
from .lhs import sample_lhs
import numpy as np
import matplotlib.pyplot as plt



num_samples = 20
parameter_space = ParameterSpace(Parameters({"x": 2}), (0.0, 4.0))
first = sample_lhs(
    parameter_space,
    name="x",
    samples=num_samples,
    criterion="center",
    random_state=123321
)
second = sample_lhs(
        parameter_space, name="x", samples=num_samples, criterion="center", random_state=923421)
third = sample_lhs(
        parameter_space, name="x", samples=num_samples, criterion="center", random_state=9256)

def to_numpy(dataset):
    x = []
    for mu in dataset:
        x.append(mu.to_numpy())
    X = np.vstack(x)
    return X

X = to_numpy(first)
Z = to_numpy(second)
W = to_numpy(third)
breakpoint()

plt.plot(X[:, 0], X[:, 1], "bo")
plt.plot(Z[:, 0], Z[:, 1], "r+")
plt.plot(W[:, 0], W[:, 1], "gx")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
