import numpy as np


def main():
    from .tasks import example

    for k in range(11):
        sigma = np.load(example.hapod_singular_values(0, k))


if __name__ == "__main__":
    main()
