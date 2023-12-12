import matplotlib.pyplot as plt
import numpy as np
from definitions import Example


def main():
    ex = Example(name="beam")
    svals = np.load(ex.singular_values)

    plt.semilogy(np.arange(svals.size), svals / svals[0], 'k-x')
    plt.show()


if __name__ == "__main__":
    main()
