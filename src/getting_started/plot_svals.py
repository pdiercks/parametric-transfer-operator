import matplotlib.pyplot as plt
import numpy as np
from definitions import BeamData


def main():
    ex = BeamData(name="beam")
    svals = np.load(ex.singular_values)

    plt.semilogy(np.arange(svals.size), svals / svals[0], 'k-x')
    plt.show()


if __name__ == "__main__":
    main()
