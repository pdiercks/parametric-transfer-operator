import numpy as np
import matplotlib.pyplot as plt


def main(nreal, config):
    from .tasks import example
    
    distr = example.distributions[0]
    for method in example.methods:
        infile = example.projerr(nreal, method, distr, config)
        err = np.load(infile)
        plt.semilogy(np.arange(err.size), err, label=method)
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    main(0, "left")
    main(0, "inner")
    main(0, "right")
