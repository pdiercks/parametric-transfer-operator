import matplotlib.pyplot as plt
import numpy as np


def main():
    from parageom.tasks import example

    number_of_modes = example.validate_rom['num_modes']
    method = 'heuristic'
    ei = False
    nreal = 0

    num_modes = []
    kappa = []
    for N in number_of_modes:
        fpath = example.rom_condition(nreal, N, method=method, ei=ei)
        data = np.load(fpath)
        num_modes.append(N)
        kappa.append(data)

    kappa = np.hstack(kappa)

    cm = 1 / 2.54
    fig, ax = plt.subplots(figsize=(16 * cm, 9 * cm))
    ax.set_title(f'Condition number, min={kappa.min():1.2e}, max={kappa.max():1.2e}.', fontsize=11)
    ax.semilogy(num_modes, kappa, 'k-s')
    ax.set_ylabel('Condition number', fontsize=10)
    ax.set_xlabel('Local basis size', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.show()


if __name__ == '__main__':
    main()
