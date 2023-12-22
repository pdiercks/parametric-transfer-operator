from collections import defaultdict
from pymor.basic import StationaryModel, StationaryRBReductor, pod
from pymor.core.pickle import dump
from numpy import save


def main():
    from .tasks import beam
    from .fom import discretize_fom
    fom = discretize_fom(beam)
    parameter_space = fom.parameters.space((1., 2.))
    reductor = StationaryRBReductor(fom, product=fom.h1_0_semi_product, check_orthonormality=False)
    ntrain = 60
    num_modes = 20
    rom, svals = build_rom(fom, parameter_space, reductor, num_modes, ntrain)

    errors = defaultdict(list)
    ntest = 10
    test_set = parameter_space.sample_randomly(ntest)
    for mu in test_set:
        fom_data = fom.compute(solution=True, output=True, mu=mu)
        rom_data = rom.compute(solution=True, output=True, mu=mu)
        for key in ('solution', 'output'):
            if key == 'solution':
                ERR = fom_data.get(key) - reductor.reconstruct(rom_data.get(key))
                err = ERR.norm(fom.h1_0_semi_product)
            else:
                ERR = fom_data.get(key) - rom_data.get(key)
                err = ERR[0, 0]

            errors[key].append(err)

    for key in ('solution', 'output'):
        print(f"Max {key} error = {max(errors[key])} over test set of size {len(test_set)}")

    # write ROM and parameter space to disk
    with open(beam.reduced_model.as_posix(), 'wb') as f:
        dump((rom, parameter_space), f)

    # write singular_values to disk
    save(beam.singular_values, svals)


def build_rom(fom, parameter_space, reductor, basis_size, num_samples) -> StationaryModel:
    """Builds ROM"""

    # define training set
    training_set = parameter_space.sample_randomly(num_samples)
    snapshots = fom.solution_space.empty()
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    basis, singular_values = pod(snapshots, modes=basis_size, product=reductor.products['RB'])
    reductor.extend_basis(basis, method='trivial')

    rom = reductor.reduce()

    return rom, singular_values


if __name__ == "__main__":
    main()
