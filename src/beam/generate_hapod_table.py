import csv
import json
import numpy as np


def main(args):
    from .tasks import beam

    train_set_size = beam.lhs_options[args.configuration]["samples"]
    num_testvecs = beam.rrf_num_testvecs
    edges = ["bottom", "left", "right", "top"]
    rrf_bases_length = np.load(beam.hapod_rrf_bases_length("normal", args.configuration))
    avg_bases_length = {}
    for edge in edges:
        avg_bases_length[edge] = np.around(np.average(rrf_bases_length[edge]), decimals=0)

    with beam.pod_data("normal", args.configuration).open("r") as fh:
        data = fh.read()
        pod_data = json.loads(data)

    basis_size = {}
    num_snapshots = {}
    for edge, data in pod_data.items():
        basis_size[edge] = data[0]
        num_snapshots[edge] = data[1]

    table = []
    top_row = ["Edge", "Basis size", "Snapshots", "Average basis size after RRF"]
    table.append(top_row)
    for edge in edges:
        row = []
        row.append(edge)
        row.append(basis_size[edge])
        row.append(num_snapshots[edge])
        row.append(avg_bases_length[edge])
        table.append(row)

    with beam.hapod_table(args.configuration).open("w") as fh:

        fh.write(f"# {train_set_size=}, {num_testvecs=}\n")
        csv_writer = csv.writer(fh, delimiter=",")
        for row in table:
            csv_writer.writerow(row)


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configuration",
        type=str,
        help="The type of oversampling problem.",
        choices=("inner", "left", "right"),
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
