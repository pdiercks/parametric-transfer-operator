import csv
import json


def main(args):
    from .tasks import beam

    train_set_size = beam.lhs_options[args.configuration]["samples"]
    num_testvecs = beam.rrf_num_testvecs
    edges = ["bottom", "left", "right", "top"]
    with beam.heuristic_data("normal", args.configuration).open("r") as fh:
        data_ = fh.read()
        data = json.loads(data_)

    table = []
    top_row = ["Edge", "Basis size", "Basis size after RRF", "Neumann modes"]
    table.append(top_row)
    for edge in edges:
        row = []
        row.append(edge)
        row.append(data["final_basis_size"][edge])
        row.append(data["rrf_bases_length"][edge])
        row.append(data["neumann_modes"][edge])
        table.append(row)

    with beam.heuristic_table(args.configuration).open("w") as fh:

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
