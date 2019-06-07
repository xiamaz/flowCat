"""
Overview of cases dataset:

- number of samples
"""
import time
import contextlib
from argparse import ArgumentParser
import flowcat


@contextlib.contextmanager
def timing(label):
    time_a = time.time()
    yield
    time_b = time.time()
    time_d = time_b - time_a
    print(f"{label} time: {time_d}s")


def main(args):
    cases = flowcat.CaseCollection.from_path(args.path)
    print(cases)
    print(cases.group_count)
    with timing("Selected markers"):
        markers = cases.get_selected_markers()
    print(markers)

    with timing("Filter train"):
        train_cases, _ = cases.filter_reasons(
            date=(None, "2018-06-30"),
            selected_markers=markers,
            materials=flowcat.ALLOWED_MATERIALS,
            tubes=(1, 2, 3))
    print(train_cases)
    print(train_cases.group_count)

    with timing("Filter test"):
        test_cases, _ = cases.filter_reasons(
            date=("2018-07-01", None),
            selected_markers=markers,
            materials=flowcat.ALLOWED_MATERIALS,
            tubes=(1, 2, 3))
    print(test_cases)
    print(test_cases.group_count)

    print("Ratio", len(test_cases) / (len(train_cases) + len(test_cases)))

    with timing("Filter all"):
        all_cases, reasons = cases.filter_reasons(
            selected_markers=markers,
            materials=flowcat.ALLOWED_MATERIALS,
            tubes=(1, 2, 3))
    print(all_cases, len(reasons))

    newest = None
    for case in all_cases:
        if not newest:
            newest = case
        elif newest.date < case.date:
            newest = case
    print(newest)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "path",
        type=flowcat.utils.URLPath,
        help="Path to dataset",
    )
    main(PARSER.parse_args())
