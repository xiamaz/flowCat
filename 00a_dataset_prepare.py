#!/usr/bin/env python3
"""
Overview of cases dataset:

- number of samples
"""
import time
import contextlib
from argparse import ArgumentParser
import collections

import flowcat


@contextlib.contextmanager
def timing(label):
    time_a = time.time()
    yield
    time_b = time.time()
    time_d = time_b - time_a
    print(f"{label} time: {time_d}s")


@contextlib.contextmanager
def block(label):
    print(label)
    yield
    print("-" * len(label))


def deduplicate_cases_by_sureness(cases):
    """Remove duplicates by taking the one with the higher sureness score."""
    label_dict = collections.defaultdict(list)
    for case in cases:
        label_dict[case.id].append(case)
    deduplicated = []
    duplicates = []
    for same_id_cases in label_dict.values():
        same_id_cases.sort(key=lambda c: c.sureness, reverse=True)
        if len(same_id_cases) == 1 or same_id_cases[0].sureness > same_id_cases[1].sureness:
            deduplicated.append(same_id_cases[0])
        else:
            duplicates.append(same_id_cases[0].id)
            print(
                "DUP both removed: %s (%s), %s (%s)" % (
                    same_id_cases[0].id, same_id_cases[0].group, same_id_cases[1].id, same_id_cases[1].group
                ))
    if duplicates:
        print("%d duplicates removed" % len(duplicates))
    deduplicated_cases = flowcat.CaseCollection(deduplicated, **cases.config)
    deduplicated_cases.add_filter_step({"custom_filter": "deduplicate_cases_by_sureness"})
    return deduplicated_cases


def print_marker_ratios(counts, num, tube):
    print(f"Tube {tube}")
    for marker, count in counts.items():
        print(f"\t{marker}\t\t{count}/{num} ({count / num:.2})")


def get_selected_markers(cases, tubes, marker_threshold=0.9):
    """Get a list of marker channels available in all tubes."""
    selected_markers = {}
    for tube in tubes:
        marker_counts = collections.Counter(
            marker for t in [case.get_tube(tube) for case in cases]
            if t is not None for marker in t.markers)
        print_marker_ratios(marker_counts, len(cases), tube)
        # get ratio of availability vs all cases
        selected_markers[tube] = [
            marker for marker, count in marker_counts.items()
            if count / len(cases) > marker_threshold and "nix" not in marker
        ]
    selected_cases, _ = cases.filter_reasons(selected_markers=selected_markers)
    selected_cases.selected_markers = selected_markers
    return selected_cases


def print_diff(cases_a, cases_b):
    """Print differences in number of cases in a and number of cases in b."""
    num_a = len(cases_a)
    num_b = len(cases_b)
    diff_total = abs(num_a - num_b)
    print(f"Total: {num_a} → {num_b} -- |{diff_total}|")

    group_a = cases_a.group_count
    group_b = cases_b.group_count
    for key, value_a in group_a.items():
        value_b = group_b[key]
        diff_key = value_a - value_b
        print(f"{key}: {value_a} → {value_b} -- |{diff_key}|")


def preprocess_cases(cases: flowcat.CaseCollection, tubes=(1, 2, 3)):
    with block("Deduplicate cases by sureness"):
        deduplicated = deduplicate_cases_by_sureness(cases)
        print_diff(cases, deduplicated)
        cases = deduplicated

    with block("Filter cases on selected markers"):
        selected_cases = get_selected_markers(cases, tubes)
        print_diff(cases, selected_cases)
        cases = selected_cases
        print("Markers selected are:")
        print("\n".join(f"{k}: {v}" for k, v in cases.selected_markers.items()))

    with block("Filter train"):
        train_cases, _ = cases.filter_reasons(
            date=(None, "2018-06-30"),
            materials=flowcat.ALLOWED_MATERIALS,
            tubes=(1, 2, 3))
        print_diff(cases, train_cases)

    with block("Filter test"):
        test_cases, _ = cases.filter_reasons(
            date=("2018-07-01", None),
            materials=flowcat.ALLOWED_MATERIALS,
            tubes=(1, 2, 3))
        print_diff(cases, test_cases)

    train_labels = train_cases.labels
    test_labels = test_cases.labels
    train_not_test = all(t not in test_labels for t in train_labels)
    test_not_train = all(t not in train_labels for t in test_labels)
    assert train_not_test and test_not_train

    print("Ratio", len(test_cases) / (len(train_cases) + len(test_cases)))

    with timing("Filter all"):
        all_cases, reasons = cases.filter_reasons(
            materials=flowcat.ALLOWED_MATERIALS,
            tubes=(1, 2, 3))
        print_diff(cases, all_cases)
        reason_count = collections.Counter([r for _, rs in reasons for r in rs])
        print("Reasons are", ", ".join(f"{k}: {v}" for k, v in reason_count.items()))

    newest = None
    for case in all_cases:
        if not newest:
            newest = case
        elif newest.date < case.date:
            newest = case
    print("Newest case", newest)

    return train_cases, test_cases


def main(args):
    cases = flowcat.parser.get_dataset(args)
    train, test = preprocess_cases(cases)
    train.save(args.output / "train")
    test.save(args.output / "test")


if __name__ == "__main__":
    PARSER = ArgumentParser(
        description="Get overview of available data and create a dataset used for SOM transformation."
    )
    flowcat.parser.add_dataset_args(PARSER)
    PARSER.add_argument(
        "output", type=flowcat.utils.URLPath,
        help="Path to save output metadata",
    )
    main(PARSER.parse_args())
