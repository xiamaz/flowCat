"""
Overview of cases dataset:

- number of samples
"""
from argparse import ArgumentParser
import flowcat


def main(args):
    cases = flowcat.CaseCollection.from_path(args.path)
    print(cases)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument(
        "path",
        type=flowcat.utils.URLPath,
        help="Path to dataset",
    )
    main(PARSER.parse_args())
