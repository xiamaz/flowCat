from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser(
        prog="Clustering",
        description="Clustering preprocessing of flow cytometry data."
    )
    parser.add_argument("-o", "--output", help="Output file directory")
    parser.add_argument("--refcases", help="Json with list of reference cases.")
    parser.add_argument("--tubes", help="Selected tubes.")
    parser.add_argument(
        "--groups", help="Semicolon separated list of groups"
    )
    parser.add_argument(
        "--refnormal", help="Exclude normal cohort in consensus generation.",
        action="store_true"
    )
    parser.add_argument(
        "--num", help="Number of selected cases", default=5, type=int
    )
    parser.add_argument(
        "--upsampled", help="Number of cases per cohort to be upsampled.",
        default=300, type=int
    )
    parser.add_argument("-i", "--input", help="Input case json file.")
    parser.add_argument("-t", "--temp", help="Temp path for file caching.")
    parser.add_argument(
        "--plotdir", help="Plotting directory", default="plots"
    )
    parser.add_argument(
        "--plot", help="Enable plotting", action="store_true"
    )
    parser.add_argument(
        "--pipeline", help="Select pipeline type.", default="normal"
    )
    parser.add_argument(
        "--prefit", help="Preprocessing for each case in fitting.",
        default="normal"
    )
    parser.add_argument(
        "--pretrans", help="Preprocessing for each case in transform.",
        default="normal"
    )
    parser.add_argument("--bucketname", help="S3 Bucket with data.")
    return parser
