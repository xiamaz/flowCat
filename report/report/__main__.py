from argparse import ArgumentParser
from .report import generate_report


parser = ArgumentParser("Report generation tool.")
parser.add_argument("dir", help="Report directory to output plots and tables.")

args = parser.parse_args()

generate_report(args.dir)
