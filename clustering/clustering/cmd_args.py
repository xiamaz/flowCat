from argparse import ArgumentParser
from .collection import CaseCollection
from .utils import create_stamp, load_json, get_file_path


def create_parser():
    parser = ArgumentParser(
        prog="Clustering",
        description="Clustering preprocessing of flow cytometry data."
    )
    parser.add_argument("-o", "--output", help="Output file directory")
    parser.add_argument("-i", "--input", help="Input case directory.")
    parser.add_argument(
        "--refcases", help="List of reference cases."
    )
    parser.add_argument("--tubes", help="Selected tubes.")
    parser.add_argument(
        "--fitgroups", help="Semicolon separated list of groups in fitting"
    )
    parser.add_argument(
        "--transgroups", help="Semicolon separated list of groups in training"
    )
    parser.add_argument(
        "--fitnum", help="Number of selected cases", default=5, type=int
    )
    parser.add_argument(
        "--transnum", help="Number of cases per cohort to be upsampled.",
        default=-1, type=int
    )
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

    return parser


def get_args():
    """Create parser, parse arguments and return processed arguments."""
    return create_parser().parse_args()


class CmdArgs:
    def __init__(self):
        self._refcases = None

        self.args = get_args()

        self.output_directory = "{}_{}".format(
            self.args.output, create_stamp()
        )
        self.tubes = [int(t) for t in self.args.tubes.split(";")]
        self.collection_args = {
            "inputpath": self.args.input,
            "tubes": self.tubes
        }

        self.collection = CaseCollection(**self.collection_args)

        self.transform_args = {
            "labels": None,
            "num": self.args.transnum if self.args.transnum > 0 else None,
            "groups": list(
                map(lambda x: x.strip(), self.args.transgroups.split(";"))
            ) if self.args.transgroups else self.collection.groups,
        }

        self.train_args = {
            "labels": self.refcases,
            "num": self.args.fitnum if self.args.fitnum != -1 else None,
            "groups": list(
                map(lambda x: x.strip(), self.args.fitgroups.split(";"))
            ) if self.args.fitgroups else self.collection.groups,
        }

        self.pipeline_args = {
            "main": self.args.pipeline,
            "prefit": self.args.prefit,
            "pretrans": self.args.pretrans,
        }

    @property
    def refcases(self):
        """Return a list of ids or None if not specified. The latter will be
        ignored in filtering."""
        if self._refcases is None and self.args.refcases:
            with open(get_file_path(self.args.refcases), "r") as fobj:
                lines = [l.strip() for l in fobj.readlines()]
                lines = [l for l in lines if l]
            self._refcases = lines
        return self._refcases

    @property
    def clustering_args(self):
        return {
            "cases": self.collection,
            "train_opts": self.train_args,
            "transform_opts": self.transform_args,
            "pipeline_opts": self.pipeline_args,
            "output_path": self.output_directory,
        }
