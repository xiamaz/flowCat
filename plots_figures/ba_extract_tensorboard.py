"""
Extract information from tensorboard into a format easier usable for
custom plots.
"""
import sys
import pathlib
import argparse

import tensorflow as tf

# add the project as search path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from flowcat import utils


def read_tensorboard(path):
    """Read tensorboard data into an easier format to work on."""
    data = {}
    for event in tf.train.summary_iterator(str(path.get())):
        print(f"New event {event.step}")
        assert event.step not in data
        stepdata = {}

        for value in event.summary.value:
            print(value.tag)
            vtype = value.WhichOneof("value")
            if vtype == "image":
                vdata = value.image
            elif vtype == "simple_value":
                vdata = value.simple_value
            else:
                raise TypeError(vtype)
            stepdata[value.tag] = vdata

        if stepdata:
            data[event.step] = stepdata
    return data


def save_tensorboard(data, path):
    """Write tensorboard log data into a separate directory."""
    pass


def get_args():
    """Get commandline arguments."""
    parser = argparse.ArgumentParser(description="Dump data from tensorboard")
    parser.add_argument("input", help="Tensorboard file.", type=utils.URLPath)
    parser.add_argument("output", help="Output directory to dump to.", type=utils.URLPath)
    return parser.parse_args()


def main():
    args = get_args()

    tbdata = read_tensorboard(args.input)
    save_tensorboard(tbdata, args.output)


if __name__ == "__main__":
    main()
