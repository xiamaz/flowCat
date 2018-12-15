# pylint: skip-file
# flake8: noqa
"""
Extract information from tensorboard into a format easier usable for
custom plots.
"""
import sys
import pathlib
import argparse
import io
import collections

from PIL import Image
import pandas as pd

import tensorflow as tf

# add the project as search path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from flowcat import utils


def read_tensorboard(path):
    """Read tensorboard data into an easier format to work on."""
    imgdata = {}
    simpledata = {}
    for event in tf.train.summary_iterator(str(path.get())):
        simplestep = {}
        imgstep = {}
        for value in event.summary.value:
            vtype = value.WhichOneof("value")
            if vtype == "image":
                colorspace = value.image.colorspace
                img = Image.open(io.BytesIO(value.image.encoded_image_string))
                imgstep[value.tag] = img
            elif vtype == "simple_value":
                vdata = value.simple_value
                simplestep[value.tag] = vdata
            else:
                raise TypeError(vtype)
        if simplestep:
            simpledata[event.step] = simplestep
        if imgstep:
            imgdata[event.step] = imgstep

    simpledata = pd.DataFrame.from_dict(simpledata, orient="index")
    return simpledata, imgdata


def save_tensorboard(values, images, path):
    """Write tensorboard log data into a separate directory."""
    valuepath = path / "values.csv"
    utils.save_csv(values, valuepath)

    imgpath = path / "imgs"

    for epoch, data in images.items():
        for itype, img in data.items():
            ipath = imgpath / itype.replace("/", "-") / f"{epoch}.png"
            ipath.put(img.save)


def get_args():
    """Get commandline arguments."""
    parser = argparse.ArgumentParser(description="Dump data from tensorboard")
    parser.add_argument("input", help="Tensorboard file.", type=utils.URLPath)
    parser.add_argument("output", help="Output directory to dump to.", type=utils.URLPath)
    return parser.parse_args()


def main():
    args = get_args()

    tbvals, tbimgs = read_tensorboard(args.input)
    save_tensorboard(tbvals, tbimgs, args.output)


if __name__ == "__main__":
    main()
