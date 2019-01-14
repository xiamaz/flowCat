import enum
from .. import loaders
from . import fcs_cnn, histo_nn, som_cnn


class ModelType(enum.Enum):

    HISTOGRAM = 1
    SOM = 2
    ETEFCS = 3

    @classmethod
    def __getitem__(cls, name):
        if isinstance(name, cls):
            return name
        name = name.upper()
        return super[name]


MODELS = {
    ModelType.HISTOGRAM: histo_nn.create_model_histo,
    ModelType.SOM: som_cnn.create_model_cnn,
    ModelType.ETEFCS: fcs_cnn.create_model_fcs,
}


def mtype_to_model(mtype):
    try:
        mtype = ModelType[mtype]
        return MODELS[mtype]
    except KeyError:
        raise KeyError(f"Invalid model type {mtype}")


def mtype_to_loader(mtype, tubes, data_args):
    mtype = ModelType[mtype]

    if mtype == ModelType.HISTOGRAM:
        args = data_args[loaders.CountLoader.__name__]
        loader = [loaders.CountLoader(tube=tube, **args) for tube in tubes]
    elif mtype == ModelType.SOM:
        args = data_args[loaders.Map2DLoader.__name__]
        loader = [loaders.Map2DLoader(tube=tube, **args) for tube in tubes]
    elif mtype == ModelType.ETEFCS:
        args = data_args[loaders.FCSLoader.__name__]
        loader = [loaders.FCSLoader(tubes=tubes, **args)]

    return loader
