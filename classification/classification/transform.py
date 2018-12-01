import numpy as np


class Transformer:
    """Apply transformations to input data before further processing."""
    def __init__(self, transarg):
        if transarg == "sqrt":
            self.method = np.sqrt
        else:
            self.method = None

    def apply(self, dataview):
        dataview.values = self.method(dataview.values)
        return dataview
