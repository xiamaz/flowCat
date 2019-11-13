from flowcat import utils


def predict(case, model: utils.URLPath, output: utils.URLPath):
    """Generate predictions and plots for a single case.

    Args:
        case: Single case with FCS files.
        model: Path to model containing CNN and SOMs.
        output: Destination for plotting.
    """
    raise NotImplementedError
