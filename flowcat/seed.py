import logging
import random
import numpy


LOGGER = logging.getLogger(__name__)

SEED = None


def set_seed(seed):
    """This sets a seed for multiple libraries used in flowcat.

    Args:
        seed - Number used to initialize, pass None to set no seed.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    LOGGER.info("Setting seed %s for random and numpy", seed)

    # setting the global seed for non-trivial seed components
    global SEED
    SEED = seed
