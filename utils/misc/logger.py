import logging
import sys

_logger_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def get_logger_verbose(name, filepath):
    return get_logger(name, filepath, level=logging.DEBUG, stdout=True)


def get_logger(name=None, filepath=None, level=logging.INFO, stdout=False):
    """
    makes a new or uses an existing logger
    :param name:
    :param filepath:
    :param level:
    :param stdout:
    :return:
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # only add handlers if these don't exist (this avoids repeated handlers)
    if not logger.hasHandlers():
        if stdout:
            logger.addHandler(logging.StreamHandler(sys.stdout))

        if filepath is not None:
            handler = logging.FileHandler(filepath)
            handler.setFormatter(_logger_formatter)
            logger.addHandler(handler)
            print(f"Made a file logger to \"{filepath}\"")

    return logger
