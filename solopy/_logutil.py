import logging
import os

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def get_logger(name, log_file=None, level=logging.INFO):
    """
    Return the named logger with exactly one console handler and at most one file handler.

    Safe to call once per class instance: handlers are never duplicated, and the logger
    does not propagate to the root logger (whose handlers would print every line twice).
    If ``log_file`` differs from the file currently attached, the old file handler is
    closed and replaced, so a long-lived process (e.g. a notebook handling several
    nights) writes each night to its own log. ``log_file=None`` leaves file handlers as they are.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT)

    # FileHandler subclasses StreamHandler, so test the exact type for the console handler.
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        target = os.path.abspath(os.fspath(log_file))
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename != target:
                logger.removeHandler(handler)
                handler.close()
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(target)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
