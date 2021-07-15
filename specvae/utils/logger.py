#!/usr/bin/env python3

"""Basic logging module."""

import logging

DEFAULT_LOGGER_STRING_FORMAT = '%(asctime)s %(levelname)-8s' \
    '[%(filename)s:%(lineno)d] %(message)s'


def setup_logger(
    name, logger_string_format=DEFAULT_LOGGER_STRING_FORMAT
):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(logger_string_format)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler('debug.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.FileHandler('warn.log')
    handler.setLevel(logging.WARN)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


log = setup_logger('log')
