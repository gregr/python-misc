from __future__ import absolute_import
import logging
import time

default_level = logging.INFO
default_format = '[%(asctime)s %(levelname)s] %(message)s'
default_datefmt = '%Y%m%d %H:%M:%S'


def config(level=default_level,
           format=default_format,
           datefmt=default_datefmt,
           use_utc=True):
    logging.basicConfig(
        level=level, format=default_format, datefmt=default_datefmt)
    if use_utc:
        logging.Formatter.converter = time.gmtime
