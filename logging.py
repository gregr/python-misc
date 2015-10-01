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


class Meter(object):
    def __init__(self, threshold, msg_template='%d'):
        self.msg_template = msg_template
        self.threshold = threshold
        self.current = 0
        self.total = 0

    def log(self):
        self.total += self.current
        self.current = 0
        logging.info(self.msg_template, self.total)

    def inc(self, amount):
        self.current += amount
        if self.current >= self.threshold:
            self.log()
