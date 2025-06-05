#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os

__version__ = "0.2.0"

def get_logger():
    """
    Creates a logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


log = get_logger()

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .machine import Machine  # noqa
from .roman import RomanMachine  # noqa

__all__ = ["Machine", "RomanMachine"]