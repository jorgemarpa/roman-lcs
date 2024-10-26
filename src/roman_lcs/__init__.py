#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os

__version__ = "0.1.0"

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .machine import Machine  # noqa
from .roman import RomanMachine

__all__ = ["Machine", "RomanMachine"]