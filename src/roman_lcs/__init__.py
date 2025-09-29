#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
from importlib import metadata

__version__ = metadata.version("roman_lcs")

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .machine import Machine  # noqa
from .roman import RomanMachine  # noqa

__all__ = ["Machine", "RomanMachine"]
