"""
Actual test won't be possible as copying image files to GitHub repo will use too much
memory.
"""

import pytest

from roman_lcs import Machine


def test_machine():
    """
    Dummy function to test error when no files are provided
    """
    with pytest.raises(TypeError) as excinfo:
        _ = Machine()
    assert excinfo.type is TypeError
