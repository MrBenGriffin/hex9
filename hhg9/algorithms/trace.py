# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Trace utilities for Hex9 (H9) project. Useful when debugging.
"""
import os
import traceback


def caller(msg):
    """Print the caller. Skip this, and the call to this!"""
    dx = traceback.extract_stack()[-3]
    sx = f'{msg}: {os.path.relpath(dx.filename, "..")}, line {dx.lineno}; at {dx.line}'
    print(sx)
