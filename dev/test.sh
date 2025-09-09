#!/bin/bash

# Project must have been installed in editable mode for this to work.
# For namespace package discovery need to pass the path to the part of
# the namespace that has an __init__.py as the start directory.
python -m unittest discover -s cherab.solps -t . $1 $2 $3 $4 $5
