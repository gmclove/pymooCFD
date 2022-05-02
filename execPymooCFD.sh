#!/bin/bash
source activate pymooCFD
python -u $1 > opt.$$.out
