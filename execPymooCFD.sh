#!/bin/bash
source activate pymooCFD
python $1 > opt.$$.out
