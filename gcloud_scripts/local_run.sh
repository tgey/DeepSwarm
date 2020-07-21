#!/bin/bash

# python3 -m pip install -r requierements-dev-gpu.txt

python3 $1 -s $2
python3 $1 -s $2
python3 $1 -s $2
# python3 $1 -s $2
# python3 $1 -s $2

notify-send "TESTS FINISHED"