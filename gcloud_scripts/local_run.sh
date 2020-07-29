#!/bin/bash

python3 -m pip install -r requierements-dev-cpu.txt

for v in $(seq 1 $3)
do
    python3 $1 -s $2
done

notify-send "TESTS FINISHED"