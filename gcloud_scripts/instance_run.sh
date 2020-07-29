#!/bin/bash

python3.6 -m pip install -r requierements-dev-gpu.txt

for v in $(seq 1 $3)
do


python3.6 $1 -s $2
python3.6 google_drive_sync.py

done

sudo poweroff
