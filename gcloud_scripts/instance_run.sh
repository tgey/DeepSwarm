#!/bin/bash

python3.6 -m pip install -r requierements-dev-gpu.txt

python3.6 $1 -s $2
python3.6 google_drive_sync.py

python3.6 $1 -s $2
python3.6 google_drive_sync.py

python3.6 $1 -s $2
python3.6 google_drive_sync.py

python3.6 $1 -s $2
python3.6 google_drive_sync.py

python3.6 $1 -s $2
python3.6 google_drive_sync.py

sudo poweroff
