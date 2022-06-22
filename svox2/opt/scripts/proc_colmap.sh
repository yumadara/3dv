#!/bin/bash
set -e

# USAGE: bash proc_colmap.sh <dir of images>
python3 run_colmap.py $1 ${@:2}
python3 colmap2nsvf.py $1/sparse/0 
python3 create_split.py -y $1
