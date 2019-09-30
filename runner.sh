#!/bin/bash
cd /python-env/mldriver-discrete-steering

while getopts ":c:e:" opt; do
  case $opt in
    e)
      pipenv shell
      xvfb-run -a -s "-screen 0 128x128x24" -- python main.py $OPTARG
      ;;
    c)
      cp "output/config/$OPTARG" consts.py
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done