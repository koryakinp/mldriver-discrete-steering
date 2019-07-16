#!/bin/bash

while getopts ":e:" opt; do
  case $opt in
    e)
      xvfb-run -a -s "-screen 0 128x128x24" -- python main.py $OPTARG
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