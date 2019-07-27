#!/bin/bash
while getopts "ce" opt; do
  case $opt in
    e)
      xvfb-run -a -s "-screen 0 128x128x24" -- python main.py $OPTARG
      ;;
    c)
      mv output/config/$OPTARG consts.py
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