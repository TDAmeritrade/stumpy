#!/bin/sh

cat PID | xargs kill -9 2>/dev/null
nohup ./get_timing.py &> LOG &
echo $! > PID
