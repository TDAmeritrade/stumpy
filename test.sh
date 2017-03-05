#!/bin/sh

python3 `which nosetests` -v --nocapture

#python3 `which nosetests` -v --nocapture tests/test_core.py:TestCore.test_calculate_distance_profile
