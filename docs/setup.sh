#!/bin/bash

#rm -rf _build
python -m sphinx -T -b html -d _build/doctrees -D language=en . _build/html
python -m http.server
