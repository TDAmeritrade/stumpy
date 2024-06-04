#!/bin/bash

make html
#sphinx-build -nW --keep-going -b html . ./_build/html
python -m http.server
