#!/bin/bash

#Convert environment.yml to requirements.dev.txt
sed -n '/python/,$p' environment.yml | sed '1d' | sed 's/  - //g' > requirements.dev.txt
python -m pip install -r requirements.dev.txt
rm -rf requirements.dev.txt
