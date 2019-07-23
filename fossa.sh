#!/bin/sh

# Change the execution location
mkdir fossa-cli
export BINDIR=./fossa-cli

# Install latest FOSSA cli
curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/fossas/fossa-cli/master/install.sh | bash

# Analyze the project and submit to FOSSA
FOSSA_API_KEY=$1 ./fossa-cli/fossa analyze
