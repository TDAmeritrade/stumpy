#!/bin/sh

# Change the execution location
mkdir fossa-cli
export BINDIR=./fossa-cli

# Install latest FOSSA cli
curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/fossas/fossa-cli/master/install.sh | bash

# Export the FOSSA API key
export FOSSA_API_KEY=$1
echo 'FOSSA_API_KEY=$FOSSA_API_KEY'

# Generate the FOSSA yaml file to direct dependency discovery
./fossa-cli/fossa init
cat .fossa.yml

# Analyze the project and submit to FOSSA
./fossa-cli/fossa analyze
