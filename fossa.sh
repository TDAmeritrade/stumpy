#!/bin/sh

# Install latest FOSSA cli
curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/fossas/fossa-cli/master/install.sh | bash

# Change the execution location
mkdir fossa-cli
export BINDIR=./fossa-cli

# API Key
export FOSSA_API_KEY=59a5cb358c757b5be7c2ed677b2f852b

# Analyze the project
./fossa-cli/fossa -analyze
