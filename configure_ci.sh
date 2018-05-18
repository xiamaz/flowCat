#!/bin/sh
CIURL="git@ec2-35-158-105-254.eu-central-1.compute.amazonaws.com"
# fail script immediately
set -e
ssh -T $CIURL 'echo "Login possible.";exit'

# set up two remote urls
git remote set-url --add --push origin $CIURL:flowCat.git
git remote set-url --add --push origin $(git remote get-url origin)
