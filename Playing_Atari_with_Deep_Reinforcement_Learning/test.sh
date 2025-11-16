#!/usr/bin/env bash

set -e
set -x

utc=`date -u "+%Y.%m.%dT%H.%M.%SZ"` #UTC Script Start Time (filename safe)
owd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" #Path to THIS script.
#_______________________________________________________________________________
cd "$owd"

python ./dqn.py

exit
#_______________________________________________________________________________
