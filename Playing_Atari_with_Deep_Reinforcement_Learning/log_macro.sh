#!/usr/bin/env bash

set -e
set -x

utc=`date -u "+%Y.%m.%dT%H.%M.%SZ"` #UTC Script Start Time (filename safe)
owd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" #Path to THIS script.
#_______________________________________________________________________________
cd "$owd"

# logfile="$owd/test_output_$utc.log"
logfile="$owd/test_output.log"
exec > >(tee "$logfile") 2>&1
echo "Running test at $utc"

cd "$owd"
nix-shell ./devShells.nix --run "bash ./test.sh"

echo "Test completed."
exit
#_______________________________________________________________________________
