#!/usr/bin/env bash
#
# Permission is  hereby  granted,  free  of  charge,  to  any  person
# obtaining a copy of  this  software  and  associated  documentation
# files  (the  "Software"),  to  deal   in   the   Software   without
# restriction, including without limitation the rights to use,  copy,
# modify, merge, publish, distribute, sublicense, and/or sell  copies
# of the Software, and to permit persons  to  whom  the  Software  is
# furnished to do so.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT  WARRANTY  OF  ANY  KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES  OF
# MERCHANTABILITY,   FITNESS   FOR   A   PARTICULAR    PURPOSE    AND
# NONINFRINGEMENT.  IN  NO  EVENT  SHALL  THE  AUTHORS  OR  COPYRIGHT
# OWNER(S) BE LIABLE FOR  ANY  CLAIM,  DAMAGES  OR  OTHER  LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM,
# OUT OF OR IN CONNECTION WITH THE  SOFTWARE  OR  THE  USE  OR  OTHER
# DEALINGS IN THE SOFTWARE.
#
##########################License above is Equivalent to Public Domain
ISO_8601=`date -u "+%FT%TZ"` #ISO 8601 Script Start UTC Time
utc=`date -u "+%Y.%m.%dT%H.%M.%SZ"` #UTC Time (filename safe)
owd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" #Path to THIS script.

# Define a function to run a script and log its output
# Usage: run_and_log <script> <logfile> [devshell]
run_and_log() { local script="$1" local logfile="$2" local devshell="$3"
    echo "Running $script at $utc"

    if [[ -n "$devshell" ]]; then
        # Run inside nix-shell if devshell path is provided
        nix-shell "$devshell" --run "bash $script" 2>&1 | tee "$logfile"
    else
        # Otherwise run directly with bash
        bash "$script" 2>&1 | tee "$logfile"
    fi

    echo "Completed $script at $utc"
}

combine_files() {
  # First argument is the output file
  output_file=$1
  shift   # Remove the first argument so only input files remain
  rm $output_file
  # Loop through all remaining arguments (the input files)
  for file in "$@"; do
    filename=$(basename "$file")
    # Print header with 10 # chars before and after
    {
      echo -e "##########\nFile: $filename\n##########\n\n\`\`\`"
      # Escape triple backticks in file content
      sed 's/```/\\```/g' "$file"
      echo -e "\n\`\`\`"
    } >> "$output_file"
  done
}

######################################################################

# Return to main dir
cd "$owd"

# Example call: run test.sh inside nix-shell, log to test_output.log
run_and_log "./test.sh" "$owd/test_output.log" "$owd/devShells.nix"

combine_files concat.md ./concat_macro.sh ./devShells.nix ./requirements.md ./requirements.txt ./test_output.log

exit

######################################################################

# checkout with
cd ~/
git clone https://github.com/GlassGhost/Papers-in-100-Lines-of-Code

##########

# Test if your patch worked
cd ~/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning
./concat_macro.sh

##########

# Version Control
cd ~/Papers-in-100-Lines-of-Code/Playing_Atari_with_Deep_Reinforcement_Learning
git gui &
