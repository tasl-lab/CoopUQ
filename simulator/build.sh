#!/usr/bin/env bash
# Build a simulator executable from a scene description.
#   ./build.sh <config.json> [name] [options]
#   ./build.sh configs/id_1.json                         # -> Builds/id_1/id_1.x86_64
#   ./build.sh configs/id_1.json id_1_n5 -robots 5 -pedestrians 12
# Options (they override the config file):
#   -robots N  -pedestrians N  -pedestrianSpeed MIN,MAX  -goalDistance MIN,MAX  -maxSteps N
#   -fixedSeed  -continueAfterGoal
# UNITY selects the editor binary. The log is written to Builds/<name>/build.log.
set -euo pipefail
UNITY=${UNITY:-$HOME/Unity/Hub/Editor/2023.2.20f1/Editor/Unity}
PROJ=$(cd "$(dirname "$0")" && pwd)
config=$(realpath "${1:?usage: build.sh <config.json> [name] [options]}"); shift
name=$(basename "$config" .json)
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then name=$1; shift; fi
out="$PROJ/Builds/$name"
mkdir -p "$out"
"$UNITY" -batchmode -nographics -quit -projectPath "$PROJ" -executeMethod BuildCLI.Build \
    -logFile "$out/build.log" -config "$config" -out "$out/$name.x86_64" "$@" \
    || { echo "BUILD FAILED, see $out/build.log"; grep -E "\[BuildCLI\]|error CS" "$out/build.log" | tail -20; exit 1; }
grep "\[BuildCLI\]" "$out/build.log"
