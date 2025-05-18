#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

launchctl stop com.apple.audio.coreaudiod && sudo launchctl start com.apple.audio.coreaudiod

killall coreaudiod

echo "Sound reset completed successfully."
