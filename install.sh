#!/bin/bash

set -e

if [ "$#" -lt 2 ]; then
  echo "install.sh - Install ShakeTune files "
  echo "Usage: install.sh [KLIPPER_DIR_PATH] [KLIPPER_CONFIG_DIR_PATH]"
  echo "Example: install.sh /usr/lib/klipper /etc/klipper"
  echo ""
  exit 1
fi

KLIPPER_DIR_PATH="$1"
KLIPPER_CONFIG_DIR_PATH="$2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

patch -p1 -d /usr/lib/klipper < scripts/adxl345-sync-write.patch
echo "Patched Klipper ADXL345 module"

cp -r shaketune                 "$KLIPPER_DIR_PATH/klippy/extras"
cp scripts/shaketune_process.py "$KLIPPER_DIR_PATH/scripts"
echo "ShakeTune module installed to $KLIPPER_DIR_PATH"

mkdir -p                        "$KLIPPER_CONFIG_DIR_PATH/shaketune"
cp macros/*.cfg                 "$KLIPPER_CONFIG_DIR_PATH/shaketune"
echo "ShakeTune macros installed to $KLIPPER_CONFIG_DIR_PATH/shaketune"
