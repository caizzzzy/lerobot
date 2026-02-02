#!/bin/bash

set -e

echo "Step 1: Install Python package to current environment..."
python -m pip install .
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
TARGET_DIR=$SITE_PACKAGES/diana_robot

echo "Step 2: Check and set LD_LIBRARY_PATH in ~/.bashrc ..."
LD_PATH_LINE="export LD_LIBRARY_PATH=$TARGET_DIR:\$LD_LIBRARY_PATH"
if ! grep -Fxq "$LD_PATH_LINE" ~/.bashrc; then
    echo "$LD_PATH_LINE" >> ~/.bashrc
    echo "LD_LIBRARY_PATH added to ~/.bashrc"
else
    echo "LD_LIBRARY_PATH already exists in ~/.bashrc"
fi

echo "Step 3: Make LD_LIBRARY_PATH effective for current shell..."
export LD_LIBRARY_PATH=$TARGET_DIR:$LD_LIBRARY_PATH

echo "Installation finished! Please restart your terminal or run 'source ~/.bashrc' to make LD_LIBRARY_PATH effective."