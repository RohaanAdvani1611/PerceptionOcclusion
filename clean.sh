#!/bin/zsh
set -e

rm -rf build
mkdir build
cd build
cmake .. -G Ninja
ninja
./engine
cd ..
