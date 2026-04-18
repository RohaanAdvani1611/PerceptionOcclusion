#!/bin/zsh
set -e

mkdir -p build
cd build
cmake .. -G Ninja
ninja
./engine
cd ..
