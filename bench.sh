#!/bin/zsh
# Release build + fixed-length benchmark run.
# Usage: ./bench.sh [frames] [csv]   (defaults: 600 frames, bench_results.csv)
set -e

FRAMES=${1:-600}
CSV=${2:-bench_results.csv}

mkdir -p build-release
cd build-release
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
./engine --bench "$FRAMES" --csv "../$CSV"
cd ..
