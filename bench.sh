#!/bin/zsh
# Release build + fixed-length benchmark run. Results are saved in inference/.
# Usage: ./bench.sh [frames] [csv name]   (defaults: 600 frames, bench_results.csv)
set -e

FRAMES=${1:-600}
CSV=${2:-bench_results.csv}
ROOT=${0:A:h}

mkdir -p "$ROOT/inference" "$ROOT/build-release"
cd "$ROOT/build-release"
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
./engine --bench "$FRAMES" --csv "$ROOT/inference/${CSV:t}"
