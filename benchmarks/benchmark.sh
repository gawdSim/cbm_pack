#!/usr/bin/bash

# assume this script is run from within the benchmarks folder
cd ..

python_ver=
if command -v python &> /dev/null; then
    python_ver=python
elif command -v python3 &> /dev/null; then
    python_ver=python3
else
    echo "python executable not found. Exiting..."
    break
fi

case "$1" in
    "analysis")
        $python_ver -m benchmarks.analysis_bench
        ;;
    "transform")
        $python_ver -m benchmarks.transform_bench
        ;;
    "")
        echo "command not recognized: '${1}'. Exiting..."
        ;;
esac

cd benchmarks