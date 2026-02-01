#!/bin/bash
set -e

ORT_VER="1.16.3"
ORT_DIR="onnxruntime-linux-x64-${ORT_VER}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_DIR}.tgz"

if [ -d "$ORT_DIR" ]; then
    echo "ONNX Runtime $ORT_VER already present in $ORT_DIR"
    exit 0
fi

echo "Downloading ONNX Runtime $ORT_VER..."
wget -q --show-progress "$ORT_URL"

echo "Extracting..."
tar -xzf "${ORT_DIR}.tgz"
rm "${ORT_DIR}.tgz"

echo "Done."
echo "ONNX Runtime is in $ORT_DIR"
