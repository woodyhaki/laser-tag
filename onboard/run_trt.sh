#!/bin/bash

# Check if an ONNX file is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <onnx_filename>"
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

# ONNX and ENGINE directories
ONNX_DIR="${SCRIPT_DIR}/onnx"
ENGINE_DIR="${SCRIPT_DIR}/engine"

# Make sure the engine directory exists
mkdir -p "${ENGINE_DIR}"

# Path to the input ONNX file
ONNX_FILE="${ONNX_DIR}/$1"

# Check if the ONNX file exists
if [ ! -f "${ONNX_FILE}" ]; then
    echo "ONNX file not found: ${ONNX_FILE}"
    exit 1
fi

# Dynamically generate the engine file name (placed under engine directory)
ENGINE_FILE="${ENGINE_DIR}/$(basename "${ONNX_FILE%.onnx}.engine")"

# Run trtexec command
/usr/src/tensorrt/bin/trtexec --onnx="${ONNX_FILE}" --fp16 --saveEngine="${ENGINE_FILE}"

echo "Engine file saved as: ${ENGINE_FILE}"