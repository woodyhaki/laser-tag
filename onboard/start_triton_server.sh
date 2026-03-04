#!/bin/bash

cd /home/$(whoami)

./triton-server/bin/tritonserver \
  --model-repository=triton-deploy/models \
  --backend-directory=/home/$(whoami)/triton-server/backends \
  --backend-config=tensorrt,version=8
