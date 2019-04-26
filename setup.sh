#!/usr/bin/env bash

set -Eeuxo pipefail

pip install -r requirements.txt

proto_dir="./onnx_transformer/proto"
protoc -I ${proto_dir} --python_out ${proto_dir} ${proto_dir}/*.proto
