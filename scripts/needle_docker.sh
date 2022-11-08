#!/bin/bash

docker run -d -v $HOME/needle:/workspace --gpus all --rm --pid=host --net=host --name=needle --interactive --tty nvcr.io/nvidia/pytorch:22.08-py3
docker exec needle bash -c "pip install --upgrade --no-deps git+https://github.com/dlsys10714/mugrade.git"
docker exec needle bash -c "pip install numdifftools"
docker exec -it needle bash
