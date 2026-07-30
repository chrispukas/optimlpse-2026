#!/bin/bash

conda init

conda activate botorch
uv pip install -e .
