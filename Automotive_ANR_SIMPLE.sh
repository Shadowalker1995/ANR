#!/usr/bin/env bash
# Example script for ANRS (i.e. the 'Simplified Model' used for obtaining the pretrained weights of the ARL layer)
# Model Pretraining for ANR, i.e. the ARL layer
python PyTorchTEST.py -d "Automotive" -m "ANRS" -e 10 -p 1 -v 19438 -rs 1337 -gpu 0 -vb 1 -sm "Automotive_ANRS"
