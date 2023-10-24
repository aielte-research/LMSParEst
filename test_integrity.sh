#!/bin/bash
for filename in ./configs/Integrity_tests/*.yaml; do
    python run.py -c $filename
done