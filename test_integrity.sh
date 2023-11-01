#!/bin/bash
for filename in ./configs/Integrity_tests/*.yaml; do
    python main_trainer.py -c $filename
done