#!/bin/bash

for i in {12..30..2}; do
    python QAOA_square.py results_dac.csv $((i*2)) $i
done