#!/usr/bin/env bash
# needs a GPU; three seeds, about 45 min on an RTX 3090
python src/fair_cnn.py results/renders.json results/fair_cnn.json --steps 1500 --seeds 0 1 2
