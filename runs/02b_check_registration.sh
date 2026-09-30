#!/usr/bin/env bash
python src/check_registration.py results/renders.json results/registration.json
python src/box_size_check.py results/renders.json results/alignment_box_sizes.json
