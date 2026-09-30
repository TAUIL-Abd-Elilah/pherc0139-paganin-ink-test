#!/usr/bin/env bash
# Offset of pag0 / pag50 from the standard volume at twelve points spread over the scroll (two radii x six heights).
python src/align_check.py results/alignment.json \
  6000 3300 2600  12000 2500 3300  2500 3088 3680  2500 3136 1184  7000 3648 3664  7000 4896 3296 \
  11500 2080 3376  11500 1216 2752  16000 3760 3520  16000 4816 3472  19500 2784 2736  19500 2832 3616
