#!/bin/bash

# Assign the variable name
name="GBM"
type="clinical"

# Use the train_eval.py script to train the model and cross-validate on NSCLC-Radiomics dataset
python train_eval.py \
    -n "$name" \
    --save_dir "../tmp/${name}_${type}_model/" \
    --feature_type "${type}" \
    --features_norm \
    --feature_selection
