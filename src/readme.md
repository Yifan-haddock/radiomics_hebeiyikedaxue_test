## use train_eval.py to train model and cross-validate on NSCLC-Radiomics dataset
python train_eval.py
    -n RSF
    --save_dir './tmp/rsf_model/'
    --features_norm
    --use_clincal_feature
    --feature_selection