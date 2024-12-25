# QuickGuides
Download and prepare CT scan in ../data/
## feature extraction
python feature_extraction.py
## Use train_eval.py to train and cross-validate on NSCLC Radiomics dataset
python train_eval.py
    -n RSF
    --save_dir './tmp/rsf_model/'
    --features_norm
    --use_clincal_feature
    --feature_selection
