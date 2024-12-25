from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import glob
import re
import joblib
from tqdm import tqdm
import shap
import os
import argparse


## choose whether to use z-score normalization
def zscore_2dnorm(feature_matrix):
    mean = np.mean(feature_matrix, axis=0, keepdims=True)
    std = np.std(feature_matrix, axis=0, keepdims=True)
    feature_matrix = (feature_matrix - mean) / std
    return feature_matrix

## feature selection
def select_features(X, Y, X_label=None):
    lasso_cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=np.logspace(-2, 0, 20))
    lasso_cox.fit(X, Y)
    print(f"{(lasso_cox.coef_ != 0).all(axis=1).sum()} features are selected.")
    if X_label is not None:
        return X_label[(lasso_cox.coef_ != 0).all(axis=1)]
    else:
        return (lasso_cox.coef_ != 0).all(axis=1)
## feature interpretation
def shap_explainer(model, X, feature_names = None):
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    return shap_values

## KM plot and test
def km_plot(y_test, risk_scores, save_dir=None):
    results = pd.DataFrame({
        "risk_score": risk_scores,
        "time": y_test["time"],  
        "event": y_test["event"],
    })
    threshold = results["risk_score"].median()
    results["risk_group"] = np.where(results["risk_score"] > threshold, "High Risk", "Low Risk")
    # Log-rank test
    high_risk = results[results["risk_group"] == "High Risk"]
    low_risk = results[results["risk_group"] == "Low Risk"]
    logrank_result = logrank_test(
        high_risk["time"], low_risk["time"], 
        event_observed_A=high_risk["event"], event_observed_B=low_risk["event"]
    )
    p_value = logrank_result.p_value
    # print(f"Log-rank test p-value: {logrank_result.p_value}")
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8,8))
    for group in sorted(results["risk_group"].unique()):
        group_data = results[results["risk_group"] == group]
        kmf.fit(group_data["time"], group_data["event"], label=group)
        # kmf.plot_survival_function()
        kmf.plot_survival_function(show_censors=True, ci_show=False) 
        plt.legend(title="Risk Group")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=labels,
        title=f"Risk Group",
        loc="best",
        frameon=True,
    )
    plt.text(
        0.7, 0.65, f"Log-rank p-value: {p_value:.3e}",
        fontsize=12, transform=plt.gca().transAxes,
        ha="center", bbox=dict(facecolor="white", alpha=0.8, edgecolor="white")
    )

    plt.title("Kaplan-Meier Survival Curve by Predicted Risk Group")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    if os.path.exists(save_dir):
        image_path = os.path.join(save_dir,'KaplanMeier_curve.png')
    else:
        raise FileNotFoundError(f"The specified directory save_dir does not exist.")
    plt.savefig(image_path, dpi = 300)

if __name__ == '__main__':
    ## global_vars
    clinical_info_path = '../data/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'
    feature_name_path = '../data/manifest-1603198545583/NSCLC-Radiomics/feature_names.txt'
    feature_path = glob.glob('../data/manifest-1603198545583/NSCLC-Radiomics/*/*/CT_features*')
    ## parse args
    parser = argparse.ArgumentParser(description="model_hyperparams")
    parser.add_argument('-n', '--model_name', type=str, required=True, help="RSF|GBM|COX")
    parser.add_argument('--save_dir', type=str, help="save directory")
    parser.add_argument('--feature_type', type=str, required= True, help="choose clinical|radiomics|mixed")
    parser.add_argument('--features_norm', action='store_true', help="choose whether to use z-score normalization")
    parser.add_argument('--feature_selection', action='store_true', help="choose whether to use lasso to filter features")
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    ## load data and features
    df = pd.read_csv(clinical_info_path)
    ### features
    with open(feature_name_path, 'r') as f:
        feature_name = [name.strip('\n') for name in f.readlines()]
    
    patient_id = []
    feature_matrix = []
    for path in feature_path:
        with open(path, 'rb') as f:
            feature_dict = joblib.load(f)
        feature_matrix.append(
            np.array([feature_dict[name] for name in feature_name])      
        )
        patient_id.append(
            path.split('/')[4]
        )
    feature_matrix = np.stack(feature_matrix)
    if args.features_norm:
        feature_matrix = zscore_2dnorm(feature_matrix)
    ## concatenate features and clincal info
    filtered_df = df[df['PatientID'].isin(patient_id)]
    sorted_df = filtered_df.set_index('PatientID').loc[patient_id].reset_index()
    feature_df = pd.DataFrame(feature_matrix, columns=feature_name)
    dataset = pd.concat([sorted_df, feature_df], axis=1)
    ## feature labels
    if args.feature_type == 'mixed':
        X_label = ['age', 'gender',"clinical.T.Stage","Clinical.N.Stage","Clinical.M.Stage","Overall.Stage"] + feature_name
    elif args.feature_type == 'radiomics':
        X_label = feature_name
    elif args.feature_type == 'clinical':
        X_label = ['age', 'gender',"clinical.T.Stage","Clinical.N.Stage","Clinical.M.Stage","Overall.Stage"]
    else:
        raise ValueError("Feature type not found")
    y_label = ['deadstatus.event','Survival.time']
    ## handle nan    
    for label in dataset.columns:
        nan_num = dataset[label].isnull().sum()
        if nan_num > 0:
            print(label, nan_num)        
    ### only a few nan, rem.
    dataset.dropna(subset = [column for column in X_label], inplace= True)
    ## handle ordinal and categorical values
    if args.feature_type != 'radiomics':
        dataset['clinical.T.Stage'] = dataset['clinical.T.Stage'].astype('int64')
        ordinal_mapping = {'Overall.Stage': {'I': 1, 'II': 2, 'IIIa': 3, 'IIIb': 4}}
        for column, mapping in ordinal_mapping.items():
                dataset[column] = dataset[column].map(mapping)
        categorical_columns = ['gender']  
        dataset = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
    if args.feature_type == 'mixed':
        X_label = np.array(['age', 'gender_male',"clinical.T.Stage","Clinical.N.Stage","Clinical.M.Stage","Overall.Stage"] + feature_name)
    elif args.feature_type == 'clinical':
        X_label = np.array(['age', 'gender_male',"clinical.T.Stage","Clinical.N.Stage","Clinical.M.Stage","Overall.Stage"])
    else:
        X_label = np.array(feature_name)
    X = dataset[X_label].to_numpy()
    y = dataset[y_label].to_numpy()
    structured_y = np.zeros(y.shape[0], dtype = [('event', bool), ('time', float)])
    structured_y['event'] = y[:, 0].astype(bool)
    structured_y['time'] = y[:, 1] 
    y = structured_y
    ## feature selection (optional)
    if args.feature_selection and args.feature_type != 'clinical':
        X_label = select_features(X, y, X_label = X_label)
        X = dataset[X_label].to_numpy()
        X = zscore_2dnorm(X)
    X_label_path = os.path.join(args.save_dir, 'X_label.txt')
    with open(X_label_path, 'w') as f:
        f.writelines([i+'\n' for i in X_label])
    
    ## 5 fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=23)
    c_index_scores = []
    best_c_index = -np.inf
    best_model = None
    best_index = None
    ### initialize survival model
    if args.model_name == "RSF":
        model = RandomSurvivalForest(n_estimators=100, min_samples_split=5, random_state=32)
    elif args.model_name == 'GBM':
        model = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=32)
    else:
        model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=np.logspace(-2, 0, 20), max_iter=10000,)
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        risk_scores = model.predict(X_test)    
        ### Calculate the C-index
        c_index = concordance_index_censored(
            y_test["event"], y_test["time"], risk_scores
        )[0]
        c_index_scores.append(c_index)
        print(f"C-index for fold: {c_index:.3f}")
        if c_index > best_c_index:
            best_c_index = c_index
            best_model = model
            y_test_km = y_test
            risk_scores_km = risk_scores
    ## Final average C-index across all folds
    average_c_index = np.mean(c_index_scores)
    print(f"\nAverage C-index across 5 folds: {average_c_index:.3f}")
    with open(f'{args.save_dir}/c-index.txt','w') as f:
        f.writelines([f'{i:.3f}\n' for i in c_index_scores])
        f.write(f"Average C-index across 5 folds: {average_c_index:.3f}\n")
    ## SHAP analysis using the best model
    print("\nRunning SHAP analysis on the best model...")
    explainer = shap.Explainer(best_model.predict, X)  # Use SHAP's TreeExplainer
    shap_values = explainer(X)
    shap_value_path = os.path.join(args.save_dir, 'shap_values.joblib')
    with open(shap_value_path,'wb') as f:
        joblib.dump(shap_values, f)
    plt.figure()
    shap.summary_plot(shap_values, feature_names = X_label, plot_size=(12, 8), show=False)
    shap_image_path = os.path.join(args.save_dir, 'shap.png')
    plt.savefig(shap_image_path, dpi =300)
    plt.close()
    ## plotting KM
    km_plot(y_test_km, risk_scores_km, save_dir=args.save_dir)
    ## save model
    model_path = os.path.join(args.save_dir, f'{args.model_name}_model.pkl')
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)
    