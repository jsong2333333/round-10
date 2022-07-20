import torch
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

DATA_PATH = '/scratch/data/TrojAI/round10-train-dataset/'

def get_all_features():
    all_features = []
    num_models = 144
    for model_num in range(num_models):
        model_filepath = _get_model_filepath(model_num)
        all_features.append(get_model_features(model_filepath)['features'])

    X = np.asarray(all_features)
    return X

def get_all_labels():
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    y = metadata['poisoned'].to_numpy()
    return y


def get_features_and_labels_by_model_class():
    model_A_features, model_B_features, model_A_labels, model_B_labels = [], [], [], []
    num_models = 144
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    labels = metadata['poisoned'].to_numpy()
    for model_num in range(num_models):
        model_filepath = _get_model_filepath(model_num)
        class_and_features_by_model = get_model_features(model_filepath)
        model_class, model_features = class_and_features_by_model['model_class'], class_and_features_by_model['features']
        if model_class == 'A':
            model_A_features.append(model_features)
            model_A_labels.append(labels[model_num])
        elif model_class == 'B':
            model_B_features.append(model_features)
            model_B_labels.append(labels[model_num])
    return {'model_A_features': np.asarray(model_A_features), 'model_B_features': np.asarray(model_B_features), 
            'model_A_labels': np.asarray(model_A_labels), 'model_B_labels': np.asarray(model_B_labels)}


def get_model_features(model_filepath):
    model = torch.load(model_filepath)
    model_backbone = model.backbone

    num_of_params = sum(p.numel() for p in model.parameters())/1000.0

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())

    model_class, features = '', []
    if num_of_params == 41755.2860:  # model A
        model_class = 'A'
        features = _get_eigen_vals(all_backbone_params, 1, 3)
    elif num_of_params == 35641.8260:  # model B
        model_class = 'B'
        features = _get_eigen_vals(all_backbone_params, 0, 4)

    return {'model_class': model_class, 'features': features}


def _get_eigen_vals(all_backbone_params, idx_low=0, idx_high=3):
    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) > 2:
            if num_layers >= idx_low and num_layers <= idx_high:
                reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
                _, singular_values, _ = np.linalg.svd(reshaped_params,False)
                squared_singular_values = singular_values**2
                # top_five_sq_sv = squared_singular_values[:5]
                features += squared_singular_values.tolist()
                num_layers += 1
            num_layers += 1
    return features


def _get_model_filepath(model_num: int) -> os.path:
    num_as_str = str(100000000 + model_num)[1:]
    model_id = f"id-{num_as_str}"
    return os.path.join(DATA_PATH, model_id, 'model.pt')

if __name__ == "__main__":
    all_features_and_labels = get_features_and_labels_by_model_class()
    model_A_features, model_B_features, model_A_labels, model_B_labels = all_features_and_labels['model_A_features'], all_features_and_labels['model_B_features'], all_features_and_labels['model_A_labels'], all_features_and_labels['model_B_labels']
    # np.save('features_A', model_A_features)
    # np.save('features_B', model_B_features)
    # np.save('labels_A', model_A_labels)
    # np.save('labels_B', model_B_labels)
    base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss',  bootstrap=True, random_state=0)
    clf_A, clf_B = CalibratedClassifierCV(base_estimator=base_clf, cv=5), CalibratedClassifierCV(base_estimator=base_clf, cv=5)
    clf_A.fit(model_A_features, model_A_labels)
    clf_B.fit(model_B_features, model_B_labels)
    dump(clf_A, 'classifier_model_A.joblib')
    dump(clf_B, 'classifier_model_B.joblib')