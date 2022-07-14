import torch
import os
import numpy as np
import pandas as pd
import itertools

DATA_PATH = '/scratch/data/TrojAI/round10-train-dataset/'

def get_all_features():
    all_features = []
    num_models = 144
    for model_num in range(num_models):
        model_filepath = _get_model_filepath(model_num)
        all_features.append(get_model_features(model_filepath))

    X = np.asarray(all_features)
    return X

def get_all_labels():
    metadata_filepath = os.path.join(DATA_PATH, "METADATA.csv")
    metadata = pd.read_csv(metadata_filepath)
    y = metadata['poisoned'].to_numpy()
    return y


def get_model_features(model_filepath):

    model = torch.load(model_filepath)
    model_backbone = model.backbone

    all_backbone_params = []
    for param in model_backbone.parameters():
        all_backbone_params.append(param.data.cpu().numpy())


    features = []
    num_layers = 0
    for backbone_params in all_backbone_params:
        if len(backbone_params.shape) > 2:
            reshaped_params = backbone_params.reshape(backbone_params.shape[1], -1)
            _, singular_values, _ = np.linalg.svd(reshaped_params,False)
            squared_singular_values = singular_values**2
            top_five_sq_sv = squared_singular_values[:5]
            features += top_five_sq_sv.tolist()
            num_layers += 1
        if num_layers == 2:
            break

    return features

def _get_model_filepath(model_num: int) -> os.path:
    num_as_str = str(100000000 + model_num)[1:]
    model_id = f"id-{num_as_str}"
    return os.path.join(DATA_PATH, model_id, 'model.pt')

if __name__ == "__main__":
    X = get_all_features()
    y = get_all_labels()

    np.save('features.npy', X)
    np.save('labels.npy', y)