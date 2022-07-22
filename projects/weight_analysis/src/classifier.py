from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import logging
import torch
import feature_extractor as fe
from joblib import load


def bootstrap_performance(X, y, n=10, test_size=.2, eps=.01):
    all_cross_entropy, all_accuracy = [], []
    for i in tqdm(range(n)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        base_clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss',  bootstrap=True, random_state=i)
        clf = CalibratedClassifierCV(base_estimator=base_clf, cv=5)
        
        clf.fit(X_train, y_train)
        
        all_cross_entropy.append(log_loss(y_test, clf.predict_proba(X_test), eps=eps))
        all_accuracy.append(clf.score(X_test, y_test))
    return {'cross_entropy': all_cross_entropy, 
            'accuracy': all_accuracy}

# Only used for checking the performance
# X_A = np.load('features_A.npy')
# X_B = np.load('features_B.npy')
# y_A = np.load('labels_A.npy')
# y_B = np.load('labels_B.npy')

# pA = bootstrap_performance(X_A, y_A, n=50)
# pB = bootstrap_performance(X_B, y_B, n=50)
# ce_A, acc_A = pA['cross_entropy'], pA['accuracy']
# ce_B, acc_B = pB['cross_entropy'], pB['accuracy']

# print('mean ce_A: ', sum(ce_A)/len(ce_A), 'mean ce_B: ', sum(ce_B)/len(ce_B))
# print('mean acc_A: ', sum(acc_A)/len(acc_A), 'mean acc_B: ', sum(acc_B)/len(acc_B))
# print(ce_A, acc_A, ce_B, acc_B)


def weight_analysis_detector(model_filepath):
    predict_model_class_and_features = fe.get_model_features(model_filepath)
    predict_model_class = predict_model_class_and_features['model_class']
    predict_model_features = np.asarray([predict_model_class_and_features['features']])
    if predict_model_class == 'A':
        clf = load('/classifier_model_A.joblib')
    elif predict_model_class == 'B':
        clf = load('/classifier_model_B.joblib')
    else:
        logging.warning('No able to detect such model class')
        return
    predict_probability = clf.predict_proba(predict_model_features)
    logging.info('Trojan Probability of this class {} model is: {}'.format(predict_model_class, predict_probability[0, -1]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Weight Analysis Classifier')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("weight analysis classifier launched")

    if args.model_filepath is not None:
        logging.info('Calling the weight analysis classifier')
        weight_analysis_detector(args.model_filepath)

