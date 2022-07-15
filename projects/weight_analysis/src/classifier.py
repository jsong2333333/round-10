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

def bootstrap_performance(X, y, n=50, test_size=.2, eps=.01):
    all_cross_entropy, all_accuracy = [], []
    for i in tqdm(range(n)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        
        base_clf = RandomForestClassifier(n_estimators=1000, max_depth=2, criterion='log_loss',  random_state=i)
        clf = CalibratedClassifierCV(base_estimator=base_clf, cv=5)
        
        clf.fit(X_train, y_train)
        
        all_cross_entropy.append(log_loss(y_test, clf.predict_proba(X_test), eps=eps))
        all_accuracy.append(clf.score(X_test, y_test))
    return {'cross_entropy': all_cross_entropy, 
            'accuracy': all_accuracy}

X = np.load('features.npy')
y = np.load('labels.npy')

# performance = bootstrap_performance(X, y)
# all_cross_entropy, all_accuracy = performance['cross_entropy'], performance['accuracy']

# TODO: write output somewhere and save the classifier

def weight_analysis_detector(model_filepath):
    base_clf = RandomForestClassifier(n_estimators=1000, max_depth=2, criterion='log_loss',  random_state=0)
    clf = CalibratedClassifierCV(base_estimator=base_clf, cv=5)
    clf.fit(X, y)

    predict_model_features = np.asarray([fe.get_model_features(model_filepath)])
    predict_probability = clf.predict_proba(predict_model_features)
    logging.info('Trojan Probability of this model is: {}'.format(predict_probability[0, -1]))


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

