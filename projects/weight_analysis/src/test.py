import torch
import numpy as np
import itertools


import pandas as pd
df=pd.read_csv('/scratch/jialin/round-10/projects/weight_analysis/src/METADATA.csv', sep=',',header=None)


store_eigs_modelA = []
store_eigs_modelB = []

targets_modelA = []
targets_modelB = []


def get_eigs(model_sub, idx_low=0, idx_high=3):
    output = []

    for param in model_sub.parameters():
        output.append(param.data.cpu().numpy())
    k = 0
    eigs = []
    for o in output:
        if len(o.shape)>2:
            if k >= idx_low and k <= idx_high:  
                x = o.reshape(o.shape[1], -1)
                _, s, _ = np.linalg.svd(x, False)
                eigs.append(s[0::]**2)
            k += 1
    return np.asarray(list(itertools.chain(*eigs))).flatten()



nmodels = 143
for i in range(nmodels):
    model_id = df[0][i]
    path = '/scratch/data/TrojAI/round10-train-dataset/' + str(model_id) + '/model.pt'
    m = torch.load(path)
    
    pars = sum(p.numel() for p in m.parameters())/1000.0
    if pars == 41755.2860:
        model_sub = m.backbone # select a block
        
        targets_modelA.append(df[1][i]) # get target
        store_eigs_modelA.append(get_eigs(model_sub, idx_low=1, idx_high=3)) # get features

    elif pars == 35641.8260:

        model_sub = m.backbone

        targets_modelB.append(df[1][i]) # get target
        store_eigs_modelB.append(get_eigs(model_sub, idx_low=0, idx_high=4)) # get features



### Build Classifier 
### ---------------------------------------------------------------------------

def classify(features, targets):
    from sklearn.metrics import log_loss
    from sklearn.ensemble import RandomForestClassifier

    X = features
    y = targets
    split = int(X.shape[0]*0.6)
    clf = RandomForestClassifier(n_estimators=2000, max_depth=2, criterion='log_loss', bootstrap=True, random_state=0)
    clf.fit(X[0:split,:],y[0:split])

    #print(log_loss(y, clf.predict_proba(X)[:,1]))
    print('Cross-Entropy train: ', log_loss(y[0:split], clf.predict_proba(X[0:split,:])[:,1]))
    print('Cross-Entropy test: ', log_loss(y[split::], clf.predict_proba(X[split::,:])[:,1]))
    
    ## save model here
    #return None

print('***** ******')
print('***** ******')

print('***** Model A ******')
targets_modelA = np.asarray(targets_modelA)
store_eigs_modelA = np.asarray(store_eigs_modelA)
#print(store_eigs_modelA.shape)
classify(store_eigs_modelA, targets_modelA)

print('***** Model B ******')
targets_modelB = np.asarray(targets_modelB)
store_eigs_modelB = np.asarray(store_eigs_modelB)
#print(store_eigs_modelB.shape)
classify(store_eigs_modelB, targets_modelB)
