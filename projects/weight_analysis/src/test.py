import torch
import numpy as np
import itertools




store = []
store_eigs = []

nmodels=140
for i in range(nmodels):
    if i<10:
        path = '/scratch/data/TrojAI/round10-train-dataset/id-0000000'+str(i)+'/model.pt'
    elif i<100:
        path = '/scratch/data/TrojAI/round10-train-dataset/id-000000'+str(i)+'/model.pt' 
    elif i>100:
        path = '/scratch/data/TrojAI/round10-train-dataset/id-00000'+str(i)+'/model.pt' 

    m = torch.load(path)
    print(path)
    #print(m)
    idx=1
    #print(m.head.regression_head)
    
    #model_sub = m
    #model_sub = m.backbone
    #model_sub = m.backbone.extra
    #model_sub = m.head.classification_head
    #model_sub = m.head.regression_head
    model_sub = m.backbone
    #print(model_sub)

    output = []

    for param in model_sub.parameters():
        output.append(param.data.cpu().numpy())
            
    print(output[idx].shape)            

    def eigs(x):
        _,s,_=np.linalg.svd(x, False)
        return s

    #s = [eigs(output[idx][i,:,:,:].reshape(512,9)) for i in range(16)]
    #s = list(itertools.chain.from_iterable(s))

    
    k = 0
    metric = []
    eigs = []
    for o in output:
        if len(o.shape)>2:
            x = o.reshape(o.shape[1],-1)
            print(x.shape)

            _,s,_ = np.linalg.svd(x,False)
            s = s**2

    #        #print(np.asarray(s).min())
            metric.append(np.asarray(s[0:10]).sum())
            eigs.append(s[0:5])
            #print(s.shape)
            k += 1
        if k == 2: 
            break

    store.append(np.asarray(metric).mean())
    store_eigs.append(np.asarray(list(itertools.chain(*eigs))).flatten())
    #print(np.asarray(store_eigs).shape)
    
    
    #metric = []
    #for o in output:
    #    if len(o.shape)>2:
            #print(len(o.shape))
    #        metric.append(np.abs(o).mean())

    #store.append(np.asarray(metric).max())
    #print(store[-1])

#print(store)
store = np.asarray(store)
#store[:] = 0
#print(store)

store_eigs = np.asarray(store_eigs)
print(store_eigs.shape)



import pandas as pd
df=pd.read_csv('METADATA.csv', sep=',',header=None)
targets = df[1].to_numpy()
#print(targets)

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X = store_eigs#.reshape(nmodels,-1)
y = targets[0:nmodels]

print(X.shape)
print(y.shape)

clf = RandomForestClassifier(n_estimators=1000, max_depth=2, criterion='log_loss',  random_state=0)
clf.fit(X[0:80,:],y[0:80])

print(log_loss(y, clf.predict_proba(X)[:,1]))
print(log_loss(y[0:80], clf.predict_proba(X[0:80,:])[:,1]))
print(log_loss(y[80:140], clf.predict_proba(X[80:140,:])[:,1]))



#print(clf.predict_proba(store.reshape(50,-1)))
