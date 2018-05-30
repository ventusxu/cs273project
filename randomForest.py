# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:22:55 2018

@author: Zhixuan
"""

import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import ShuffleSplit

#%%

Xtr = np.genfromtxt("data/X_train.txt",delimiter=None)
Ytr = np.genfromtxt("data/Y_train.txt",delimiter=None)
Xte = np.genfromtxt("data/X_test.txt",delimiter=None)

Xtr,param = ml.transforms.rescale(Xtr)
Xte,_ = ml.transforms.rescale(Xte,param)

X = Xtr[:50000]
Y = Ytr[:50000]

rs0 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
Xsp0 = rs0.split(X)

#for i, j in Xsp0:
#    pick = i
#
#X0,Y0 = Xtr[pick],Ytr[pick]
## Cross Validation
#rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
#Xsplit = rs.split(X0)

XtCV = []
XvCV = []
YtCV = []
YvCV = []
for tr_idx, va_idx in Xsp0:
    XtCV.append(X[tr_idx])
    XvCV.append(X[va_idx])
    YtCV.append(Y[tr_idx])
    YvCV.append(Y[va_idx])

errTD = []
errVD = []
D = list(range(5,60,5))
for d in D:
    errti = []
    errvi = []
    for i in range(5):
        rfr = RFR(n_estimators=50,max_depth=d)
        rfr.fit(XtCV[0],YtCV[0])
        YtHat = rfr.predict(XtCV[0])
        YvHat = rfr.predict(XvCV[0])
        errti.append(mse(YtCV[0],YtHat))
        errvi.append(mse(YvCV[0],YvHat))
    errti = np.array(errti)
    errvi = np.array(errvi)
    errTD.append(np.mean(errti))
    errVD.append(np.mean(errvi))

#%%
plt.plot(D, errTD,'*-', label='Train Err')
plt.plot(D, errVD,'*-', label='Valid Err')
plt.legend()
plt.title('RandomForest Err vs MaxDepth')
plt.xticks(D,D)
plt.xlabel('depth')
plt.ylabel('err')
plt.savefig('rf_depth',dpi=2000)
plt.show()