# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:36:05 2018

@author: Zhixuan
"""

from sklearn import neighbors
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error as mse
import mltools as ml
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#%%

Xtr = np.genfromtxt("data/X_train.txt",delimiter=None)
Ytr = np.genfromtxt("data/Y_train.txt",delimiter=None)
Xte = np.genfromtxt("data/X_test.txt",delimiter=None)
Xtr,param = ml.transforms.rescale(Xtr)
Xte,_ = ml.transforms.rescale(Xte,param)

rs0 = ShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
Xsp0 = rs0.split(Xtr)
for i, j in Xsp0:
    pick = i

X0,Y0 = Xtr[pick],Ytr[pick]
# Cross Validation
rs = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
Xsplit = rs.split(X0)

XtCV = []
XvCV = []
YtCV = []
YvCV = []
for tr_idx, va_idx in Xsplit:
    XtCV.append(X0[tr_idx])
    XvCV.append(X0[va_idx])
    YtCV.append(Y0[tr_idx])
    YvCV.append(Y0[va_idx])

#%% KNN method
K = [2**k for k in range(2,13)]
errt = []
errv = []
for k in K:
    print("k=",k)
    errtk = 0
    errvk = 0
    knnL = neighbors.KNeighborsRegressor(k)
    for i in range(10):
        Xt,Yt = XtCV[i], YtCV[i]
        Xv,Yv = XvCV[i], YvCV[i]
        knnL.fit(Xt,Yt)
        errvi = mse(knnL.predict(Xv),Yv)
        errti = mse(knnL.predict(Xt),Yt)
        print("errt: %.5f\t"%errti,"errv: %.5f"%errvi)
        errtk += errti
        errvk += errvi
    errtk /= 10
    errvk /= 10
    errv.append(errvk)
    errt.append(errtk)

#%% KNN plot
plt.semilogx(K,np.array(errt)*2,"*-",label="Train Err")
plt.semilogx(K,np.array(errv)*2,"*-",label="Valid Err")
plt.xticks(K,K)
plt.title("KNN Err vs K")
plt.xlabel("k")
plt.ylabel("err")
plt.legend()
plt.savefig("knn_k_outlier",dpi=2000)
plt.show()

#%%
print(np.mean(errv[5]*2))

#%% Final KNN
knnL = neighbors.KNeighborsRegressor(128)
knnL.fit(X,Y)
YteHat = knnL.predict(Xte)
