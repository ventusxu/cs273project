# -*- coding: utf-8 -*-
"""
Created on Sat May 26 20:19:08 2018

@author: Zhixuan
"""

import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import ShuffleSplit
from sklearn import neighbors

Xtr = np.genfromtxt("data/X_train.txt",delimiter=None)
Ytr = np.genfromtxt("data/Y_train.txt",delimiter=None)
Xte = np.genfromtxt("data/X_test.txt",delimiter=None)

Xtr,param = ml.transforms.rescale(Xtr)
Xte,_ = ml.transforms.rescale(Xte)

#%%
out4 = np.all([Xtr[:,4]>12000,Ytr>30],axis=0)
out5 = np.all([Xtr[:,5]>6000, Ytr>30],axis=0)
out7 = np.any([np.all([Xtr[:,7]>15, Ytr>20],axis=0),\
               np.all([Xtr[:,7]>25, Ytr>10],axis=0)],\
              axis=0)
out8 = np.any([np.all([Xtr[:,8]>17,   Ytr>10],axis=0),\
               np.all([Xtr[:,8]>17.5, Ytr>10],axis=0)],axis=0)
out9 =  np.all([Xtr[:,9]>11,   Ytr>10],axis=0)
out10 = np.all([Xtr[:,10]>13,  Ytr>20,  Ytr<40],axis=0)
out12 = np.all([Xtr[:,12]>40,  Ytr>10],axis=0)
out13 = np.all([Xtr[:,13]>400, Ytr>10],axis=0)

outX0 = np.any([Xtr[:,9]==0, Xtr[:,10]==0, Xtr[:,11]==0], axis=0)

outMask = np.any([Xtr[:,7]>40, Xtr[:,12]>400, Xtr[:,11]>40],axis=0)

outMask = np.any([outMask, out4, out5, out7, out8, out10, out12, out13], axis=0)
print(np.sum(outMask))
X,Y = Xtr[~outMask],Ytr[~outMask]

# Cross Validation
rs0 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
Xsp0 = rs0.split(X)

XtCV = []
XvCV = []
YtCV = []
YvCV = []
for tr_idx, va_idx in Xsp0:
    XtCV.append(X[tr_idx])
    XvCV.append(X[va_idx])
    YtCV.append(Y[tr_idx])
    YvCV.append(Y[va_idx])

#%% KNN method
K = [2**k for k in range(2,13)]
errt = []
errv = []
for k in K:
    print("k=",k)
    errtk = 0
    errvk = 0
    knnL = neighbors.KNeighborsRegressor(k)
    for i in range(5):
        Xt,Yt = XtCV[i], YtCV[i]
        Xv,Yv = XvCV[i], YvCV[i]
        knnL.fit(Xt,Yt)
        errvi = mse(knnL.predict(Xv),Yv)
        errti = mse(knnL.predict(Xt),Yt)
        print("errt: %.5f\t"%errti,"errv: %.5f"%errvi)
        errtk += errti
        errvk += errvi
    errtk /= 5
    errvk /= 5
    errv.append(errvk)
    errt.append(errtk)

#%% KNN plot
plt.semilogx(K,errt,"*-",label="Train Err")
plt.semilogx(K,errv,"*-",label="Valid Err")
plt.xticks(K,K)
plt.title("KNN Err vs K")
plt.xlabel("k")
plt.ylabel("err")
plt.legend()
plt.show()

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
YteHat[YteHat<0] = 0
fh = open("Yte_knn_outlier.csv",'w')
fh.write('ID,prediction\n')
for i, pred in enumerate(YteHat):
    fh.write("{},{}\n".format(i,pred))

fh.close()

#%% Data Visualization
#for i in range(14):
#    plt.figure()
#    plt.clf()
#    plt.title("feature %d"%i)
#    for y in range(0,80,10):
#        mask1 = Ytr>=y
#        mask2 = Ytr<(y+10)
#        mask = np.all([mask1,mask2,~outMask], axis=0)
#        plt.plot(Xtr[mask][:,i],Ytr[mask],".")
#    plt.show()