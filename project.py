import mltools as ml
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse

#%%

Xtr = np.genfromtxt("data/X_train.txt",delimiter=None)
Ytr = np.genfromtxt("data/Y_train.txt",delimiter=None)
Xte = np.genfromtxt("data/X_test.txt",delimiter=None)

#%%
out4 = np.all([Xtr[:,4]>12000,Ytr>30],axis=0)
out5 = np.all([Xtr[:,5]>6000, Ytr>30],axis=0)
out7 = np.any([np.all([Xtr[:,7]>15, Ytr>20],axis=0),\
               np.all([Xtr[:,7]>25, Ytr>10],axis=0)],\
              axis=0)
out10 = np.all([Xtr[:,10]>13,  Ytr>20,  Ytr<40],axis=0)
outX0 = np.any([Xtr[:,9]==0, Xtr[:,]])
out12 = np.all([Xtr[:,12]>40,  Ytr>10], axis=0)
out13 = np.all([Xtr[:,13]>400, Ytr>10], axis=0)

outMask = np.any([Xtr[:,7]>40, Xtr[:,12]>400, Xtr[:,11]>40],axis=0)

outMask = np.any([outMask, out4, out5, out7, out10, out12, out13], axis=0)
print(np.sum(outMask))
#%%
#
#X = Xtr[~outMask]
#Y = Ytr[~outMask]
#
#Xt = X[0:50000]
#Yt = Y[0:50000]
#Xv = X[10000:80000]
#Yv = Y[10000:80000]
#Xt,param = ml.transforms.rescale(Xt)
#Xv,_ = ml.transforms.rescale(Xv,param)
##Xt = sel.fit_transform(Xt)
##sel.fit_transform(Xv)
#for i in range(1,10):
#    mlp = MLPRegressor(hidden_layer_sizes=(2, ),
#                       activation='relu', 
#                       solver='lbfgs', 
#                       alpha=0.000009555, 
#                       batch_size=150, 
#                       learning_rate='adaptive', 
#                       learning_rate_init=0.001, 
#                       power_t=0.5, max_iter=175, 
#                       shuffle=True, 
#                       random_state=None, 
#                       tol=0.000011, 
#                       verbose=False, 
#                       warm_start=False, 
#                       momentum=0.5, 
#                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
##                       beta_1=0.9, 
##                       beta_2=0.999, 
##                       epsilon=1e-08
#                       )
#
#    mlp.fit(Xt, Yt)
#    YHat = mlp.predict(Xv)
#    err=mse(YHat,Yv)
#    print('size:',i, 'err',err)
#%% Data Visualization
for i in range(14):
    plt.figure()
    plt.clf()
    plt.title("feature %d"%i)
    for y in range(0,80,10):
        mask1 = Ytr>=y
        mask2 = Ytr<(y+10)
        mask = np.all([mask1,mask2,~outMask], axis=0)
        plt.plot(Xtr[mask][:,i],Ytr[mask],".")
    plt.show()