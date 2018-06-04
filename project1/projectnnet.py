import mltools as ml
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse

X = np.genfromtxt("X_train.txt", delimiter=None)
Y = np.genfromtxt("Y_train.txt", delimiter=None)
Xt = X[0:50000]
Yt = Y[0:50000]
Xv = X[10000:80000]
Yv = Y[10000:80000]
Xt,param = ml.transforms.rescale(Xt)
Xv,_ = ml.transforms.rescale(Xv,param)
#Xt = sel.fit_transform(Xt)
#sel.fit_transform(Xv)
for i in range(1,10):
    mlp = MLPRegressor(hidden_layer_sizes=(2, ),
                       activation='relu', 
                       solver='lbfgs', 
                       alpha=0.000009555, 
                       batch_size=150, 
                       learning_rate='adaptive', 
                       learning_rate_init=0.001, 
                       power_t=0.5, max_iter=175, 
                       shuffle=True, 
                       random_state=None, 
                       tol=0.000011, 
                       verbose=False, 
                       warm_start=False, 
                       momentum=0.5, 
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
#                       beta_1=0.9, 
#                       beta_2=0.999, 
#                       epsilon=1e-08
                       )

    mlp.fit(Xt, Yt)
    YHat = mlp.predict(Xv)
    err=mse(YHat,Yv)
    print('size:',i, 'err',err)

Xte,_ = ml.transforms.rescale(Xte,param)
Yhat = mlp.predict(Xte)
fh = open('pred_nnet%d.csv'%i,'w')
fh.write('ID,Prediction\n')
for i,yi in enumerate(Yhat):
    fh.write('{},{}\n'.format(i,yi))
    
fh.close()