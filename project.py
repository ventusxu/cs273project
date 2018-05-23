import mltools as ml
import numpy as np

Xtr = np.genfromtxt("data/X_train.txt",delimiter=None)
Ytr = np.genfromtxt("data/Y_train.txt",delimiter=None)
Xte = np.genfromtxt("data/X_test.txt",delimiter=None)

print(Xtr.shape)
print(Ytr.shape)
print(Xte.shape)