# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:45:13 2018

@author: Zhixuan
"""
import matplotlib.pyplot as plt

errt = []
errv = []
K = [2**k for k in range(2,13)]
with open('knn_errv','r') as ve:
    vtmp = ve.read().splitlines()

with open('knn_errt','r') as te:
    ttmp = te.read().splitlines()
    
i=0
errti = 0
for e in ttmp:
    errti += float(e)
    i += 1
    if i == 10:
        errt.append(errti/10)
        errti = 0
        i = 0
         
i=0
errvi = 0
for e in vtmp:
    errvi += float(e)
    i += 1
    if i == 10:
        errv.append(errvi/10)
        errvi = 0
        i = 0
        
plt.semilogx(K,errt,"*-",label="Err Train")
plt.semilogx(K,errv,"*-",label="Err Valid")
plt.legend()
plt.xticks(K,K)
plt.xlabel("k")
plt.ylabel("err")
plt.title("KNN Err vs K w/ CV")
plt.savefig("knn_k_cv",dpi=2000)
plt.show()