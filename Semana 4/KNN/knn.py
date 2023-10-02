# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:49:29 2020

@author: Victor
"""
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def aprendizaje(self,X,C):
        self.X=X # matriz de vectores de caracteristicas
        self.c=C # clases asociadas a cada vector x(n)
        self.n_muestras=X.shape[1] # cantidad de muestras
    
    def clasificacion(self,Y):
        clases=[]
        for i in range(Y.shape[1]): # por cada vector y(n) a clasificar
            distancias=np.empty(self.n_muestras)
            for n in range(self.n_muestras): # por cada vector x(n) de caracteristicas
                distancias[n]=EUCLIDIANA(self.X[:,n],Y[:,i])
            
            # distancias mas cercanas
            k_distancias=np.argsort(distancias)
            # identificar las k distancias - clases
            k_etiqueta=self.c[k_distancias[:self.k]]
            print(k_etiqueta)
            # votacion
            c = Counter(k_etiqueta).most_common(1)#(5,0)
            clases.append(c[0][0]) # almacenamos la clase asignada al vector y(n)
        return clases
            

def EUCLIDIANA(x,y):
    return np.sqrt(np.sum((x-y)**2))