# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:04:01 2021

# Filename : CSPfeatureVECTORModLive.py
# Author   : Maximilian Schmitt
# Date     : xx.xx.2019
# eigene Funktion zur extrahiernung eines Featurevektors unter Verwendung
# von Common Spatial Pattern

# not yet fixed for multiclass
"""
import numpy as np
from math import log10

def CSPfeatureVECTORModLive (Filt_Neu, M_FILTER_ROWS):
    
    # input= 2matrixes trial*subband(oder channel zahl)*sample


             
    # FEATURE VECTOR ERSTELLEN
    
    # aus den gefilterten signalen (oder vorher aus der matrix der filtercoeffizienten)
    # wird ein anzahl 2 mal 'm' an signalen extrahiert, die ersten und die
    # letzten 'm' reihen.
    # This property makes the eigenvectors B useful for classification of the two distributions, in fact, the projection
    # of whitened EEG onto the first and last eigenvectors in B will give feature vectors which
    # are optimal for discriminating two populations of EEG in the least squares sense.
            
#     fenster_groese = size(CSP_FILT_SIGNAL_plu,3);
#     M_FILTER_ROWS = 2;%2;
#     ROW_EXTR_PLU = zeros(2*M_FILTER_ROWS,fenster_groese);
#     ROW_EXTR_minu = zeros(2*M_FILTER_ROWS,fenster_groese);
# 
# #      for FVi= 1 : size(CSP_FILT_SIGNAL_plu,1)
# 
#                 ROW_EXTR_PLU(1:M_FILTER_ROWS,:) = CSP_FILT_SIGNAL_plu(1:M_FILTER_ROWS,:);
#                 ROW_EXTR_PLU(M_FILTER_ROWS+1:end,:) = CSP_FILT_SIGNAL_plu(end-M_FILTER_ROWS+1:end,:);
# 
# 
#                 ROW_EXTR_minu(1:M_FILTER_ROWS,:) = CSP_FILT_SIGNAL_minu(1:M_FILTER_ROWS,:);
#                 ROW_EXTR_minu(M_FILTER_ROWS+1:end,:) = CSP_FILT_SIGNAL_minu(end-M_FILTER_ROWS+1:end,:);

    fenster_groese = np.shape(Filt_Neu)[1];
    Zp = np.zeros(shape=(2*M_FILTER_ROWS,fenster_groese));
    Zp[0:M_FILTER_ROWS,:]= Filt_Neu[0:M_FILTER_ROWS,:]
    Zp[M_FILTER_ROWS:2*M_FILTER_ROWS,:]= Filt_Neu[M_FILTER_ROWS-1:M_FILTER_ROWS+1,:];
    # Zp = [Filt_Neu(1:M_FILTER_ROWS,:); Filt_Neu(end-M_FILTER_ROWS+1:end,:)];
#      VAR_Zp = var(Zp);
           #     VAR_Zpi = var(Zp(1,:));
    sum_VAR_zpi =0;
    FEATURE_CSP = np.zeros(shape=(M_FILTER_ROWS*2,1))

    for i in range(M_FILTER_ROWS*2):
        sum_VAR_zpi = sum_VAR_zpi+(np.var(Zp[i,:]));
    # end
    #     sum_VAR_zpi =VAR_Zp*2;
    for i in range(M_FILTER_ROWS*2):
        FEATURE_CSP[i,:] = log10(np.var(Zp[i,:])/sum_VAR_zpi);
    # end
    # FEATURE_VEC = np.transpose(FEATURE_CSP);
    FEATURE_VEC_1 = np.real(sum(sum(np.transpose(FEATURE_CSP))));
    # double sum, since sum once does only reduce the dimensions from 1x4 to 4x in python
    # output = feature vector (trials*classes)... spalte 1 = feature
    # classe für 1 etc.
    # muss für feature tabelle umsortiert werden 
#     end
      
    return(FEATURE_VEC_1)