# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:05:04 2021
# Filename : CSPown.m
# Author   : Maximilian Schmitt
# Date     : xx.xx.2019
# Funktion zur extrahiernung eines Featurevektors unter Verwendung
# von Common Spatial Pattern

# modifikation, basierend auf CSP.m Function Coded by James Ethridge and William Weaver

# vargin{1} ist der input nach dessen lable man sucht/ das Inputsignal zur
# aktuellen klasse
"""
import numpy as np
from numpy.linalg import inv
import numpy.linalg as LA # import linear algebra stuff for eigenvalues etc.
# la.eig eigenvalues of 1 input matrix is needed
import scipy.linalg as la
# la.eig eigenvalues of 2 input matrices are needed




def CSPown(*args):
    # b5= [FBCSP_SUBB_UBERGABE_PLU_C3,FBCSP_SUBB_UBERGABE_PLU_C3]
    varargin = args
    nargin = len(args);

    if (nargin != 2):
        print('Must have 2 classes for CSP!')# mind. 2 signals - signals for hand and foot
    # end
    dim_E=len(args[0])
    E_spatia_cova = np.zeros(shape=(len(args),dim_E,dim_E))
    E_tra = np.zeros(shape=2)
    NORM_SPATIA_COVA = np.zeros(shape=(len(args),dim_E,dim_E))
    C = np.zeros(shape=(len(args),dim_E,dim_E))

    COMPOSIT_SPATIA_COV=0;
    # COMP_COV_Rest = np.zeros(shape=(len(b5),dim_E,dim_E));
    COMP_COV_Rest = 0;
    #finding the covariance of each class and composite covariance
    for i in range(nargin):
        # input data of single trial, number channels X number sample matrix
        varargin_transpose = np.transpose(varargin[i])
        E_spatia_cova[i] = np.matmul(varargin[i],varargin_transpose);
        # signal mal seiner transponierten gibt eine number-channel X number_channel matrix(105x3500*3500x105)
        E_tra[i] = np.trace(E_spatia_cova[i]); # trace summiert die diagonal elemente auf
        # matrix*matrix' -> symmetrisch =>  basis of eigenvectors and every eigenvalue is real
        # In case A is not a square matrix and A*A' is too large to efficiently compute the eigenvectors 
        # (like it frequently occurs in covariance matrix computation), 
        # then it's easier to compute the eigenvectors of A'*A given by A'*A*ui=?i*ui.
        NORM_SPATIA_COVA[i] = E_spatia_cova[i] / E_tra[i];  # normalized spatial covariance - Ramoser (1)
        #instantiate me before the loop!

        C[i,:,:] = NORM_SPATIA_COVA[i];
        
        #mean here?
        # test=mean (NORM_SPATIA_COVA_plu(1,:));
        # 
        #     AVER_SPATIA_COV_plu = mean (NORM_SPATIA_COVA_plu,1);
        #     AVER_SPATIA_COV_minu = mean (NORM_SPATIA_COVA_minu,1);
        #     # the spatial covariance Cd; d element of [plu; minu] is calculated by averaging over the trials of each group.
        # Cov_spatia= AVER_SPATIA_COV_plu'+AVER_SPATIA_COV_minu';
        ### STEHT ZWAR SO IM PAPER, ERGIBT ABER KEINE QUADRATISCHE MATRIX UND WURDE
        ### IM ALTERNATIVEN CODE AUCH NICHT VERWENDET

        COMPOSIT_SPATIA_COV = COMPOSIT_SPATIA_COV + NORM_SPATIA_COVA[i];     # Ramoser (2)
#        COMPOSIT_SPATIA_COV = COMPOSIT_SPATIA_COV + E_spatia_cova{i};  % Sheng Ge (1) 
        # Sheng Ge (1) from an multicass-classification paper
        if i>0:
            COMP_COV_Rest = COMP_COV_Rest + NORM_SPATIA_COVA[i];
            # COMP_COV_Rest = C2+C3+C4+... für Lable 1; da nur 2 zustände -> Rest = zustand 2
            # für multiclass-classification, ansonsten bleibt COMP_COV_Res = NORM_SPATIA_COVA{2}
        # end
    # end 

    # The composite spatial covariance can be factored as C = Uc*lambda_c*Uc', where Uc is the matrix of eigenvectors and lambda_c is the diagonal matrix of eigenvalues.
    # [Uc,lambda_c] = eig (COMPOSIT_SPATIA_COV);
    lambda_c, Uc = LA.eig(COMPOSIT_SPATIA_COV)
    # Matlab: [V, D] = eig(A)
    # Python: D, V = la.eig(A) :https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html
    # # returns diagonal matrix lambda_c of generalized eigenvalues 
    # # and full matrix Uc whose columns are the corresponding right eigenvectors, so that A*Uc = B*Uc*lambda_c.
        # vergleichs test mit teil der anderen fkt:
        #   Sort eigenvalues in descending order
        # [lambda_c_zwei,ind] = np.sort(diag(lambda_c),'descend');
    lambda_diag = np.diag(lambda_c)
    ind = lambda_diag.argsort()[::-1]
    lambda_c_zwei = lambda_diag[ind]
    Uc_zwei = Uc[:,ind];
        # NOTIZ!!!:
        # eigenvalue/vector matrizen müssen nachsortiert werden, da die funktion eig
        # diese nicht immer geordnet auswirft, dies ist eine eigenheit der eig-funktion
        # mathematisch ist das automatisch der fall, deshalb steht das nicht im paper
        # https://de.mathworks.com/help/matlab/ref/eig.html
        
    # matlab diag gibt zero matrix die elemente der diagonalen enthät
    # python diag gibt liste (number_elements,) die die elemente der diagonalen enthält
    Uc_zwei_transpose = Uc_zwei[0].transpose()
    # lambda_diag_inv = inv(lambda_c_zwei[0])
    # original: WHITENING_TRANSFORM = sqrt(inv(diag(lambda_c_zwei))) * Uc_zwei';    % Ramoser (3)
    # ein diag() ist in python weg gefallen wegen python vector fuckery
    WHITENING_TRANSFORM = np.matmul(np.sqrt(inv(lambda_c_zwei[0])), Uc_zwei_transpose);    # Ramoser (3)
    WHITENING_TRANSFORM_transpose = np.transpose(WHITENING_TRANSFORM)

    # WHITENING_TRANSFORM = sqrt(inv(lambda_c))*Uc';  # Ramoser (3)
    SHARED_PATT = np.zeros(shape=(len(args),dim_E,dim_E))

    for k in range(nargin):
        if k == 0:
            SHARED_PATT[k] = np.matmul(WHITENING_TRANSFORM, NORM_SPATIA_COVA[k], WHITENING_TRANSFORM_transpose);    # Ramoser (4) - Sl = P*Cl*P' and Sr = P*Cr*P'
        else:
            SHARED_PATT[k] = np.matmul(WHITENING_TRANSFORM, COMP_COV_Rest, WHITENING_TRANSFORM_transpose);
        # end
    # end 

    # Ramoser equation (5)
    # [U{1},Psi{1}] = eig(S{1});
    # [U{2},Psi{2}] = eig(S{2});

    #generalized eigenvectors/values
    # [B,D] = eig(SHARED_PATT[1],SHARED_PATT[2]);
    D,B = la.eig(SHARED_PATT[0],SHARED_PATT[1])
    # Simultanous diagonalization
    # Should be equivalent to [B,D]=eig(S{1});
    #sort ascending by default
    #[Psi{1},ind] = sort(diag(Psi{1})); U{1} = U{1}(:,ind);
    #[Psi{2},ind] = sort(diag(Psi{2})); U{2} = U{2}(:,ind);
    
    # SORT EIGENVALUES IN PYTHON:
    # w, v = LA.eig(A)
    # idx = np.argsort(w)
    # w = w[idx]
    # v = v[:,idx]
    
    idx= np.argsort(D)
    D = D[idx]
    B = B[:,idx]
    # [D,ind]=sort(np.diag(D)); #sortieren an dieser stelle um größte eigenwerte zu ermitteln
    # B=B[:,idx];
    B_transpose = np.transpose(B)
    #Resulting Projection Matrix-these are the spatial filter coefficients
    CSPcoeffi = np.matmul(B_transpose, WHITENING_TRANSFORM); # 3x3 array
    # The columns of WHITENING_TRANSFORM^1 are the common spatial patterns and can be seen as time-invariant EEG source distribution vectors.
    # b18 = [CSPcoeffi, B, idx, COMPOSIT_SPATIA_COV, C]

    return(CSPcoeffi, B, idx, COMPOSIT_SPATIA_COV, C)

