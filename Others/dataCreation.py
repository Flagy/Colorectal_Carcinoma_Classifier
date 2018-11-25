import numpy as np
import os


AC=np.load("AC_test.npy")
T=np.load("T_test.npy")
V=np.load("V_test.npy")

AC_y=np.zeros(AC.shape[0])
T_y=np.zeros(T.shape[0])+1
V_y=np.zeros(V.shape[0])+2

ACTV = np.concatenate((AC,T,V))
ACTV_y = np.concatenate((AC_y,T_y,V_y))
np.save('ACTV_x_test.npy',ACTV)
np.save('ACTV_y_test.npy',ACTV_y)


