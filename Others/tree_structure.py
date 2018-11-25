# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:36:28 2018
0 = AC/AD, 1 = H,  0 = S, 1 = T, 2 = V
@author: super
"""
import numpy as np

''' 
Philosofy:
If the object can be classified with enough accuracy 
in one of the nodes the tree will grow vertically, otherwise
it'll grow orizontally.

'''

if __name__ == "__main__":
    '''
    Note, this L is obtained by a CNN trained on HAC dataset.
    If neuron 0 lights on means that the net classifies the element as AC otherwise as H
    Since we are feeding the net with non H or AC elements we are developing some rules to
    understand where the element belongs to. An attivation power close to neuron 0 means the element
    has features close to the H, same for AC. New classes are needed where the 2 activation powers are similar.
    '''
    act_func = 'softmax'
    f = open('test.txt','w')
    L = np.load('Likelihood_matrix_'+act_func+'.npy')[:,0,:]
    L_dict = {'AC':L[0],'H':L[1],'S':L[2],'T':L[3],'V':L[4]}
    T = 0.7 # user defined threshold a priori 
    print('activation function is:',act_func,'T is: ',T,file = f)
    label = ('AC','H')
    for param in L_dict.keys():
        classification = label[np.argmax(L_dict[param])]
        print('Classification:',classification,'\tParam:',param, file = f)
        val = max(L_dict[param])
        if val >= T: 
            #The new class is added to the child node
            print('The tree grows vertically,',param,'is joint to the class',classification, file = f)
        else:
            print("The tree grows orizontally,",param,'is a new child node', file = f)

    print('\n',file = f)
    f.close()



            
        
        
        
    