# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:46:49 2018

@author: super
"""

from vgg16 import VGG16
import os

os.chdir("C:/Users/super/OneDrive/Documenti/Project_bioinfo")

#VGG16('HS').start('no_aug','ft','both')
VGG16('HAC').start('no_aug','fulltraining','softmax')
VGG16('HAC').start('no_aug','fulltraining','sigmoid')


#VGG16('HS').start('no_aug','fulltraining')
#VGG16('ACTV',3).start('no_aug','fulltraining')


