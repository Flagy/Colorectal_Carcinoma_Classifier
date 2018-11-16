# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:46:49 2018

@author: super
"""

from vgg16 import VGG16
#import os

#os.chdir("C:/Users/super/OneDrive/Documenti/Project_bioinfo")


VGG16('5classes').start('fulltraining','softmax')
VGG16('5classes').start('fulltraining','sigmoid')
VGG16('5classes').start('fulltraining','both')

VGG16('ACHS').start('fulltraining','softmax')
VGG16('ACHS').start('fulltraining','sigmoid')
VGG16('ACHS').start('fulltraining','both')

VGG16('ACTV').start('fulltraining','softmax')
VGG16('ACTV').start('fulltraining','sigmoid')
VGG16('ACTV').start('fulltraining','both')




