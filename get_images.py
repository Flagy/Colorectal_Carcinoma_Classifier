import numpy as np
from skimage.io import imsave
from image_loading import LoadingData

data=LoadingData("json",5)
choice = {'AC':0,'H':1,'S':2,'T':3,'V':4}

for key in choice.keys():
	print(key)
	y_train = data.getYTrain() == choice[key]
	pos = np.where(y_train)[0]
	x_train = data.getXTrain()[pos]
	#print('Training')
	for i in range(len(x_train)):
		imsave(str('C:/Users/super/OneDrive/Documenti/Project_bioinfo/images/training/'+key+'/image_'+key+'_'+str(i)+'.jpg'),x_train[i,:,:,:])
		#if(i%100 == 0):
			#print(i)
	'''y_test = data.getYTest() == choice[key]
	pos = np.where(y_test)[0]
	x_test = data.getXTest()[pos]
	print('Test')
	for i in range(len(x_test)):
		imsave(str('C:/Users/super/OneDrive/Documenti/Project_bioinfo/images/test/'+key+'/image_'+key+'_'+str(i)+'.jpg'),x_test[i,:,:,:])
		if(i%100 == 0):
			print(i)'''
print('Conversion is over!')