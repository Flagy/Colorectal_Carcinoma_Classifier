
from keras.models import model_from_json
from keras import optimizers
import os
from Token import TOKEN
import telepot
import numpy as np

bot = telepot.Bot(TOKEN)
chat_id= "-287388612"

x_test_ACHS = np.load('dataset/ACHS/ACHS_x_test.npy')
y_test_ACHS = np.load('dataset/ACHS/ACHS_y_test.npy')
x_test_ACTV = np.load('dataset/ACTV/ACTV_x_test.npy')
y_test_ACTV = np.load('dataset/ACTV/ACTV_y_test.npy')
x_test_tot = np.load('database5/test/x_test_tot.npy')
y_test_tot = np.load('database5/test/y_test_tot.npy')

os.chdir('Nets/')

def charge_model(param):
    filename = param + ".json"
    json_file = open(filename,"r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    filename = param + "_w.h5"
    model.load_weights(filename)
    sgd = optimizers.SGD(lr=0.0001, nesterov=True)
    model.compile(optimizer=sgd,loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
    model.summary()
    return model

#model = charge_model('ACHSsoftmax')
#bot.sendMessage(chat_id,"Evaluation ACHSsoftmax: "+str(model.evaluate(x_test_ACHS,y_test_ACHS,batch_size=16)))
#model = charge_model('ACHSsigmoid')
#bot.sendMessage(chat_id,"Evaluation ACHSsigmoid: "+str(model.evaluate(x_test_ACHS,y_test_ACHS,batch_size=16)))
#model = charge_model('ACHSboth')
#bot.sendMessage(chat_id,"Evaluation ACHSboth: "+str(model.evaluate(x_test_ACHS,y_test_ACHS,batch_size=16)))
model = charge_model('ACTVsoftmax')
bot.sendMessage(chat_id,"Evaluation ACTVsoftmax: "+str(model.evaluate(x_test_ACTV,y_test_ACTV,batch_size=16)))
model = charge_model('ACTVsigmoid')
bot.sendMessage(chat_id,"Evaluation ACTVsigmoid: "+str(model.evaluate(x_test_ACTV,y_test_ACTV,batch_size=16)))
model = charge_model('ACTVboth')
bot.sendMessage(chat_id,"Evaluation ACTV5both: "+str(model.evaluate(x_test_ACTV,y_test_ACTV,batch_size=16)))
model = charge_model('5classessoftmax')
bot.sendMessage(chat_id,"Evaluation 5classessoftmax: "+str(model.evaluate(x_test_tot,y_test_tot,batch_size=16)))
model = charge_model('5classessigmoid')
bot.sendMessage(chat_id,"Evaluation 5classessigmoid: "+str(model.evaluate(x_test_tot,y_test_tot,batch_size=16)))
model = charge_model('5classesboth')
bot.sendMessage(chat_id,"Evaluation 5classesboth: "+str(model.evaluate(x_test_tot,y_test_tot,batch_size=16)))