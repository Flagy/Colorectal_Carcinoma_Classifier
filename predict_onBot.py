import telepot
from telepot.loop import MessageLoop
from telepot.namedtuple import InlineKeyboardMarkup, InlineKeyboardButton
from pprint import pprint
from PIL import Image
import requests
import numpy as np
from keras.models import model_from_json
from keras import optimizers
import os
from Token import TOKEN

os.chdir('Nets/')
bot = telepot.Bot(TOKEN)
API = 'https://api.telegram.org'
url_bot = API +'/bot'+ TOKEN
url_files = API + '/file/bot'+ TOKEN
label_5classes = ['AC','H','S','T','V']
label_ACHS = ['AC','H','S']
label_ACTV = ['AC','T','V']
decided = False


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


def tot_classes_pred(img):
    model = charge_model('Softmax/5classes')
    data = np.asarray(img, dtype="uint8")
    image_batch = np.expand_dims(data, axis=0)
    prediction = model.predict(image_batch)
    return(prediction)


def TreeCNN_pred(img):
    model = charge_model('Softmax/ACHS')
    data = np.asarray(img, dtype="uint8")
    image_batch = np.expand_dims(data, axis=0)
    pred = model.predict(image_batch)
    print(pred)
    acc_branch1 = np.max(pred)
    try:
	    if np.argmax(pred) != 0:
	        prognosis = label_ACHS[np.argmax(pred)]
	        accuracy = str("{0:.2f}".format((acc_branch1*100)))+'%'
	    else:
	        print("Charging ACTV model")
	        model = charge_model('Softmax/ACTV')
	        pred2 = model.predict(image_batch)
	        prognosis = label_ACTV[np.argmax(pred2)]
	        acc_branch2 = np.max(pred2)
	        #Law of total probability
	        accuracy = acc_branch1*acc_branch2
			#It's true that other terms should be added but they should
	        #be neglictables and above all in this way it's more preservative
	        accuracy = str("{0:.2f}".format((accuracy*100)))+'%'
	        print('Prediction1:',pred,'\nPrediction2:',pred2,'\nacc_branch1:',acc_branch1,'\nacc_branch2:',acc_branch2,'\nTot accuracy:',accuracy)
    except:
    	raise AssertionError("Unexpected value of 'prediction'!", str(pred))
    return(prognosis,accuracy)


def on_chat_message(msg):
    global query_id, from_id, query_data, decided
    pprint(msg)
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(decided)
    if content_type == 'photo' and decided: 
		#when you download a pic from telegram servers you can choose among the compression. 
		#The last one has no compression and mantain the original size of the picture.
        file_id = msg['photo'][-1]['file_id']
        #print(file_id)
        file_path = requests.get(url_bot+'/getfile?file_id='+file_id).json()['result']['file_path']
        #print(file_path)
        url_img = url_files+'/'+file_path
        img = Image.open(requests.get(url_img, stream = True).raw)
        #img.show()
        print('Callback query:',query_id, from_id, query_data)
        try:
	        if query_data == '5classes':
	            result = tot_classes_pred(img)
	            print(result)
	            prognosis = label_5classes[np.argmax(result)]
	            accuracy = str("{0:.2f}".format((np.max(result)*100)))+'%'
	            bot.sendMessage(chat_id,"In my opinion this is class "+prognosis+' whit accuracy of '+accuracy)
	        elif query_data == 'TreeCNN':
	            prognosis,accuracy = TreeCNN_pred(img)
	            bot.sendMessage(chat_id,"In my opinion this is class "+prognosis+' whit accuracy of '+accuracy)
        except:
            raise AssertionError("Unexpected value of 'query_data'!", query_data)
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text = '5classes for 5 classes',callback_data = '5classes')],
            [InlineKeyboardButton(text = 'Tree CNN',callback_data = 'TreeCNN')]])
        bot.sendMessage(chat_id, "Choose a net and send a picture", reply_markup = keyboard)


def on_callback_query(msg):
    global query_id, from_id, query_data, decided
    query_id, from_id, query_data = telepot.glance(msg, flavor = 'callback_query')
    print('Callback query:',query_id, from_id, query_data)
    decided = True
    bot.answerCallbackQuery(query_id, text = 'Great, now send me a picture!')
  

MessageLoop(bot, {'chat':on_chat_message,'callback_query':on_callback_query}).run_forever()
