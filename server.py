import random
import string
import cherrypy
import request
import base64
import json
import numpy as np
import cv2
from testing import Testing
import io
from imageio import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg









class NN(object):
    exposed = True
    def GET (self,*uri,**args):
        return ("model working")

    def POST (self,*uri):
        print(uri[0])


        print('arriving')
        image=cherrypy.request.body.read()
        img =imread(io.BytesIO(base64.b64decode(image)))
        print(img.shape)


        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(cv2_img.shape)
        img=np.asarray(cv2_img)
        print(img)

        dic= test.predict(img,uri[0])
        if uri[0]=="1":
            typeofnet="VGG16 5 classes"
        else:
            typeofnet="Tree CNN"
        print(dic)
        print(dic["class"])
        data={
        "typeofnet":typeofnet,
        "percentage":str(dic["percentage"]),
        "class":dic["class"]
        }
        return (json.dumps(data))









if __name__ == '__main__':
    test=Testing()
    conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			'tools.sessions.on': True
		}
	}
    cherrypy.tree.mount(NN(), '/', conf)
    cherrypy.config.update({'server.socket_host': '192.168.56.1'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
