from flask import Flask,render_template,request
from scipy.misc import imsave,imread,imresize
import numpy as np 
import tensorflow as tf
import keras.models 
import re
import sys
import os
sys.path.append(os.path.abspath('./model')) #Add the folder named model where python looks to find modules or functions
from load import *
import base64

#initalize flask app
app= Flask(__name__) 
global model,graph
model,graph=init() #Call the init() function located in /model folder which loads the graph and model to return here
sess,tf_input,output=init_bangla("bangla")
label_bangla=['অ','ও','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ঔ','ক','খ','গ','ঘ','ঙ','চ','ছ','জ','ঝ','ঞ','ট','ঠ','ড','ঢ','ণ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','র','ল','শ','ষ','স','হ','ড়','ঢ়','য়','ৎ','ং','ঃ','ঁ','০','১','২','৩','৪','৫','৬','৭','৮','৯']

label_bangla_kolkata=['অ','আ','ই','ঈ','উ','ঊ','এ','ঐ','ও','ঔ','ঋ','ক','খ','গ','ঘ','ঙ','চ','ছ','জ','ঝ','ঞ','ট','ঠ','ড','ঢ','ণ','ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','র','ল','শ','ষ','স','হ','ড়','ঢ়','য়','ং','ঃ','ঁ','ৎ','০']

def convertImage(imgData): ##function use to convert image and save it as output.png
	imgstr= re.search(b'base64,(.*)',imgData).group(1) #decode it from base64 into raw binary data, as js use base64 to decode image
	print("In ConvertImage")
	with open('output.png','wb') as output:
		output.write(base64.b64decode(imgstr))

@app.route('/') #index route
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	print("In Predict")
	imgData=request.get_data()#get raw input image data from the views 
	convertImage(imgData)
	x= imread('output.png', mode='L') #read the image into memory
	x= np.invert(x)			#Turn Black into white and white into black makes it easy to classify
	x= imresize(x,(28,28))	#Reshape the input image into 28x28 size

	imsave('output_conv.png',x)
	#print(x)
	x= x.reshape(1,28,28,1) #4D tensor feed into our model
	with graph.as_default():
		out=model.predict(x) #give the image input to keras model to predict
		print(out)
		response= np.array_str(np.argmax(out,axis=1))#axis=1 means single dimentional response i.e one string of prediction
		return response
@app.route('/predict_bangla/',methods=['GET','POST'])
def predict_bangla():
	print("In Predict Bangla")
	imgData=request.get_data()
	convertImage(imgData)
	x=imread('output.png',mode='L')
	x=np.invert(x)
	x=imresize(x,(28,28))

	imsave('output_conv.png',x)

	x=x.reshape(1,28,28,1)
	##out=sess.run(output,feed_dict={tf_input:x})
	out=sess.run(tf.argmax(output,1),feed_dict={tf_input:x})
	print(out)
	#response= np.argmax(out,axis=1)#axis=1 means single dimentional response i.e one string of prediction
	#print(response)
	res=label_bangla[int(out)]
	#print (res)
	#print (response)
	#res=label_bangla[0]
	return res
	# try:
	# 	res=label_bangla[response]
	# 	print(response)
	# 	return res
	# except:
	# 	print("Array out of bound")
	# 	return response

@app.route('/predict_bangla_kolkata/',methods=['GET','POST'])
def predict_bangla_kolkata():
	print("In Predict Bangla Kolkata")
	imgData=request.get_data()
	convertImage(imgData)
	x=imread('output.png',mode='L')
	x=np.invert(x)
	x=imresize(x,(28,28))

	imsave('output_conv.png',x)

	x=x.reshape(1,28,28,1)
	out=sess.run(tf.argmax(output,1),feed_dict={tf_input:x})
	print(out)
	##response= np.argmax(out,axis=1)#axis=1 means single dimentional response i.e one string of prediction
	##print(response)
	res=label_bangla_kolkata[int(out)]
	#print (res)
	##print (response)
	return res


if __name__== "__main__":
	port= int(os.environ.get('PORT',2000))
	app.run(host='0.0.0.0',port=port)



