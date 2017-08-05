import numpy as np 
import keras.models
from keras.models import model_from_json
from scipy.misc import imread,imresize,imshow
import tensorflow as tf 
import os
import sys
def init():
	json_file= open("C:/Users/bmabi/GitHub/bangla_inception/serve/model/model.json","r")
	loaded_model_json= json_file.read()
	json_file.close()
	loaded_model=model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights('C:/Users/bmabi/GitHub/bangla_inception/serve/model/model.h5')
	print("Loaded Model from disk")

	#compile loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph =tf.get_default_graph()
	print("Graph and Model Loaded")

	return loaded_model,graph