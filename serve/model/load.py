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

def init_bangla(modelName):
	graph=tf.Graph()
	with graph.as_default():
		sess=tf.Session()
		with sess.as_default():
			if(modelName=="kolkata"):
				#Loading the meta graph and restore weights
				saver=tf.train.import_meta_graph('C:/Users/bmabi/GitHub/bangla_inception/serve/model/bangla/banglaInceptionM2_banglaKolkata.ckpt-30000.meta')
				#saver.restore(sess,tf.train.latest_checkpoint('C:/Users/bmabi/GitHub/bangla_inception/serve/model/'))
				saver.restore(sess,'C:/Users/bmabi/GitHub/bangla_inception/serve/model/bangla/banglaInceptionM2_banglaKolkata.ckpt-30000')
			else:
				#Loading the meta graph and restore weights
				saver=tf.train.import_meta_graph('C:/Users/bmabi/GitHub/bangla_inception/serve/model/bangla/banglaInception_mini_M2.ckpt-30000.meta')
				#saver.restore(sess,tf.train.latest_checkpoint('C:/Users/bmabi/GitHub/bangla_inception/serve/model/'))
				saver.restore(sess,'C:/Users/bmabi/GitHub/bangla_inception/serve/model/bangla/banglaInception_mini_M2.ckpt-30000')
			tf_input = graph.get_operation_by_name("tf_input").outputs[0]
			output=graph.get_operation_by_name("output").outputs[0]
	return sess,tf_input,output


