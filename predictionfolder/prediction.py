from preprocessingfolder  import preprocessingfile
import pandas as pd
import numpy as np
import joblib
from Data_ingestion import data_ingestion

class predict():
	def __init__(self):
		pass

	def predictor(self,file):	
		file_object = open(r'.\Prediction_logs\data_getter_logs.txt','a+')


		instance1 = data_ingestion.data_getter(file_object)
		data = 	instance1.data_load(file)

		file_object = open(r'.\Prediction_logs\preprocessing_logs.txt', 'a+')
		instance2 = preprocessingfile.preprocess(file_object)
		set1 = instance2.set_columns(data)

		remove = instance2.remove_columns(set1)

		type1 = instance2.set_type(remove)

		impute = instance2.imputation(type1)

		feature  = instance2.feature_remove(impute)

		scaled = instance2.scaling(feature)

		encoder = instance2.encoding(scaled)

		model = joblib.load('model.pkl')

		result =  model.predict(encoder)

		encoder['output'] = result

		return encoder