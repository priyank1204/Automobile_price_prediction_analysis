from retraining_preprocessing  import retraining_preprocessing
import pandas as pd
import numpy as np
import joblib
from Data_ingestion import data_ingestion
from model_fitting import model_fitter


class retraining:
	def __init__(self,file):
		self.file = file	
		

	def retrain_model(self):
		file_object = open(r'.\Retraining_logs\retraining_logs.txt','a+')
		instance1 = data_ingestion.data_getter(file_object)
		data = 	instance1.data_load(self.file)

		instance2 = retraining_preprocessing.preprocess(file_object)

		set1 = instance2.set_columns(data)

		target =instance2.target(set1)

		remove = instance2.remove_columns(target)

		type1 = instance2.set_type(remove)

		impute = instance2.imputation(type1)

		feature  = instance2.feature_remove(impute)

		scaled = instance2.scaling(feature)

		encoder = instance2.encoding(scaled)

		result = model_fitter.model_fit(encoder,file_object)

		result.training()






