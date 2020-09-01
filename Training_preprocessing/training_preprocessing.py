import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from logs.logger import App_Logger
from sklearn.impute import SimpleImputer

class preprocess:

	def __init__(self,file):
		self.logger = App_Logger()
		self.file = file

	def gather(self):
		log_file = open(r'C:\Users\poorvi\Desktop\auto_project\Training_logs\training_preprocessing_logs.txt', "a+")
		try:
			self.logger.log(log_file,"DATA is being gathered ")

			auto_data = pd.read_csv(self.file,header = None,na_values="?")
			self.logger.log(log_file,"DATA gathering completed ")


			auto_data.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style",'drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinder','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
			self.logger.log(log_file,"columns for data set has been set")
			log_file.close()
			return auto_data	

		except Exception as e:
			self.logger.log(log_file,"Files gathering is not succesful")
			log_file.close()
			raise e 

		


	def set_types(self,auto_data):
		log_file = open(r'C:\Users\poorvi\Desktop\auto_project\Training_logs\training_preprocessing_logs.txt', "a+")
		try:

			log_file = open("./Training_logs/preprocessing_logs.txt","a+")
			self.logger.log(log_file,"Now we  will set the types of data into required")


			auto_data["normalized-losses"] = auto_data["normalized-losses"].astype("float")
			auto_data["bore"] = auto_data["bore"].astype("float")
			auto_data["stroke"] = auto_data["stroke"].astype("float")
			auto_data["horsepower"] = auto_data["horsepower"].astype("float")
			auto_data["peak-rpm"] = auto_data["peak-rpm"].astype("float")
			auto_data["price"] = auto_data["price"].astype("float")

			self.logger.log(log_file,"DATA Types has been set for each feature")
			log_file.close()
			return auto_data



		except Exception as e:
			self.logger.log(log_file,"setting data types was not completed")
			log_file.close()
			raise e


		
		

	def imputation(self,auto_data):
		log_file = open(r'C:\Users\poorvi\Desktop\auto_project\Training_logs\training_preprocessing_logs.txt', "a+")
		try:

			self.logger.log(log_file,"Now we  will remove the missing values from the data")


			num_col = auto_data.select_dtypes(include = [np.number]).columns
			num_col.drop("price")
			imputer = SimpleImputer()
			imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
			imputer.fit(auto_data[num_col])
			auto_data[num_col] = imputer.transform(auto_data[num_col])

			self.logger.log(log_file,"missing values imputation for numerical data is done ,,,,, Now we  will handle the target variable")

			auto_data.dropna(subset=["price"],axis = 0,inplace=True)
			auto_data.reset_index(drop =True,inplace = True)


			cat_col = auto_data.select_dtypes(exclude = [np.number]).columns
			imputer = SimpleImputer()
			imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
			imputer.fit(auto_data[cat_col])
			auto_data[cat_col] = imputer.transform(auto_data[cat_col])
			self.logger.log(log_file,"Imputations of missing values is done----")
			auto_data.to_csv(r"C:\Users\poorvi\Desktop\auto_project\Training_preprocessing\preprocessed_file.csv")
			log_file.close()

		except Exception as e:
			self.logger.log(log_file,"Imputation of missing values failed")
			log_file.close()
			raise e
	
			



	

