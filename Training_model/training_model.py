from sklearn.model_selection import train_test_split
import pandas as pd 
from logs.logger import App_Logger
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import xgboost

class model_building:

	log_file = open("./Training_logs/training_model_building_logs.txt","a+")
	def __init__(self,file):
		self.logger = App_Logger()
		self.file = file

	def data_splitting(self):
		log_file = open("./Training_logs/training_model_building_logs.txt", "a+")
		'''
	                        Method Name: data_splitting
	                        Description: This method loads the data and splits it into train and test
	                        Output: Returns training and testing set
	                        On Failure: Raise Exception .
	    '''
		try:
			data = pd.read_csv(self.file)
			self.logger.log(log_file,"Data splitting is now started")
			X = data.drop("price",axis =1)
			Y = data['price']
			x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size  =0.25,random_state=3)
			self.logger.log(log_file,"Data is now splitted into training and test set")
			log_file.close()
			return x_train,y_train,x_test,y_test

		except Exception as e:
			self.logger.log(log_file,"Data splitting is not finished")
			log_file.close()
			raise e



	def randomforest_reg(self,x_train,y_train,x_test,y_test):
		log_file = open("./Training_logs/training_model_building_logs.txt", "a+")
		'''
	                        Method Name: randomforest_reg
	                        Description: This method fits the randomforest regressor on the training data
	                        Output: Returns a Dataframes, which is our data for training
	                        On Failure: Raise Exception .
	    '''
		try:
			self.logger.log(log_file,"Now we will fit RandomForestRegressor in the training set")
			rf  = RandomForestRegressor()
			rf.fit(x_train,y_train)
			ypred1 = rf.predict(x_test)
			self.logger.log(log_file,"RandomForestRegressor is now fitted on to the training set")
			log_file.close()
			return rf.score(x_test,y_test)


		except Exception as e:
			self.logger.log(log_file,"Model fitting randomforest not succesful")
			log_file.close()
			raise e

		


	def xgboost_reg(self,x_train,y_train,x_test,y_test):
		log_file = open("./Training_logs/training_model_building_logs.txt", "a+")
		'''
	                        Method Name: data_load
	                        Description: This method loads the data from the file and convert into a pandas dataframe
	                        Output: Returns a Dataframes, which is our data for training
	                        On Failure: Raise Exception .
	    '''
		try:
			self.logger.log(log_file,"Now we will fit Xgboostregressor in the training set")
			xg  = RandomForestRegressor()
			xg.fit(x_train,y_train)
			ypred1 = xg.predict(x_test)
			self.logger.log(log_file,"Xgboostregressor is now fitted on to the training set")
			log_file.close()
			return xg.score(x_test,y_test)

		except Exception as e:
			self.logger.log(log_file,"Model fitting xgboost not succesful")
			log_file.close()
			raise e
