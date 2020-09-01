import pandas as pd 
import numpy as np
from logs.logger import App_Logger

class data_getter:
	
	def __init__(self,file_object):
		self.logger = App_Logger()
		self.log_file = file_object

	def data_load(self, file):
		self.logger.log(self.log_file, "Entering into DATA  GETTER METHOD")
		'''
	                        Method Name: data_load
	                        Description: This method loads the data from the file and convert into a pandas dataframe
	                        Output: Returns a Dataframes, which is our data for training
	                        On Failure: Raise Exception .
	    '''
		try:
			self.logger.log(self.log_file,"Now we are starting data gathering from the file source")
			data = pd.read_csv(file,na_values='?')
			self.logger.log(self.log_file, "Now we have gathered the data frome the source and converted it into a pandas dataframe")
			return data

		except Exception as e:
			self.logger.log(self.log_file, "oops!!Data gathering not succesful")
			raise e


	        