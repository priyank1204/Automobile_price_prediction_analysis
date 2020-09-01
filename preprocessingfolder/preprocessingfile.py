import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from logs.logger import App_Logger

class preprocess:
	def __init__(self,file_object):
		self.logger = App_Logger()
		self.log_file = file_object		
		self.logger.log(self.log_file, "Now we are starting the preprocessing of the data")
		

	def set_columns(self,data):
		"""
                        Method Name: set_columns
                        Description: This method Sets the coloumn names for each of the columns 
                        Output: Returns a Dataframes, one in which columns indexes are proper
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "Now firstly we will set the names for each column i.e column index")
		try:
			data.columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style",'drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinder','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg']
			self.logger.log(self.log_file, "COLumn index set for each features succesfully")
			return data

		except Exception as e:
			self.logger.log(self.log_file, "oops!! column index for the columns can not be succesfully set")
			raise e



	def remove_columns(self,data):
		"""
                        Method Name: remove_columns
                        Description: This method removes unncessary columns from the data
                        Output: Returns a Dataframes, one in There are only important features
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "Now  we come to the third step of preprocessing i.e removing unnecessary columns")
		try:
			self.logger.log(self.log_file, "Here we are reomving some unnnecessary columns from the data which are of no use in the model building ")
			useful_data = data[["length","width",'horsepower','curb-weight',"engine-size","city-mpg","highway-mpg",'drive-wheels','num-of-cylinder']]
			self.logger.log(self.log_file, "we have succesfully removed our unnnecessary columns ")		
			return useful_data	

		except Exception as e:
			self.logger.log(self.log_file, "Removal fo unncessary columns was not successful")
			raise e 


	def set_type(self,data):
		"""
                        Method Name: set_type
                        Description: This method set the data type oof each column corectly
                        Output: Returns a Dataframes, one in which there are correct data type of each feature
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "Now we are entering to third preprocessing step i.e setting correct daat type for each feature")
		try:
			self.logger.log(self.log_file, "Here we are setting required data types for each column and then returning correct dataframe")
			data.length = data.length.astype('float')
			data.width = data.width.astype('float')
			data.horsepower = data.horsepower.astype('float')
			data['curb-weight'] = data['curb-weight'].astype('float')
			data['engine-size'] = data['engine-size'].astype('float')
			data['city-mpg'] = data['city-mpg'].astype('float')
			data['highway-mpg'] = data['highway-mpg'].astype('float')
			data['drive-wheels'] = data['drive-wheels'].astype('object')
			data['num-of-cylinder'] = data['num-of-cylinder'].astype('object')
			self.logger.log(self.log_file, "we have succesfully set the correct data type for each column")
			return data

		except Exception as e:
			self.logger.log(self.log_file, "looks like there is some error occured in setting data types for each columns")
			raise e 




	def imputation(self,data):
		"""
                        Method Name: imputation
                        Description: This method removes null or missing values from the dataset
                        Output: Returns a Dataframes, one in which there are no missing values
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "Now we are starting the next step of preprocessing i.e imputation of missing values")
		try:
			self.logger.log(self.log_file, "NOW WE are starting to impute missing values as per reuirements on the columns")
			num_col = data.select_dtypes(include = [np.number]).columns
			cat_col = data.select_dtypes(exclude = [np.number]).columns
			imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
			self.logger.log(self.log_file, "imputing the numerical columns Nan VALUES WITH MEAN")
			imputer.fit(data[num_col])
			data[num_col] = imputer.transform(data[num_col])
			imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
			self.logger.log(self.log_file, "nOW WE ARE IMPUTING THE CATEGORICAL COLUMNS MISSING VALUES WITH MODE")
			data[cat_col] = imputer.fit_transform((data[cat_col]))
			self.logger.log(self.log_file, "IMPUTATION OF MISSING VALUES IS COMPLETED")
			return data

		except Exception as e:
			self.logger.log(self.log_file, "LOOKS LIKE THERE IS SOME ERROR IN IMPUTING ISSING VALUES")
			raise e
			


	def feature_remove(self,data):
		"""
                        Method Name: feature_remove
                        Description: This method removes some columns by replacing them with new columns
                        Output: Returns a Dataframes, one in which columns are added and some are removed inplace of them
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "now we have entered in the step of feature removal or adding")
		try:
			self.logger.log(self.log_file, "Now we will add some featurs new and remove some old features")
			data['area'] = data['length']*data['width']
			data['miles']  = data['city-mpg']-data['highway-mpg']
			self.logger.log(self.log_file, "adding two new features area and miles")
			data.drop('length',inplace = True,axis =1)
			data.drop('width',inplace = True,axis =1)
			data.drop('city-mpg',inplace = True,axis =1)
			data.drop('highway-mpg',inplace = True,axis =1)	
			self.logger.log(self.log_file, "removing four old features on their places")
			arrange_data = data[['horsepower', 'curb-weight', 'engine-size', 'drive-wheels','num-of-cylinder', 'miles', 'area']]
			self.logger.log(self.log_file, "feature engineering completed succesfully")
			return arrange_data

		except Exception as e:
			self.logger.log(self.log_file, "featuree engineering unsuccesful")
			raise e


	def scaling(self,data):
		"""
                        Method Name: scaling
                        Description: This method Scales all the numerical features into a same range 
                        Output: Returns a Dataframes, one in which all the numerical columns are in same range
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "In this step we are gonna scale all numerical features in the same range")
		try:
			self.logger.log(self.log_file, "here we have started scaling the features with MinMaxScaler]")
			num_col = data.select_dtypes(include = [np.number]).columns
			sc = MinMaxScaler()
			data[num_col] = sc.fit_transform(data[num_col])
			data['num-of-cylinder'].replace({"three":"eight","twelve":"eight"},inplace  = True)
			self.logger.log(self.log_file, "Now here we have scaled all the numerical features in the same range")
			return data

		except Exception as e:
			self.logger.log(self.log_file, "oops!!   feature scaling not succesfull")
			raise e

	def encoding(self,data):
		"""
                        Method Name: encoding
                        Description: This method encodes the categorical features into numerical for machine learning algortihms 
                        Output: Returns a Dataframes, one in encoded columns for categorical columns are introduced
                        On Failure: Raise Exception .
        """
		self.logger.log(self.log_file, "Now it is the end step of preprocessing i.e encoding categorical variables")
		try:
			self.logger.log(self.log_file, "here we are using dummy variables  function for encoding categorical features")
			encoded_data = pd.get_dummies(data,drop_first=True)
			self.logger.log(self.log_file, "encoding categorical feature done succesfully")
			
			return encoded_data

		except Exception as e:	
			self.logger.log(self.log_file, "oops!! encoding categorical features can not be succesfully done")
			raise e

		