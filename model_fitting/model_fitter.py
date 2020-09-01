import joblib
from sklearn.ensemble import RandomForestRegressor
#Randomized Search CV
import numpy as np
from logs.logger import App_Logger
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from preprocessingfolder import preprocessingfile

class model_fit:
	def __init__(self,data,file_object):
		self.log_file = file_object
		self.data = data
		self.logger = App_Logger()


	def training(self):
		'''
			                        Method Name: training
			                        Description: This method TRAINS TEH PREPROCESSED DATA FOR THE BEST MODEL
			                        Output: Returns a best model for predictions
			                        On Failure: Raise Exception .
		'''
		try:
			self.logger.log(self.log_file, "Entering into training method ")
			self.logger.log(self.log_file, "Now we willl firstly split the data into training and testing set")
			X=self.data.drop('price',axis=1)
			Y = self.data['price']
			x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state  =3)
			self.logger.log(self.log_file, "Dataset splitting succesfully done")

			##fitting with random forest regressor()
			self.logger.log(self.log_file, "Now we will fit the randomforestregressor on the training and test set")
			rf = RandomForestRegressor()
			rf.fit(x_train,y_train)
			self.logger.log(self.log_file, "Randomforestregressr fitted succesfully on the training set")

			##Now applying tuning on the randomforestregressor
			self.logger.log(self.log_file, "Now we will perfrom hyperparameter tuning on the randomforestregressor for better  results")
			self.logger.log(self.log_file, "Now we are setting best paramterers range")
			# Number of trees in random forest
			n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
			# Number of features to consider at every split
			max_features = ['auto', 'sqrt']
			# Maximum number of levels in tree
			max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
			# max_depth.append(None)
			# Minimum number of samples required to split a node
			min_samples_split = [2, 5, 10, 15, 100]
			# Minimum number of samples required at each leaf node
			min_samples_leaf = [1, 2, 5, 10]
			self.logger.log(self.log_file, "Best parameters ranged succesfullly")
			random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf}
			rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error',
			 n_iter = 10, cv = 5, random_state=42, n_jobs = 1)
			self.logger.log(self.log_file, "Randomized search cv done on randomforestregressor")
			rf_random.fit(x_train,y_train)
			self.logger.log(self.log_file, "fitting model with best parameters on the training set")
			joblib.dump(rf_random,'model.pkl')
			self.logger.log(self.log_file, "saving the best model")
			self.log_file.close()
		except Exception as e:
			self.logger.log(self.log_file, "looks like there is some error in model training !!!try with removing errors")
			self.log_file.close()
			raise e







