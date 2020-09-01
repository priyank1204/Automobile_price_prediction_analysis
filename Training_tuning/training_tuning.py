import pandas as pd 
from logs.logger import App_Logger
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import xgboost
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

class tuning:

	def __init__(self):
		self.logger = App_Logger()

	def tuning_xgboost(self,x_train,y_train,x_test,y_test):
		log_file = open(r"./Training_logs/training_model_tuning_logs.txt", "a+")
		try: 
			self.logger.log(log_file,"Nowe will tune the xgboost regressor with GridSearchCV")
			xgr = xgboost()
			self.logger.log(log_file,"Now setting parameter range")
			params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}	
			self.logger.log(log_file,"Estimating the best parameters for xgboost")
			grid = GridSearchCV(xgr, params)
			self.logger.log(log_file,"Best parameters estimation succesful")
			grid.fit(x_train,y_train)
			self.logger.log(log_file,"NOW fitting tuned model on the training set")
			y = grid.best_estimator_
			log_file.close()
			return y.score(x_test,y_test)

		except Exception as e:
			self.logger.log(log_file,"TUNING xgboost not succesful")
			log_file.close()
			raise e

		


	def tuning_rf(self,x_train,y_train,x_test,y_test):
		log_file = open(r"./Training_logs/training_model_tuning_logs.txt", "a+")
		try: 
			self.logger.log(log_file,"Nowe will tune the randomforest regressor with RandomizedSearchCV")
			rf = RandomForestRegressor()
			self.logger.log(log_file,"Now setting parameter range")
			random_grid ={'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], 'max_features': ['auto', 'sqrt'], 'max_depth': [5, 10, 15, 20, 25, 30], 'min_samples_split': [2, 5, 10, 15, 100], 'min_samples_leaf': [1, 2, 5, 10]}
			self.logger.log(log_file,"Estimating the best parameters for randomforest")
			rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, random_state=42, n_jobs = 1)
			self.logger.log(log_file,"Best parameters estimation succesful")
			self.logger.log(log_file,"NOW fitting tuned model on the training set")
			rf_random.fit(x_train,y_train)
			joblib.dump(rf_random,r"C:\Users\poorvi\Desktop\auto_project\model.pkl")
			log_file.close()

		except Exception as e:
			self.logger.log(log_file,"TUNING xgboost not succesful")
			log_file.close()
			raise e
