import os
import sys
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_mode_file_path=os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting traing and test input data')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models={
                'Linear-Regression':LinearRegression(),
                'Decision-Tree':DecisionTreeRegressor(),
                'Adaboost-Regressor': AdaBoostRegressor(),
                'Random-Forest': RandomForestRegressor(),
                'GradientBoost-Regressor':GradientBoostingRegressor(),
                'K-Nearest-Neighbour': KNeighborsRegressor(),
                'CatBoostRegressor':CatBoostRegressor(verbose=False),
                'XGBRegressor':XGBRegressor(),
            }
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            # to get the best model score from dict
            best_model_score=max(sorted(model_report.values()))

            # to get best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score<.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_mode_file_path,
                obj=best_model)
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
    