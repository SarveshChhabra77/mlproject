import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    
    ''' This function is responsibe for data transformation'''
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        try:
            numerical_features=['writing_score','reading_score']
            categorical_feature=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("One Hot Encoding",OneHotEncoder())
                ]
            )
            logging.info(f"Numerical column {numerical_features}")
            logging.info(f"Categorical column {categorical_feature}")
            
            preprocessor=ColumnTransformer(
                [
                    ('numerical pipeline',num_pipeline,numerical_features),
                    ('categorical pipeline',cat_pipeline,categorical_feature)
                ]
            )
            
            return preprocessor
        except Exception as e:
            
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data is completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_obj()
            
            target_column='math_score'
            
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            ## except target feature all are stored
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]
            
            logging.info("Applying preprocessing object on training and test dataset")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
            
            # np.c_ is a NumPy shortcut to concatenate arrays column-wise.
            train_arr= np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr= np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved Preprocessing Object")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                )

            
        except Exception as e:
            raise CustomException(e,sys)
            