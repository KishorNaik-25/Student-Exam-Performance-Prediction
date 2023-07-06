import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object

import os

@dataclass()
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_obj(self):

        '''
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ['reading_score','writing_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")), # for treating missing values in numerical
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")), # for treating missing values in categorical
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Columns:,{numerical_columns}")
            logging.info(f"Categorical Columns:,{categorical_columns}")

            preprocessor = ColumnTransformer(                               # Combaining numerical pipeline and categorical pipeline
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data is completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "math_score"
            numerical_columns = ['reading_score','writing_score']

            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feture_train_df = train_df[target_column_name]

            input_fetaures_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_fetaures_test_df)

            train_arc = np.c_[input_feature_train_arr,np.array(target_feture_train_df)]  # find y we are using np.c_
            test_arc = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocesinng Object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arc,
                test_arc,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomeException(e, sys)






