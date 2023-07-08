import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomeException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            if not os.path.exists('artifacts'):
                os.makedirs('artifacts')

            df = pd.read_csv(r'C:\Users\kishu\PycharmProjects\ML Project\notebook\Data\stud.csv')
            print('Read the dataset as a dataframe')
            logging.info('Read the dataset as a dataframe')

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print('Raw data saved:', self.ingestion_config.raw_data_path)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            print('Train data saved:', self.ingestion_config.train_data_path)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            print('Test data saved:', self.ingestion_config.test_data_path)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomeException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_ob_file_path = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


