import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/74977/169835/StudentsPerformance.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260514%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260514T145557Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=663779c00c1fdb12997422fc6f0bb891eb67f4497ba21a2a9c79bcc85fa8db6d658d3aa197a13556ff92a9a5cbe7f90df2e037bc821f49d19bdedba02e1a91004e37bb6b0510dfc7d2029a06e0b170e215ab4e482329b4a8d77dce914e476dacfdae7c08ea39fa5acff40c558ac346171001d0c33c37c2eae1cab01b75cc51f1d601fe3e0a7eb35591909c8a4f7c96ea4cec970112ceab85ac538a1cb3c187d6bad35fe4669fff59c190d5597ccecf52162ee75f292b6059fde51865d2a1d37a8f21b21a689e90411c7e99677df8c5e324d249a949c2ffa889b53ac28e00469291b9eba76c4c5b7e5ba4074ee62300106a6c6d00df95defc1ceb0e1c139fa58f") # Added 'r' for raw string
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__": #initiate and run
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



