import os 
import sys 
 
from src.excep.exception import CustomException
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.log.logger import logging
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('data ingestion strted')

        try:
            df=pd.read_csv('D:\MLops-Recommendation-Project\dataset\Sales_Amazon_Cleaned_final.csv')
            user_id_counts=df['user_id'].value_counts()
            unique_user_ids=user_id_counts[user_id_counts==1].index.tolist()
            df=df[~df['user_id'].isin(unique_user_ids)]
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test inittaed')

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=0)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('data ingestion and splituing done')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ =="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
