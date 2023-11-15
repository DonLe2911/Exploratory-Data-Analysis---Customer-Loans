import sqlalchemy
from sqlalchemy import create_engine
import yaml
import pandas as pd
import os

def load_credentials():
    with open("EDA_CustomerLoans/credentials.yaml", "r") as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    def __init__(self):
        self.credentials = load_credentials()
        self.engine = self.create_engine()

    def create_engine(self):
        db_url = f"postgresql://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}"
        engine = create_engine(db_url)
        return engine
    
    def extract_data_to_dataframe(self):
        table_name = 'loan_payments'
        query = f"SELECT * FROM {table_name}"
        dataframe = pd.read_sql(query, self.engine)
        return dataframe
        
    def save_to_csv(self, dataframe, local_path, filename):
        full_path = os.path.join(local_path, filename)
        dataframe.to_csv(full_path, index=False)

    def load_from_csv(self, local_path, filename):
        full_path = os.path.join(local_path, filename)
        dataframe = pd.read_csv(full_path)
        return dataframe

connector = RDSDatabaseConnector()
#commenting out next two instantiations as dont need to repeat now that data is saved locally, means this script will run a lot quicker loading the data from local
#data = connector.extract_data_to_dataframe()
#connector.save_to_csv(data, "EDA_CustomerLoans", "loan_payments_data.csv")
loaded_data = connector.load_from_csv("EDA_CustomerLoans", "loan_payments_data.csv")
print(loaded_data)  